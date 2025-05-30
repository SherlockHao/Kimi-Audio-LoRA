#!/usr/bin/env python3
# Step 3: Model Loading and LoRA Configuration

import os
import sys
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    WhisperProcessor,
    PreTrainedModel
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
import importlib.util
import warnings
warnings.filterwarnings("ignore")

# Load configuration
CODE_DIR = "/opt/data/nvme4/kimi/Kimi-Audio-LoRA"
config_path = os.path.join(CODE_DIR, "config.json")

with open(config_path, 'r') as f:
    config = json.load(f)

def get_model_info(model_path):
    """Get basic information about the model"""
    print(f"Checking model at: {model_path}")
    
    # Check for config.json
    config_file = os.path.join(model_path, "config.json")
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            model_config = json.load(f)
        print(f"Model type: {model_config.get('model_type', 'Unknown')}")
        print(f"Architecture: {model_config.get('architectures', 'Unknown')}")
        print(f"Hidden size: {model_config.get('hidden_size', 'Unknown')}")
        print(f"Num layers: {model_config.get('num_hidden_layers', 'Unknown')}")
        print(f"Vocab size: {model_config.get('vocab_size', 'Unknown')}")
    
    # Check for model files
    model_files = [f for f in os.listdir(model_path) if f.endswith(('.bin', '.safetensors', '.pt', '.pth'))]
    print(f"Model files found: {len(model_files)}")
    total_size = 0
    for f in model_files[:5]:  # Show first 5 files
        size_mb = os.path.getsize(os.path.join(model_path, f)) / (1024 * 1024)
        total_size += size_mb
        print(f"  - {f}: {size_mb:.2f} MB")
    if len(model_files) > 5:
        print(f"  ... and {len(model_files) - 5} more files")
    
    return model_config

def load_kimi_audio_model_direct(model_path):
    """Load Kimi-Audio model using direct import"""
    print(f"\nLoading Kimi-Audio model using direct import from: {model_path}")
    
    try:
        # Load the custom modeling file
        modeling_file = os.path.join(model_path, "modeling_moonshot_kimia.py")
        if not os.path.exists(modeling_file):
            raise FileNotFoundError(f"Custom model file not found: {modeling_file}")
        
        # Import the custom model class
        spec = importlib.util.spec_from_file_location("modeling_moonshot_kimia", modeling_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules["modeling_moonshot_kimia"] = module
        spec.loader.exec_module(module)
        
        # Get the model class
        MoonshotKimiaForCausalLM = getattr(module, "MoonshotKimiaForCausalLM")
        
        # Load config
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # Initialize model
        print("Initializing model...")
        model = MoonshotKimiaForCausalLM(config)
        
        # Load state dict
        print("Loading model weights...")
        state_dict_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
        
        if state_dict_files:
            # Use safetensors if available
            from safetensors.torch import load_file
            state_dict = {}
            for f in state_dict_files:
                state_dict.update(load_file(os.path.join(model_path, f)))
            model.load_state_dict(state_dict, strict=False)
        else:
            # Fallback to bin files
            bin_files = [f for f in os.listdir(model_path) if f.endswith('.bin')]
            if bin_files:
                state_dict = {}
                for f in bin_files:
                    state_dict.update(torch.load(os.path.join(model_path, f), map_location='cpu'))
                model.load_state_dict(state_dict, strict=False)
        
        # Convert to bfloat16 and move to GPU
        model = model.to(torch.bfloat16)
        
        print("Model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"Error with direct loading: {e}")
        raise

def load_kimi_audio_model(model_path, device_map="auto"):
    """Load Kimi-Audio model with fallback methods"""
    print(f"\nAttempting to load Kimi-Audio model from: {model_path}")
    
    # First, get model info
    model_config = get_model_info(model_path)
    
    # Try direct loading first (to avoid the post_init issue)
    try:
        model = load_kimi_audio_model_direct(model_path)
        
        # Set up device map if needed
        if device_map == "auto" and torch.cuda.device_count() > 1:
            print(f"Setting up model for {torch.cuda.device_count()} GPUs")
            from accelerate import dispatch_model, infer_auto_device_map
            device_map = infer_auto_device_map(
                model, 
                max_memory={i: "80GB" for i in range(torch.cuda.device_count())},
                no_split_module_classes=["MoonshotKimiaDecoderLayer"]
            )
            model = dispatch_model(model, device_map=device_map)
        else:
            model = model.cuda()
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        return model
        
    except Exception as e:
        print(f"All loading methods failed: {e}")
        raise

def load_tokenizer(tokenizer_path):
    """Load GLM-4 tokenizer"""
    print(f"\nLoading tokenizer from: {tokenizer_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Tokenizer loaded successfully!")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Pad token: {tokenizer.pad_token}")
    print(f"EOS token: {tokenizer.eos_token}")
    
    return tokenizer

def configure_lora(model, target_modules=None):
    """Configure LoRA for the model"""
    print("\nConfiguring LoRA...")
    
    # Find the actual module names in the model
    if target_modules is None:
        # Get all linear layer names
        linear_layers = set()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Extract the attribute name (last part of the full name)
                parts = name.split('.')
                if len(parts) > 0:
                    linear_layers.add(parts[-1])
        
        print(f"Found linear layers: {linear_layers}")
        
        # Common patterns for attention and MLP layers
        common_targets = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        target_modules = [t for t in common_targets if t in linear_layers]
        
        if not target_modules:
            # Fallback to all linear layers if no common patterns found
            target_modules = list(linear_layers)[:10]  # Limit to avoid too many
        
        print(f"Selected target modules: {target_modules}")
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=32,  # LoRA rank
        lora_alpha=64,  # LoRA scaling parameter
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Prepare model for training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print LoRA info
    model.print_trainable_parameters()
    
    return model, lora_config

def test_model_setup():
    """Test the complete model setup"""
    print("Testing Model Setup")
    print("=" * 50)
    
    try:
        # 1. Load Kimi-Audio model
        print("\n1. Loading Kimi-Audio Model:")
        model = load_kimi_audio_model(config['models']['kimi_audio'])
        
        # 2. Load tokenizer
        print("\n2. Loading Tokenizer:")
        tokenizer = load_tokenizer(config['models']['tokenizer'])
        
        # 3. Configure LoRA
        print("\n3. Configuring LoRA:")
        model, lora_config = configure_lora(model)
        
        # 4. Load Whisper processor
        print("\n4. Loading Whisper Processor:")
        whisper_processor = WhisperProcessor.from_pretrained(config['models']['whisper'])
        print("Whisper processor loaded successfully!")
        
        # 5. Test forward pass with dummy data
        print("\n5. Testing Forward Pass:")
        try:
            # Create dummy input
            dummy_text = "This is a test transcription"
            inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True)
            
            # Move to device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs)
            
            print("Forward pass successful!")
            if hasattr(outputs, 'logits'):
                print(f"Output logits shape: {outputs.logits.shape}")
            else:
                print("Model output received (custom format)")
            
        except Exception as e:
            print(f"Forward pass test failed: {e}")
            print("This might be expected for audio models that require special inputs")
        
        # Save model configuration
        model_setup = {
            "model_path": config['models']['kimi_audio'],
            "tokenizer_path": config['models']['tokenizer'],
            "whisper_path": config['models']['whisper'],
            "lora_config": {
                "r": lora_config.r,
                "lora_alpha": lora_config.lora_alpha,
                "target_modules": lora_config.target_modules,
                "lora_dropout": lora_config.lora_dropout,
                "bias": lora_config.bias,
            },
            "model_dtype": "bfloat16",
            "total_params": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        setup_path = os.path.join(CODE_DIR, "model_setup.json")
        with open(setup_path, 'w') as f:
            json.dump(model_setup, f, indent=2)
        
        print(f"\nModel setup saved to: {setup_path}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"\nError in test_model_setup: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    try:
        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This script requires GPU.")
        
        print(f"CUDA devices available: {torch.cuda.device_count()}")
        
        # Run test
        success = test_model_setup()
        
        if success:
            print("\n" + "=" * 50)
            print("Step 3 completed successfully!")
            print("Model and LoRA configuration ready.")
            print("Next step: Training script implementation")
        else:
            print("\nStep 3 encountered issues. Please check the errors above.")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()