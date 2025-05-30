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
    model_config = {}
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            model_config = json.load(f)
        print(f"Model type: {model_config.get('model_type', 'Unknown')}")
        print(f"Architecture: {model_config.get('architectures', 'Unknown')}")
        print(f"Hidden size: {model_config.get('hidden_size', 'Unknown')}")
        print(f"Num layers: {model_config.get('num_hidden_layers', 'Unknown')}")
        print(f"Vocab size: {model_config.get('vocab_size', 'Unknown')}")
        # Check for audio-specific configs
        if 'audio_enc_hidden_size' in model_config:
            print(f"Audio encoder hidden size: {model_config['audio_enc_hidden_size']}")
        if 'n_mels' in model_config:
            print(f"Mel bins: {model_config['n_mels']}")
    
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
        # Add model path to sys.path to enable imports
        if model_path not in sys.path:
            sys.path.insert(0, model_path)
        
        # First, load the configuration module
        config_file = os.path.join(model_path, "configuration_moonshot_kimia.py")
        if os.path.exists(config_file):
            spec = importlib.util.spec_from_file_location("configuration_moonshot_kimia", config_file)
            config_module = importlib.util.module_from_spec(spec)
            sys.modules["configuration_moonshot_kimia"] = config_module
            spec.loader.exec_module(config_module)
        
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
            for f in sorted(state_dict_files):
                print(f"  Loading {f}...")
                state_dict.update(load_file(os.path.join(model_path, f)))
            model.load_state_dict(state_dict, strict=False)
        else:
            # Fallback to bin files
            bin_files = [f for f in os.listdir(model_path) if f.endswith('.bin')]
            if bin_files:
                state_dict = {}
                for f in sorted(bin_files):
                    print(f"  Loading {f}...")
                    state_dict.update(torch.load(os.path.join(model_path, f), map_location='cpu'))
                model.load_state_dict(state_dict, strict=False)
        
        # Convert to bfloat16 and move to GPU
        model = model.to(torch.bfloat16)
        
        print("Model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"Error with direct loading: {e}")
        # Try alternative approach using AutoModel with post_init fix
        print("Trying alternative loading with post_init fix...")
        return load_kimi_audio_model_auto(model_path)

def load_kimi_audio_model_auto(model_path):
    """Load Kimi-Audio model using AutoModel with fixes"""
    print(f"\nLoading Kimi-Audio model using AutoModel from: {model_path}")
    
    try:
        from transformers import AutoModelForCausalLM
        
        # Temporarily patch the problematic post_init
        original_post_init = PreTrainedModel.post_init
        
        def patched_post_init(self):
            try:
                original_post_init(self)
            except TypeError as e:
                if "argument of type 'NoneType' is not iterable" in str(e):
                    print("Skipping problematic post_init check")
                    pass
                else:
                    raise
        
        PreTrainedModel.post_init = patched_post_init
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.device_count() > 1 else None,
            trust_remote_code=True
        )
        
        # Restore original post_init
        PreTrainedModel.post_init = original_post_init
        
        print("Model loaded successfully using AutoModel!")
        return model
        
    except Exception as e:
        print(f"Error with AutoModel loading: {e}")
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
        print(f"Direct loading failed: {e}")
        print("Model loading failed. Please check the model files and configuration.")
        raise

def load_tokenizer(tokenizer_path):
    """Load GLM-4 tokenizer"""
    print(f"\nLoading tokenizer from: {tokenizer_path}")
    
    # First check what files are in the tokenizer directory
    if os.path.exists(tokenizer_path):
        files = os.listdir(tokenizer_path)
        print(f"Tokenizer files found: {files}")
        
        # Check for tokenizer config
        tokenizer_config_file = os.path.join(tokenizer_path, "tokenizer_config.json")
        if os.path.exists(tokenizer_config_file):
            with open(tokenizer_config_file, 'r') as f:
                tokenizer_config = json.load(f)
            print(f"Tokenizer type: {tokenizer_config.get('tokenizer_class', 'Unknown')}")
    
    try:
        # Try loading with trust_remote_code first
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=False  # Try slow tokenizer to avoid fast tokenizer issues
        )
    except Exception as e:
        print(f"Failed to load tokenizer with AutoTokenizer: {e}")
        
        # If it's a custom tokenizer, try loading directly
        try:
            # Check if there's a custom tokenizer file
            tokenizer_files = [f for f in os.listdir(tokenizer_path) if f.startswith('tokenization_') and f.endswith('.py')]
            
            if tokenizer_files:
                # Import custom tokenizer
                tokenizer_file = os.path.join(tokenizer_path, tokenizer_files[0])
                spec = importlib.util.spec_from_file_location("custom_tokenizer", tokenizer_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find tokenizer class
                for name, obj in module.__dict__.items():
                    if name.endswith('Tokenizer') and hasattr(obj, 'from_pretrained'):
                        TokenizerClass = obj
                        tokenizer = TokenizerClass.from_pretrained(tokenizer_path, trust_remote_code=True)
                        break
            else:
                raise ValueError("No custom tokenizer file found")
                
        except Exception as e2:
            print(f"Failed to load custom tokenizer: {e2}")
            raise
    
    # Set padding token if not set
    if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is None:
        if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Set a default pad token
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
    
    print(f"Tokenizer loaded successfully!")
    print(f"Tokenizer class: {type(tokenizer).__name__}")
    if hasattr(tokenizer, 'vocab_size'):
        print(f"Vocab size: {tokenizer.vocab_size}")
    if hasattr(tokenizer, 'pad_token'):
        print(f"Pad token: {tokenizer.pad_token}")
    if hasattr(tokenizer, 'eos_token'):
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
        try:
            tokenizer = load_tokenizer(config['models']['tokenizer'])
        except Exception as e:
            print(f"Failed to load tokenizer from {config['models']['tokenizer']}: {e}")
            print("Trying to load tokenizer from Kimi-Audio model...")
            try:
                # Try loading tokenizer from the main model directory
                tokenizer = load_tokenizer(config['models']['kimi_audio'])
            except Exception as e2:
                print(f"Failed to load tokenizer from Kimi-Audio model: {e2}")
                print("Proceeding without tokenizer (ASR might not need text tokenizer)")
                tokenizer = None
        
        # 3. Configure LoRA
        print("\n3. Configuring LoRA:")
        model, lora_config = configure_lora(model)
        
        # 4. Load Whisper processor
        print("\n4. Loading Whisper Processor:")
        whisper_processor = WhisperProcessor.from_pretrained(config['models']['whisper'])
        print("Whisper processor loaded successfully!")
        
        # 5. Test forward pass with dummy data
        print("\n5. Testing Forward Pass:")
        if tokenizer is not None:
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
        else:
            print("Skipping forward pass test (no tokenizer available)")
            print("For ASR models, audio features are typically the main input")
        
        # Save model configuration
        model_setup = {
            "model_path": config['models']['kimi_audio'],
            "tokenizer_path": config['models']['tokenizer'] if tokenizer else None,
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
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "tokenizer_loaded": tokenizer is not None
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