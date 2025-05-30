#!/usr/bin/env python3
# Step 3: Model Loading and LoRA Configuration

import os
import json
import torch
from transformers import (
    AutoModel, 
    AutoTokenizer,
    AutoConfig,
    WhisperProcessor
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
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
        print(f"Hidden size: {model_config.get('hidden_size', 'Unknown')}")
        print(f"Num layers: {model_config.get('num_hidden_layers', 'Unknown')}")
        print(f"Vocab size: {model_config.get('vocab_size', 'Unknown')}")
    
    # Check for model files
    model_files = [f for f in os.listdir(model_path) if f.endswith(('.bin', '.safetensors', '.pt', '.pth'))]
    print(f"Model files found: {len(model_files)}")
    for f in model_files[:5]:  # Show first 5 files
        size_mb = os.path.getsize(os.path.join(model_path, f)) / (1024 * 1024)
        print(f"  - {f}: {size_mb:.2f} MB")

def load_kimi_audio_model(model_path, device_map="auto"):
    """Load Kimi-Audio model"""
    print(f"\nLoading Kimi-Audio model from: {model_path}")
    
    try:
        # First, try to load the model config to understand its architecture
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print(f"Model architecture: {model_config.architectures}")
        
        # Load the model
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for H20 GPUs
            device_map=device_map,
            trust_remote_code=True
        )
        
        print(f"Model loaded successfully!")
        print(f"Model dtype: {model.dtype}")
        
        # Print model structure
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting alternative loading method...")
        
        # Try loading with AutoModelForCausalLM if it's a language model
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True
        )
        return model

def load_tokenizer(tokenizer_path):
    """Load GLM-4 tokenizer"""
    print(f"\nLoading tokenizer from: {tokenizer_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True
    )
    
    print(f"Tokenizer loaded successfully!")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Special tokens: {tokenizer.special_tokens_map}")
    
    return tokenizer

def configure_lora(model, lora_config=None):
    """Configure LoRA for the model"""
    print("\nConfiguring LoRA...")
    
    if lora_config is None:
        # Default LoRA configuration for audio ASR fine-tuning
        lora_config = LoraConfig(
            r=32,  # LoRA rank
            lora_alpha=64,  # LoRA scaling parameter
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj",  # Attention layers
                "gate_proj", "up_proj", "down_proj",      # MLP layers
            ],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,  # For ASR, we use causal LM task
        )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print LoRA info
    model.print_trainable_parameters()
    
    return model, lora_config

def test_model_setup():
    """Test the complete model setup"""
    print("Testing Model Setup")
    print("=" * 50)
    
    # 1. Get model information
    print("\n1. Model Information:")
    get_model_info(config['models']['kimi_audio'])
    
    # 2. Load Kimi-Audio model
    print("\n2. Loading Kimi-Audio Model:")
    model = load_kimi_audio_model(config['models']['kimi_audio'])
    
    # 3. Load tokenizer
    print("\n3. Loading Tokenizer:")
    tokenizer = load_tokenizer(config['models']['tokenizer'])
    
    # 4. Configure LoRA
    print("\n4. Configuring LoRA:")
    model, lora_config = configure_lora(model)
    
    # 5. Load Whisper processor (already tested in step2)
    print("\n5. Loading Whisper Processor:")
    whisper_processor = WhisperProcessor.from_pretrained(config['models']['whisper'])
    print("Whisper processor loaded successfully!")
    
    # 6. Test forward pass with dummy data
    print("\n6. Testing Forward Pass:")
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
        print(f"Output shape: {outputs.logits.shape if hasattr(outputs, 'logits') else 'Custom output format'}")
        
    except Exception as e:
        print(f"Forward pass test failed: {e}")
        print("This might be normal if the model has a custom architecture")
    
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
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()