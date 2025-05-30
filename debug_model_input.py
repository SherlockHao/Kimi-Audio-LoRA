#!/usr/bin/env python3
# Debug script to understand Kimi-Audio model's input format

import os
import sys
import json
import torch
import numpy as np

# Add the project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from step2_dataloader import create_dataloaders
from step3_model_lora import load_kimi_audio_model, load_tokenizer

# Load configuration
CODE_DIR = "/opt/data/nvme4/kimi/Kimi-Audio-LoRA"
config_path = os.path.join(CODE_DIR, "config.json")

with open(config_path, 'r') as f:
    config = json.load(f)

def inspect_model_architecture(model):
    """Inspect model architecture to understand input requirements"""
    print("\n=== Model Architecture Inspection ===")
    
    # Print model type
    print(f"Model type: {type(model)}")
    print(f"Model class name: {model.__class__.__name__}")
    
    # Check for specific attributes
    attributes_to_check = [
        'audio_encoder', 'encoder', 'decoder', 'lm_head',
        'audio_projection', 'embed_tokens', 'embed_audio',
        'audio_enc_hidden_size', 'hidden_size', 'config'
    ]
    
    print("\nModel attributes:")
    for attr in attributes_to_check:
        if hasattr(model, attr):
            print(f"  - {attr}: {type(getattr(model, attr))}")
            if attr == 'config':
                config_obj = getattr(model, attr)
                if hasattr(config_obj, 'audio_enc_hidden_size'):
                    print(f"    - audio_enc_hidden_size: {config_obj.audio_enc_hidden_size}")
                if hasattr(config_obj, 'hidden_size'):
                    print(f"    - hidden_size: {config_obj.hidden_size}")
                if hasattr(config_obj, 'n_mels'):
                    print(f"    - n_mels: {config_obj.n_mels}")
    
    # Check model's forward signature
    print("\nModel forward method:")
    if hasattr(model, 'forward'):
        import inspect
        try:
            sig = inspect.signature(model.forward)
            print(f"  Signature: {sig}")
        except:
            print("  Could not get signature")
    
    # List all modules
    print("\nModel modules (first 10):")
    for i, (name, module) in enumerate(model.named_modules()):
        if i < 10 and name:
            print(f"  - {name}: {type(module)}")

def test_model_inputs(model, dataloader, tokenizer):
    """Test different input formats to understand what the model expects"""
    print("\n=== Testing Model Inputs ===")
    
    # Get one batch
    batch = next(iter(dataloader))
    audio_features = batch['input_features'].cuda()
    texts = batch['texts']
    
    print(f"\nOriginal audio features shape: {audio_features.shape}")
    print(f"Sample text: {texts[0][:50]}...")
    
    # Test 1: Direct audio features
    print("\nTest 1: Direct audio features")
    try:
        with torch.no_grad():
            outputs = model(audio_features)
        print("  Success! Output type:", type(outputs))
    except Exception as e:
        print(f"  Failed: {e}")
    
    # Test 2: Audio features with different shapes
    print("\nTest 2: Adjusted audio features (3000 time steps)")
    try:
        # Adjust to 3000 time steps
        adjusted_features = audio_features[:, :, :3000]
        if adjusted_features.shape[-1] < 3000:
            padding = 3000 - adjusted_features.shape[-1]
            adjusted_features = torch.nn.functional.pad(adjusted_features, (0, padding))
        
        print(f"  Adjusted shape: {adjusted_features.shape}")
        with torch.no_grad():
            outputs = model(adjusted_features)
        print("  Success! Output type:", type(outputs))
    except Exception as e:
        print(f"  Failed: {e}")
    
    # Test 3: As input embeddings
    print("\nTest 3: As input embeddings (transposed)")
    try:
        # Transpose to [batch_size, time_steps, n_mels]
        embeddings = audio_features.transpose(1, 2)[:, :3000, :]
        print(f"  Embeddings shape: {embeddings.shape}")
        with torch.no_grad():
            outputs = model(inputs_embeds=embeddings)
        print("  Success! Output type:", type(outputs))
    except Exception as e:
        print(f"  Failed: {e}")
    
    # Test 4: With tokenized text
    if tokenizer is not None:
        print("\nTest 4: With tokenized text labels")
        try:
            tokenized = tokenizer(texts[0], return_tensors="pt").to('cuda')
            print(f"  Tokenized input_ids shape: {tokenized.input_ids.shape}")
            with torch.no_grad():
                outputs = model(**tokenized)
            print("  Success with text only!")
        except Exception as e:
            print(f"  Failed: {e}")
    
    # Test 5: Check if model has audio-specific methods
    print("\nTest 5: Audio-specific methods")
    audio_methods = ['forward_audio', 'encode_audio', 'process_audio', 'embed_audio']
    for method in audio_methods:
        if hasattr(model, method):
            print(f"  Found method: {method}")
            try:
                method_func = getattr(model, method)
                with torch.no_grad():
                    result = method_func(audio_features[:, :, :3000])
                print(f"    Success! Result type: {type(result)}")
                if hasattr(result, 'shape'):
                    print(f"    Result shape: {result.shape}")
            except Exception as e:
                print(f"    Failed: {e}")

def main():
    print("Kimi-Audio Model Input Format Debugger")
    print("=" * 50)
    
    try:
        # Load model
        print("\nLoading model...")
        model = load_kimi_audio_model(config['models']['kimi_audio'], device_map=None)
        model = model.cuda()
        model.eval()
        
        # Load tokenizer
        print("\nLoading tokenizer...")
        try:
            tokenizer = load_tokenizer(config['models']['tokenizer'])
        except:
            print("Could not load tokenizer")
            tokenizer = None
        
        # Create dataloader
        print("\nCreating dataloader...")
        dataloader, _ = create_dataloaders(
            train_metadata=config['train_metadata'],
            whisper_path=config['models']['whisper'],
            batch_size=1,
            num_workers=0,
            max_audio_length=30.0
        )
        
        # Inspect model
        inspect_model_architecture(model)
        
        # Test inputs
        test_model_inputs(model, dataloader, tokenizer)
        
        print("\n" + "=" * 50)
        print("Debug completed!")
        print("\nRecommendations:")
        print("1. Check the model's config.json for audio-specific parameters")
        print("2. Look for audio encoder modules in the model")
        print("3. Adjust audio feature dimensions to match model expectations")
        print("4. Consider that the model might need special preprocessing")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()