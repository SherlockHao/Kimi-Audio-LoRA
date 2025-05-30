#!/usr/bin/env python3
# Step 1: Environment Setup and Data Validation

import os
import sys
import csv
import json
import torch
import torchaudio
from pathlib import Path
from collections import Counter

# Configuration
BASE_DIR = "/opt/data/nvme4/kimi"
DATA_DIR = os.path.join(BASE_DIR, "data/test2")
CODE_DIR = os.path.join(BASE_DIR, "Kimi-Audio-LoRA")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Create directories
os.makedirs(CODE_DIR, exist_ok=True)
os.makedirs(os.path.join(CODE_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(CODE_DIR, "checkpoints"), exist_ok=True)

def check_environment():
    """Check system environment and dependencies"""
    print("=== Environment Check ===")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check PyTorch and CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    
    print()

def validate_data():
    """Validate training data"""
    print("=== Data Validation ===")
    
    # Check data directory
    train_dir = os.path.join(DATA_DIR, "train")
    audio_dir = os.path.join(train_dir, "audio")
    metadata_file = os.path.join(train_dir, "metadata.csv")
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    print(f"Data directory: {DATA_DIR}")
    print(f"Training directory: {train_dir}")
    
    # Read and validate metadata
    metadata = []
    with open(metadata_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata.append(row)
    
    print(f"Total samples in metadata: {len(metadata)}")
    
    # Validate audio files
    valid_count = 0
    missing_files = []
    corrupted_files = []
    duration_stats = []
    text_lengths = []
    
    for idx, item in enumerate(metadata):
        audio_path = os.path.join(train_dir, item['audio_path'])
        
        # Check if file exists
        if not os.path.exists(audio_path):
            missing_files.append(audio_path)
            continue
        
        # Try to load audio
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            duration = waveform.shape[-1] / sample_rate
            duration_stats.append(duration)
            valid_count += 1
            
            # Collect text statistics
            text_lengths.append(len(item['text']))
            
            # Show progress
            if (idx + 1) % 100 == 0:
                print(f"Validated {idx + 1}/{len(metadata)} files...")
        except Exception as e:
            corrupted_files.append((audio_path, str(e)))
    
    print(f"\nValidation Results:")
    print(f"  Valid audio files: {valid_count}")
    print(f"  Missing files: {len(missing_files)}")
    print(f"  Corrupted files: {len(corrupted_files)}")
    
    if missing_files:
        print(f"\nFirst 5 missing files:")
        for f in missing_files[:5]:
            print(f"  - {f}")
    
    if corrupted_files:
        print(f"\nFirst 5 corrupted files:")
        for f, err in corrupted_files[:5]:
            print(f"  - {f}: {err}")
    
    # Duration statistics
    if duration_stats:
        print(f"\nAudio duration statistics:")
        print(f"  Min: {min(duration_stats):.2f}s")
        print(f"  Max: {max(duration_stats):.2f}s")
        print(f"  Average: {sum(duration_stats)/len(duration_stats):.2f}s")
        print(f"  Total: {sum(duration_stats)/60:.2f} minutes")
    
    # Text statistics
    if text_lengths:
        print(f"\nText length statistics:")
        print(f"  Min: {min(text_lengths)} chars")
        print(f"  Max: {max(text_lengths)} chars")
        print(f"  Average: {sum(text_lengths)/len(text_lengths):.1f} chars")
    
    # Sample some texts
    print(f"\nSample texts (first 5):")
    for i, item in enumerate(metadata[:5]):
        print(f"  {i+1}: {item['text'][:100]}...")
    
    return metadata, valid_count

def check_models():
    """Check if required models exist"""
    print("\n=== Model Check ===")
    
    models = {
        "Kimi-Audio-7B-Instruct": os.path.join(MODEL_DIR, "Kimi-Audio-7B-Instruct"),
        "glm-4-voice-tokenizer": os.path.join(MODEL_DIR, "glm-4-voice-tokenizer"),
        "whisper-large-v3": os.path.join(MODEL_DIR, "whisper-large-v3")
    }
    
    for model_name, model_path in models.items():
        if os.path.exists(model_path):
            print(f"{model_name}: Found at {model_path}")
            # Check for common model files
            files = os.listdir(model_path)
            print(f"  Files: {len(files)} items")
            if len(files) < 20:
                for f in sorted(files)[:10]:
                    print(f"    - {f}")
        else:
            print(f"{model_name}: NOT FOUND at {model_path}")
    
    return models

def save_config():
    """Save configuration for next steps"""
    config = {
        "base_dir": BASE_DIR,
        "data_dir": DATA_DIR,
        "code_dir": CODE_DIR,
        "model_dir": MODEL_DIR,
        "train_audio_dir": os.path.join(DATA_DIR, "train", "audio"),
        "train_metadata": os.path.join(DATA_DIR, "train", "metadata.csv"),
        "checkpoint_dir": os.path.join(CODE_DIR, "checkpoints"),
        "log_dir": os.path.join(CODE_DIR, "logs"),
        "models": {
            "kimi_audio": os.path.join(MODEL_DIR, "Kimi-Audio-7B-Instruct"),
            "tokenizer": os.path.join(MODEL_DIR, "glm-4-voice-tokenizer"),
            "whisper": os.path.join(MODEL_DIR, "whisper-large-v3")
        }
    }
    
    config_path = os.path.join(CODE_DIR, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nConfiguration saved to: {config_path}")

def main():
    print("Starting Kimi-Audio ASR Fine-tuning Setup")
    print("=" * 50)
    
    try:
        # Check environment
        check_environment()
        
        # Validate data
        metadata, valid_count = validate_data()
        
        # Check models
        models = check_models()
        
        # Save configuration
        save_config()
        
        print("\n" + "=" * 50)
        print("Step 1 completed successfully!")
        print(f"Found {valid_count} valid training samples")
        print("Configuration saved. Ready for Step 2: Data Loader Implementation")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()