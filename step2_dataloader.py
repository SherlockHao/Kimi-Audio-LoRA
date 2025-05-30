#!/usr/bin/env python3
# Step 2: Data Loader Implementation

import os
import json
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor
import csv
from typing import Dict, List, Tuple, Optional

# Load configuration
CODE_DIR = "/opt/data/nvme4/kimi/Kimi-Audio-LoRA"
config_path = os.path.join(CODE_DIR, "config.json")

if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found: {config_path}. Please run step1 first.")

with open(config_path, 'r') as f:
    config = json.load(f)

class AudioTextDataset(Dataset):
    """Dataset for audio-text pairs"""
    
    def __init__(
        self,
        metadata_path: str,
        whisper_processor_path: str,
        max_audio_length: float = 30.0,  # seconds
        sample_rate: int = 16000
    ):
        self.metadata_path = metadata_path
        self.max_audio_length = max_audio_length
        self.sample_rate = sample_rate
        
        # Load metadata
        self.data = []
        base_dir = os.path.dirname(metadata_path)
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                audio_path = os.path.join(base_dir, row['audio_path'])
                if os.path.exists(audio_path):
                    self.data.append({
                        'audio_path': audio_path,
                        'text': row['text'],
                        'speaker_id': row.get('speaker_id', '0'),
                        'duration': float(row.get('duration', 0))
                    })
        
        print(f"Loaded {len(self.data)} valid samples from {metadata_path}")
        
        # Initialize Whisper processor
        print(f"Loading Whisper processor from: {whisper_processor_path}")
        self.processor = WhisperProcessor.from_pretrained(whisper_processor_path)
        
    def __len__(self) -> int:
        return len(self.data)
    
    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and resample audio"""
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Trim or pad to max length
        max_samples = int(self.max_audio_length * self.sample_rate)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        
        return waveform.squeeze(0)  # Remove channel dimension
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample"""
        item = self.data[idx]
        
        # Load audio
        audio = self.load_audio(item['audio_path'])
        
        # Process audio with Whisper processor
        input_features = self.processor(
            audio.numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        ).input_features.squeeze(0)
        
        # Process text
        # For ASR, we typically use the text as-is for training
        text = item['text']
        
        return {
            'input_features': input_features,
            'text': text,
            'audio_path': item['audio_path'],
            'duration': item['duration']
        }

class DataCollator:
    """Custom data collator for batching"""
    
    def __init__(self, processor: WhisperProcessor, padding: bool = True):
        self.processor = processor
        self.padding = padding
    
    def __call__(self, features: List[Dict]) -> Dict:
        # Extract input features
        input_features = [f['input_features'] for f in features]
        texts = [f['text'] for f in features]
        
        # Stack input features
        batch_input_features = torch.stack(input_features)
        
        # For ASR training, we'll need to tokenize texts later with the actual model tokenizer
        # For now, we'll just pass them through
        batch = {
            'input_features': batch_input_features,
            'texts': texts,
            'audio_paths': [f['audio_path'] for f in features],
            'durations': torch.tensor([f['duration'] for f in features])
        }
        
        return batch

def create_dataloaders(
    train_metadata: str,
    whisper_path: str,
    batch_size: int = 8,
    num_workers: int = 4,
    max_audio_length: float = 30.0
) -> Tuple[DataLoader, WhisperProcessor]:
    """Create training dataloader"""
    
    # Create dataset
    dataset = AudioTextDataset(
        metadata_path=train_metadata,
        whisper_processor_path=whisper_path,
        max_audio_length=max_audio_length
    )
    
    # Create data collator
    collator = DataCollator(dataset.processor)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    return dataloader, dataset.processor

def test_dataloader():
    """Test the dataloader implementation"""
    print("Testing Data Loader Implementation")
    print("=" * 50)
    
    # Create dataloader
    train_dataloader, processor = create_dataloaders(
        train_metadata=config['train_metadata'],
        whisper_path=config['models']['whisper'],
        batch_size=4,
        num_workers=2,
        max_audio_length=30.0
    )
    
    print(f"\nDataloader created successfully!")
    print(f"Number of batches: {len(train_dataloader)}")
    print(f"Batch size: 4")
    print(f"Total samples: {len(train_dataloader.dataset)}")
    
    # Test loading a few batches
    print("\nTesting batch loading...")
    for i, batch in enumerate(train_dataloader):
        if i >= 2:  # Test only first 2 batches
            break
        
        print(f"\nBatch {i+1}:")
        print(f"  input_features shape: {batch['input_features'].shape}")
        print(f"  Number of texts: {len(batch['texts'])}")
        print(f"  Durations: {batch['durations'].tolist()}")
        print(f"  Sample text: {batch['texts'][0][:100]}...")
    
    print("\nData loader test completed successfully!")
    
    # Save dataset info
    dataset_info = {
        'total_samples': len(train_dataloader.dataset),
        'batch_size': 4,
        'num_batches': len(train_dataloader),
        'sample_rate': 16000,
        'max_audio_length': 30.0,
        'processor_name': config['models']['whisper']
    }
    
    info_path = os.path.join(CODE_DIR, "dataset_info.json")
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\nDataset info saved to: {info_path}")

def main():
    try:
        test_dataloader()
        print("\n" + "=" * 50)
        print("Step 2 completed successfully!")
        print("Data loader is ready. Next step: Model loading and LoRA configuration")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()