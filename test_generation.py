"""
Test generation with different seeds
"""

import torch
from src.model import CharLSTM
from src.generate import generate_text

# Load model
checkpoint = torch.load('models/lstm_model.pth', map_location='mps')
model = CharLSTM(
    vocab_size=checkpoint['vocab_size'],
    embedding_dim=checkpoint['config']['embedding_dim'],
    hidden_size=checkpoint['config']['hidden_size'],
    num_layers=checkpoint['config']['num_layers'],
    dropout=checkpoint['config']['dropout']
)
model.load_state_dict(checkpoint['model_state_dict'])

char_to_idx = checkpoint['char_to_idx']
idx_to_char = checkpoint['idx_to_char']

# Test different seeds
seeds = [
    "ROMEO:",
    "JULIET:",
    "To be or not to be",
    "First Citizen:",
]

for seed in seeds:
    print("="*70)
    print(f"Seed: '{seed}'")
    print("="*70)
    
    text = generate_text(
        model=model,
        seed_text=seed,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        length=300,
        temperature=1.0,
        device='mps'
    )
    
    print(text)
    print("\n")