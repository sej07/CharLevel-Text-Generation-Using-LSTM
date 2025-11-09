"""
Debug generation
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

print("Generating text with debugging...")
print("="*50)

try:
    text = generate_text(
        model=model,
        seed_text="ROMEO:",
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        length=100,  # Just 100 chars for testing
        temperature=1.0,
        device='mps'
    )
    
    print(f"Generated text ({len(text)} chars):")
    print(text)
    print("="*50)
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()