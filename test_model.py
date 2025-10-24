import torch
from src.config import Config
from src.data_loader import load_data
from src.model import CharLSTM

config = Config()
dataset = load_data(config)
print("Creating Model")
model = CharLSTM(vocab_size= dataset.vocab_size,
                 embedding_dim= config.embedding_dim,
                 hidden_size= config.hidden_size,
                 num_layers= config.num_layers,
                 dropout= config.dropout)
print(f"\n Model architecture")
print(model)
print("\n Testinf forward pass")
batch_size = 4
seq_length = 100
dummy_input = torch.randint(0, dataset.vocab_size, (batch_size, seq_length))
output = model(dummy_input)
print(f"Output shape: {output.shape}")
print(f"Expected: [{batch_size}, {dataset.vocab_size}]")
