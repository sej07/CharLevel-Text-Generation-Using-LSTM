from src.config import Config

config = Config()

print("Configuration Settings")
print(f"\nDATA:")
print(f"  Data path: {config.data_path}")

print(f"\nMODEL:")
print(f"  Embedding dimension: {config.embedding_dim}")
print(f"  Hidden size: {config.hidden_size}")
print(f"  Number of layers: {config.num_layers}")
print(f"  Dropout rate: {config.dropout}")

print(f"\nTRAINING:")
print(f"  Sequence length: {config.sequence_length}")
print(f"  Batch size: {config.batch_size}")
print(f"  Learning rate: {config.learning_rate}")
print(f"  Number of epochs: {config.num_epochs}")

print(f"\nOTHER:")
print(f"  Device: {config.device}")
print(f"  Save directory: {config.save_dir}")
print(f"  Results directory: {config.results_dir}")