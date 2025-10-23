from src.config import Config
from src.data_loader import load_data, get_dataloader

config = Config()
print("Loading Dataset...")
dataset = load_data(config)
dataloader = get_dataloader(dataset, config)

print("Testing One Batch")

inputs, targets = next(iter(dataloader))

print(f"\nShapes:")
print(f"  Inputs: {inputs.shape}")
print(f"  Targets: {targets.shape}")

print(f"\nFirst sequence:")
first_seq = inputs[0].tolist()
decoded = ''.join([dataset.idx_to_char[idx] for idx in first_seq])
target_char = dataset.idx_to_char[targets[0].item()]

print(f"  Input: '{decoded}'")
print(f"  Target: '{target_char}'")

print("\n Data loader works!")