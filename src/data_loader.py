import torch
from torch.utils.data import Dataset, DataLoader

class CharDataset(Dataset):
    def __init__(self, text, sequence_length):
        self.sequence_length = sequence_length
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.data = [self.char_to_idx[ch] for ch in text]
        print(f"Dataset created:")
        print(f"  Total characters: {len(text):,}")
        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  Sequence length: {sequence_length}")
    
    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, index):
        input_seq = self.data[index: index + self.sequence_length]
        target = self.data[index + self.sequence_length]
        input_seq = torch.tensor(input_seq, dtype = torch.long)
        target = torch.tensor(target, dtype = torch.long)
        return input_seq, target

def load_data(config):
    with open(config.data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    dataset = CharDataset(text, config.sequence_length)
    return dataset

def get_dataloader(dataset, config):
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,drop_last=True)
    print(f"\nDataLoader created:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Number of batches: {len(dataloader)}")
    return dataloader