import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from config import config
from data_loader import load_data, get_dataloader
from model import CharLSTM

print("Script started!")  

def train_model(model, dataloader, criterion, optimizer, device, num_epochs):
    model.to(device)
    model.train()
    losses =[]
    print("\n Starting Training")
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/ {num_epochs}")
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss' : loss.item()})
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+ 1}/ {num_epochs} - Loss: {avg_loss:.4f}")
    return losses

def plot_loss(losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth = 2)
    plt.xlabel("Epoch", fontsize = 12)
    plt.ylabel("Loss",fontsize = 12)
    plt.title("Training")
    plt.grid(True, alpha= 0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n Loss plot saved to {save_path}")
    plt.close()

def main():
    print("\n Loading Data")
    dataset = load_data(config)
    dataloader= get_dataloader(dataset, config)
    print("\n Creating Model")
    model = CharLSTM(vocab_size= dataset.vocab_size,
                     embedding_dim= config.embedding_dim,
                     hidden_size=config.hidden_size,
                     num_layers=config.num_layers,
                     dropout= config.dropout)
    print("\n Model created")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = config.learning_rate)
    losses = train_model(model= model, dataloader= dataloader, criterion=criterion,
                         optimizer=optimizer, device= config.device, 
                         num_epochs=config.num_epochs)
    os.makedirs(config.save_dir, exist_ok=True)
    model_path = os.path.join(config.save_dir, 'lstm_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': dataset.vocab_size,
        'char_to_idx': dataset.char_to_idx,
        'idx_to_char': dataset.idx_to_char,
        'config': {
            'embedding_dim': config.embedding_dim,
            'hidden_size': config.hidden_size,
            'num_layers': config.num_layers,
            'dropout': config.dropout
        }
    }, model_path)
    print(f"\n Model saved to {model_path}")

    os.makedirs(config.results_dir, exist_ok=True)
    plot_path = os.path.join(config.results_dir, 'training_loss.png')
    plot_loss(losses, plot_path)
    print("Training Complete!")

if __name__ == "__main__":
    main()