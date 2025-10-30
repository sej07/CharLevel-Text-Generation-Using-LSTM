import torch 
import torch.nn.functional as F
from model import CharLSTM
import sys
import os

def generate_text(model, seed_text, char_to_idx, idx_to_char, 
                  length = 500, temperature = 1.0, device = 'mps'):
    model.eval()
    model.to(device)
    generated = seed_text
    current_seq = [char_to_idx[char] for char in seed_text]
    for _ in range(length):
        input_seq = current_seq[-100:]
        x= torch.tensor(input_seq).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(x)
        #lower temperature is sharper probabilities
        output = output/temperature
        probs = F.softmax(output, dim = -1)
        next_idx = torch.multinomial(probs, num_samples=1).item()
        next_char = idx_to_char[next_idx]
        current_seq.append(next_idx)
        generated += next_char
    return generated

def main():
    model_path = 'models/lstm_model.pth'
    if not os.path.exists(model_path):
        print("Error: No trained model found")
        sys.exit(1)
    print("Loading model")
    checkpoint = torch.load(model_path)

    model = CharLSTM(vocab_size= checkpoint["vocab_size"],
                    embedding_dim= checkpoint["config"]["embedding_dim"],
                    hidden_size=checkpoint["config"]["hidden_size"],
                    num_layers= checkpoint["config"]["num_layers"],
                    dropout=checkpoint["config"]["dropout"])
    model.load_state_dict(checkpoint["model_state_dict"])
    char_to_idx = checkpoint["char_to_idx"]
    idx_to_char = checkpoint["idx_to_char"]
    print("Model loaded \n")

    seed_text = "Romeo:"
    temperatures = [0.5, 1.0, 1.5]
    for temp in temperatures:
        print(f"Temperature: {temp}")
        text = generate_text(model=model, 
                             seed_text=seed_text,
                             char_to_idx=char_to_idx,
                             idx_to_char= idx_to_char, 
                             length= 00, 
                             temperature=temp, 
                             device='mps')
    print(text)
    print("\n")

if __name__ == "__main__":
    main()