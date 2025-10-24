import torch
import torch.nn as nn

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size = embedding_dim, 
            hidden_size = hidden_size, 
            num_layers = num_layers, 
            dropout = dropout, 
            batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out , _ = self.lstm(embedded)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output

'''
Input: [128, 100]
"128 sequences, each with 100 character indices"  

Embedding: [128, 100, 256]
"128 sequences, 100 characters, each character is now 256 numbers"
  
LSTM: [128, 100, 512]
"128 sequences, 100 positions, LSTM outputs 512 numbers at each position"
  
Get last output [:, -1, :] : [128, 512]
"128 sequences, only the last position's 512 numbers"
  
Linear: [128, 65]
  "128 sequences, 65 scores (one for each possible character)"
'''