class Config:
    data_path = 'data/input.txt'
    embedding_dim = 256
    hidden_size = 512
    num_layers = 2
    dropout = 0.3
    sequence_length = 100
    batch_size = 128
    learning_rate = 0.002
    num_epochs= 50
    device = 'mps'
    save_dir = 'models'
    results_dir = 'results'

config = Config()