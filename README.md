# Character-Level Text Generation with LSTM
This project implements a character-level language model using LSTM (Long Short-Term Memory) networks to generate Shakespeare-style text. The model learns to predict the next character in a sequence, capturing patterns in language structure, vocabulary, and writing style.

**Key Learning Objectives:**
- Understanding recurrent neural networks and LSTM architecture
- Implementing character-level sequence modeling
- Exploring temperature-based text generation
- Experiencing LSTM limitations firsthand (vanishing gradients, limited context)

## Features

- **Character-Level Modeling**: Learns patterns at the character level for flexible text generation
- **Stacked LSTM Architecture**: Two-layer LSTM with 512 hidden units per layer
- **Temperature Sampling**: Controllable text generation with temperature parameter (0.5 to 2.0)
- **Dropout Regularization**: Prevents overfitting with 30% dropout between layers
- **Apple Silicon Support**: Optimized for MPS (Metal Performance Shaders) on M-series Macs
- **Modular Code Structure**: Clean separation of data loading, model, training, and generation
- **Comprehensive Visualization**: Training loss curves and generation comparisons

## Dataset

**Source**: Shakespeare's Complete Works (Tiny Shakespeare dataset)
**Statistics**:
- **Total Characters**: 1,115,394
- **Vocabulary Size**: 65 unique characters (a-z, A-Z, digits, punctuation)
- **Training Sequences**: ~1.1 million sequences (100-character windows)
- **Content**: 37 plays and 154 sonnets

**Download**: Automatically downloaded via `download_data.py` from [Karpathy's char-rnn repository](https://github.com/karpathy/char-rnn)

## Model Architecture

```
Input Sequence (100 characters)
         ↓
  Embedding Layer
  (65 → 256 dim)
         ↓
    LSTM Layer 1
  (256 → 512 hidden)
         ↓
    Dropout (0.3)
         ↓
    LSTM Layer 2
  (512 → 512 hidden)
         ↓
    Linear Layer
  (512 → 65 output)
         ↓
Temperature Sampling
         ↓
  Next Character
```

## Project Structure

```
lstm-text-generation/
│
├── data/
│   └── input.txt                 
│
├── src/
│   ├── config.py                 
│   ├── data_loader.py             
│   ├── model.py                  
│   ├── train.py                   
│   └── generate.py                
│
├── models/
│   └── lstm_model.pth             
│
├── results/
│   └── training_loss.png          
│
├── download_data.py              
├── requirements.txt               
├── .gitignore                    
└── README.md                     
```

## Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch 2.0+
- GPU (optional): CUDA-enabled GPU or Apple Silicon (M1/M2/M3/M4) for MPS acceleration

### Training

Train the model from scratch:

```bash
python src/train.py
```
**Training Output**:
- Model checkpoint saved to `models/lstm_model.pth`
- Loss curve saved to `results/training_loss.png`
- Console displays epoch-by-epoch progress with loss values

**Training Time**: ~40 minutes for 10 epochs on Apple M4 MacBook Air

### Text Generation

Generate Shakespeare-style text with pre-trained model:

```bash
python src/generate.py
```

**Output**: Generates 500 characters at three temperature settings (0.5, 1.0, 1.5)

## Results

### Training Performance

**Training Configuration**: 10 epochs, batch size 128, learning rate 0.002

### Loss Curve Analysis
<img width="570" height="345" alt="image" src="https://github.com/user-attachments/assets/1701eb2c-3522-41b7-89d1-44ed20f6931c" />


**Observations**:
- Rapid improvement in first 3 epochs (loss: 1.66 → 1.55)
- Overfitting detected after epoch 3 (loss starts increasing)
- Best model performance at epoch 2-3

### Generated Text Examples

#### Temperature 0.5 (Conservative)
```
ROMEO:
Let him and the house of the cannot of my heads of the good.
LUCENTIO:
But when you says of thee, be not be proferity her hand bears,
Then the grace he hath stanged seem them bring of her to his head.
```
**Characteristics**: Most predictable, real words, proper structure, some grammar issues

#### Temperature 1.0 (Balanced)
```
ROMEO:
Than no breat and be you ore, gooks, or borious and
proocour.
We shuth'd my autes leave had within to leage for effearsing that'l,
I hope of liesn off promposour:
```
**Characteristics**: Mix of real and made-up words, maintains dialogue format

#### Temperature 1.5 (Creative)
```
ROMEO:
belf you not, pape-clfichadewifg?
BRUTNIA:
To 'smy; itsee, wrongly Margi brweigz, beftore?
A feect no forkisfate;
```
**Characteristics**: Very creative, many nonsense words, still maintains basic structure

### Model Capabilities

**What the Model Learned**:
- Character-to-character transition patterns
- English word formations and common vocabulary
- Punctuation usage (periods, commas, colons, apostrophes)
- Shakespeare's dialogue format (character names with colons)
- Capitalization rules
- Basic dialogue structure

**Current Limitations**:
- Cannot maintain coherent meaning beyond a few words
- Creates plausible but non-existent words
- Inconsistent grammar
- No semantic understanding
- Limited context window (100 characters)
- Overfitting after epoch 3

## Future Improvements

### Model Architecture
- Implement attention mechanism for better long-range dependencies
- Try GRU (Gated Recurrent Unit) and compare performance
- Experiment with deeper networks (3-4 LSTM layers)
- Increase hidden size to 1024 for more capacity

### Training Enhancements
- Reduce learning rate to 0.001 to prevent overfitting
- Increase dropout to 0.5 for better regularization
- Implement learning rate scheduling (decay over epochs)
- Add gradient clipping to prevent exploding gradients
- Train for 30-50 epochs with early stopping
- Create validation set for proper model evaluation

### Data & Tokenization
- Experiment with word-level tokenization
- Try byte-pair encoding (BPE) or SentencePiece
- Train on larger corpus (multiple authors)
- Add special tokens for better structure control

### Generation Improvements
- Implement beam search for higher quality output
- Add nucleus (top-p) sampling
- Try top-k sampling strategies
- Implement generation stopping criteria

### Advanced Features
- Upgrade to Transformer architecture
- Compare with modern language models
- Build interactive web interface
- Add fine-tuning on specific Shakespeare plays

## Key Learnings

1. **LSTM Architecture**: Gained hands-on understanding of recurrent networks, hidden states, and sequential processing

2. **Overfitting Recognition**: Learned to identify overfitting from loss curves and the importance of regularization

3. **Temperature Sampling**: Understood the trade-off between predictability and creativity in text generation

4. **Character vs. Word Level**: Experienced benefits (flexibility) and drawbacks (lack of semantics) of character-level modeling

5. **RNN Limitations**: Firsthand experience with why Transformers replaced RNNs:
   - Sequential processing (can't parallelize)
   - Limited context window
   - Vanishing gradient issues
   - Difficulty with long-range dependencies
