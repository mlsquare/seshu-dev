import sklearn as sk



# Read the downloaded data into a dictionary
# words = []
# with open("./data/data-small.csv", "r") as file:
#     for line in file:
#         word = line.strip()
#         words.append(word)


words = ['the','and','have', 'that', 'for', 'you', 'with', 'say', 'this', 'they', 'but', 'his', 'from', 'not', 'she', 'as', 'what', 'their', 'can', 'who']
print('words',words)

vocab = [chr(i) for i in range(ord('a'), ord('z')+1)]
vocab.append('0')
print('vocab',vocab)

letter_to_index = {letter: index for index, letter in enumerate(vocab)}
index_to_letter = {index: letter for index, letter in enumerate(vocab)}

word_to_index = {word: index for index, word in enumerate(words)}
index_to_word = {index: word for index, word in enumerate(words)}



#print(letter_to_index)
#print(index_to_letter)

#print(word_to_index)
#print(index_to_word)

N = len(words)
# includes special end of word character
V = len(vocab)
# includes end of word character
L = max(len(word) for word in words)+1

import numpy as np
tf = np.zeros((N, V))
for i, word in enumerate(words):
    chars = list(word)
    for char in chars:
        tf[i, letter_to_index[char]] += 1
    


vocab_sim = np.dot(tf.T, tf)
word_sim = np.dot(tf, tf.T)

# create doc matrix, where cols are vocab, rows are words
import torch


def wordEmb():
    tensor = 0.05*torch.randn(N, V)
    for i in range(N):
        tensor[i,i] = 1
    return tensor




import torch.nn as nn



import math
from torch import nn, Tensor

# https://github.com/KasperGroesLudvigsen/influenza_transformer/blob/main/positional_encoder.py
class PositionalEncoder(nn.Module):
    """
    The authors of the original transformer paper describe very succinctly what 
    the positional encoding layer does and why it is needed:
    
    "Since our model contains no recurrence and no convolution, in order for the 
    model to make use of the order of the sequence, we must inject some 
    information about the relative or absolute position of the tokens in the 
    sequence." (Vaswani et al, 2017)
    Adapted from: 
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(
        self, 
        dropout: float=0.1, 
        max_seq_len: int=5000, 
        d_model: int=512,
        batch_first: bool=True
        ):

        """
        Parameters:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model 
                     (Vaswani et al, 2017)
        """

        super().__init__()

        self.d_model = d_model
        
        self.dropout = nn.Dropout(p=dropout)

        self.batch_first = batch_first

        # adapted from PyTorch tutorial
        position = torch.arange(max_seq_len).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        if self.batch_first:
            pe = torch.zeros(1, max_seq_len, d_model)
            
            pe[0, :, 0::2] = torch.sin(position * div_term)
            
            pe[0, :, 1::2] = torch.cos(position * div_term)
        else:
            pe = torch.zeros(max_seq_len, 1, d_model)
        
            pe[:, 0, 0::2] = torch.sin(position * div_term)
        
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val] or 
               [enc_seq_len, batch_size, dim_val]
        """
        if self.batch_first:
            x = x + self.pe[:,:x.size(1)]
        else:
            x = x + self.pe[:x.size(0)]

        return self.dropout(x)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, max_seq_len=L):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.pos_encoder = PositionalEncoder(d_model = input_size, dropout=0.1, max_seq_len=max_seq_len)
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden state with zeros
        # h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        
        #h0 can be improved (h0 = x*toen_embedding)

        # Forward pass through RNN
        #out, _ = self.rnn(x, h0)
        x = x * math.sqrt(self.input_size)
        x = self.pos_encoder(x)
        out, _ = self.rnn(x)
        
        # Concatenate the output of RNN with y
        #out = torch.cat((out[:, -1, :], y.unsqueeze(1)), dim=1)
        # out = out[:, -1, :]
        
        # Pass the concatenated output through the fully connected layer
        out = self.fc(out)
        
        return out

# Define the dimensions
input_size = N
hidden_size = V
num_classes = V
num_layers = 1

# Create an instance of the RNN model

model = RNN(input_size, hidden_size, num_layers=num_layers, num_classes=num_classes)

# Print the model architecture
print(model)

# Prepare training set

X_train = []
Y_train = []

for word in words:

    chars = list(word)
    x = torch.zeros(1, L, N)
    y = torch.zeros(1, L)
    
    n = len(chars)
    for i in range(L):
        x[0, i, word_to_index[word]] = 1
        if i < n:
            y[0,i] = letter_to_index[chars[i]]
        else:
            y[0,i] = V-1
    
    X_train.append(x)
    Y_train.append(y)

# Convert the training set to tensors
X_train = torch.cat(X_train, dim=0)
Y_train = torch.cat(Y_train, dim=0)

# Print the shape of the training set
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
# prepare training set


# Load the trained model
#model = RNN(input_size, hidden_size, num_layers=num_layers, num_classes=num_classes)
model.load_state_dict(torch.load("./data/detok_toy_trained_model.pth"))
model.eval()



# Convert the torch array to numpy array
rnn_weights = model.rnn.state_dict()
rnn_weights_np = {name: weight.numpy() for name, weight in rnn_weights.items()}

import matplotlib.pyplot as plt

# Visualize the matrix
w = []
for name, weight in rnn_weights_np.items():
    print(f"Layer: {name}")
    print(weight.shape)
    weight[weight>=np.quantile(weight,0.95)]=1
    weight[weight<np.quantile(weight,0.95)]=0
    w.append(weight)

plt.imshow(w[0])
plt.savefig('./data/'+name+'.png')
