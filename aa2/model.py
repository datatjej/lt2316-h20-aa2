# The classifier is supposed to take a sequence of words and return a sequence of labels, one label per token.

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class NamedEntityRecognizer(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.GRU = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)  
        self.lin3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)  

        
    def forward(self, batch, device):
        hidden = self.lin1(batch)
        print("hidden:", hidden.shape)
        gru, h_0 = self.GRU(hidden)
        print("gru:", gru.shape)
        last = self.lin3(gru)
        out = self.softmax(last).to(device)
        print("out:", out.shape)
        return out
        """super(NamedEntityRecognizer, self).__init__()  
        # linear input layer
        self.input = nn.Linear(input_size, hidden_size)
        # an LSTM layer takes the input layer output as input
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True)
        # a fully connected linear layer provides the final output layer
        self.output = nn.Linear(hidden_size, output_size)
        

    def __forward__(self, batch, device):
        #apply the embedding layer that maps each token to its embedding
        input_batch = self.input(batch) #.view((1, -1)) <-- from PyTorch tutorial, not sure if needed
                                                         #Ng: # dim: batch_size x batch_max_len x embedding_dim
        print("input_batch:", input_batch)
        #run the LSTM along the sentences of length batch_max_len
        out, _ = self.lstm(input_batch) #Ng: # dim: batch_size x batch_max_len x lstm_hidden_dim   
        print("out:", out)
        #reshape the Variable so that each row contains one token
        #out = out.view(-1, out.shape[2])  # dim: batch_size*batch_max_len x lstm_hidden_dim
        
        #apply the fully connected layer and obtain the output for each token
        out = self.output(out) # dim: batch_size*batch_max_len x num_tags
        print("out:", out)
        return out
        
        #calculate softmax liklihoods 
        #log_probs = F.log_softmax(out, dim=1).to(device)
        
        #return log_probs"""