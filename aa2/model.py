# The classifier is supposed to take a sequence of words and return a sequence of labels, one label per token.

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class NamedEntityRecognizer(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, number_of_tags)
        super(NamedEntityRecognizer, self).__init__()
        # the Embeddings module is used to store word embeddings and retrieve them using indices
        self.embeddings = nn.Embeddings(vocab_size, embedding_dim)
        # the LSTM takes the embedded sentences as input
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, batch_first=True)
        # a fully connected linear layer provides the output layer
        self.output = nn.Linear(lstm_hidden_dim, number_of_tags)
   
    def __forward__(self, tensorized_sent):
        #apply the embedding layer that maps each token to its embedding
        embedded_sent = self.embeddings(tensorized_sent) #.view((1, -1)) <-- from PyTorch tutorial, not sure if needed
                                                         #Ng: # dim: batch_size x batch_max_len x embedding_dim
        #run the LSTM along the sentences of length batch_max_len
        out, _ = self.lstm(embedded_sent) #Ng: # dim: batch_size x batch_max_len x lstm_hidden_dim   
        
        #reshape the Variable so that each row contains one token
        out = out.view(-1, out.shape[2])  # dim: batch_size*batch_max_len x lstm_hidden_dim
        
        #apply the fully connected layer and obtain the output for each token
        out = self.output(out) # dim: batch_size*batch_max_len x num_tags
        
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
    
'''
Tutorial: https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

'''


'''
#Tutorial: https://cs230.stanford.edu/blog/namedentity/
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()

    #maps each token to an embedding_dim vector
    self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim)

    #the LSTM takens embedded sentence
    self.lstm = nn.LSTM(params.embedding_dim, params.lstm_hidden_dim, batch_first=True)

    #fc layer transforms the output to give the final output layer
    self.fc = nn.Linear(params.lstm_hidden_dim, params.number_of_tags)

def forward(self, s):
    #apply the embedding layer that maps each token to its embedding
    s = self.embedding(s)   # dim: batch_size x batch_max_len x embedding_dim

    #run the LSTM along the sentences of length batch_max_len
    s, _ = self.lstm(s)     # dim: batch_size x batch_max_len x lstm_hidden_dim                

    #reshape the Variable so that each row contains one token
    s = s.view(-1, s.shape[2])  # dim: batch_size*batch_max_len x lstm_hidden_dim

    #apply the fully connected layer and obtain the output for each token
    s = self.fc(s)          # dim: batch_size*batch_max_len x num_tags

    return F.log_softmax(s, dim=1)   # dim: batch_size*batch_max_len x num_tags

''' 