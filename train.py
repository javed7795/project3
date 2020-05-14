# -*- coding: utf-8 -*-
"""Untitled4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YcC7FdlauuxSPa7VSsY_72TeTFQQXtpa
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchtext import data
from torchtext import datasets

import random
import numpy as np

import time

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy', lower = True)
LABEL = data.LabelField()

train_data, valid_data, test_data = datasets.SNLI.splits(TEXT, LABEL)

MIN_FREQ = 2

TEXT.build_vocab(train_data, 
                 min_freq = MIN_FREQ,
                 vectors = "glove.6B.300d",
                 unk_init = torch.Tensor.normal_)

LABEL.build_vocab(train_data)

BATCH_SIZE = 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE,
    device = device)

class RNN(nn.Module):
  def __init__(self,
                input_dim,
                embedding_dim,
                hidden_dim,
                output_dim,
                num_layers,
                num_fl_layers,
                dropout,
                pad_idx):
      
    super().__init__()

    self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx)
    
    self.translation = nn.Linear(embedding_dim, hidden_dim)
    
    self.rnn = nn.GRU(embedding_dim,
                       hidden_dim,
                       num_layers=num_layers,
                       bidirectional=True,
                       dropout=dropout if num_layers>1 else 0)
    
    fc_dim = 2*hidden_dim
    
    fcs = [nn.Linear(fc_dim * 2, fc_dim * 2) for _ in range(num_fl_layers-1)]
    
    self.fcs = nn.ModuleList(fcs)
    
    self.fc_out = nn.Linear(fc_dim * 2, output_dim)
    
    self.dropout = nn.Dropout(dropout)
      
  def forward(self, prem, hypo):
      
    embedded_prem = self.embedding(prem)
    embedded_hypo = self.embedding(hypo)
    
    translated_prem = F.relu(self.translation(embedded_prem))
    translated_hypo = F.relu(self.translation(embedded_hypo))
    
    outputs_prem, hidden_prem = self.rnn(translated_prem)
    outputs_hypo, hidden_hypo = self.rnn(translated_hypo)

    hidden_prem=torch.cat((hidden_prem[-1,:,:],hidden_prem[-2,:,:]), dim=-1)
    hidden_hypo=torch.cat((hidden_hypo[-1,:,:],hidden_hypo[-2,:,:]), dim=-1)

    hidden = torch.cat((hidden_prem, hidden_hypo), dim=-1)
    hidden=hidden.squeeze(0)
    for fc in self.fcs:
        hidden = fc(hidden)
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)
    
    prediction = self.fc_out(hidden)        
    return prediction

#Training Specifications
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 300
HIDDEN_DIM = 300
OUTPUT_DIM = len(LABEL.vocab)
NUM_LAYERS=1
NUM_FL_LAYERS=4
DROPOUT = 0.25
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = RNN(INPUT_DIM,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            OUTPUT_DIM,
            NUM_LAYERS,
            NUM_FL_LAYERS,
            DROPOUT,
            PAD_IDX).to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

#ADD pretrained embedding to model
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.requires_grad = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        prem = batch.premise
        hypo = batch.hypothesis
        labels = batch.label
        
        optimizer.zero_grad()
        
        #prem = [prem sent len, batch size]
        #hypo = [hypo sent len, batch size]
        
        predictions = model(prem, hypo)
        
        #predictions = [batch size, output dim]
        #labels = [batch size]
        
        loss = criterion(predictions, labels)
                
        acc = binary_accuracy(predictions, labels)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            prem = batch.premise
            hypo = batch.hypothesis
            labels = batch.label
                        
            predictions = model(prem, hypo)
            
            loss = criterion(predictions, labels)
                
            acc = binary_accuracy(predictions, labels)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 5

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), './models/best_model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

model.load_state_dict(torch.load('./models/best_model.pt',map_location=torch.device(device)))
# torch.save(model.state_dict(), 'RNN1Simple-model.pt')
test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')