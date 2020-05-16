import subprocess
import sys

try:
  from torchnlp.datasets import snli_dataset
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'pytorch-nlp'])
finally:
    from torchnlp.datasets import snli_dataset


import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchtext import data
from torchtext import datasets

word_to_ix={'entailment':0,'neutral':1,'contradiction':2,'-':3}
ix_to_word={0:'entailment',1:'neutral',2:'contradiction',3:'-'}
def map_to_ix(x):
  return word_to_ix[x]
def map_to_word(x):
  return ix_to_word[x]

#data_preprocessing
train=pd.DataFrame(snli_dataset(train=True), columns=['premise','hypothesis','label'])
train['label']=train['label'].apply(lambda x:map_to_ix(x))
val=pd.DataFrame(snli_dataset(dev=True), columns=['premise','hypothesis','label'])
val['label']=val['label'].apply(lambda x:map_to_ix(x))
test=pd.DataFrame(snli_dataset(test=True), columns=['premise','hypothesis','label'])
test['label']=test['label'].apply(lambda x:map_to_ix(x))

#TFID Logistic regression classifier
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression( penalty='l2',
                                multi_class='auto',solver='saga',
                                max_iter=100, tol=1e-3)),
     ])

text_clf.fit(train['hypothesis'], train['label'])

predicted = text_clf.predict(test['hypothesis'])
print(np.mean(predicted == test['label']))

with open("tfidf.txt", 'w') as f:
    for idx in range(len(predicted)):
        f.write("{}\n".format(map_to_word(predicted[idx])))

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

device = torch.device('cpu')

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

model.load_state_dict(torch.load('./models/best_model.pt',map_location=torch.device(device)))

def predict_inference(premise, hypothesis, text_field, label_field, model, device):
    
    model.eval()
    
    if isinstance(premise, str):
        premise = text_field.tokenize(premise)
    
    if isinstance(hypothesis, str):
        hypothesis = text_field.tokenize(hypothesis)
    
    if text_field.lower:
        premise = [t.lower() for t in premise]
        hypothesis = [t.lower() for t in hypothesis]
        
    premise = [text_field.vocab.stoi[t] for t in premise]
    hypothesis = [text_field.vocab.stoi[t] for t in hypothesis]
    
    premise = torch.LongTensor(premise).unsqueeze(1).to(device)
    hypothesis = torch.LongTensor(hypothesis).unsqueeze(1).to(device)
    
    prediction = model(premise, hypothesis)
    
    prediction = prediction.argmax(dim=-1).item()
    
    return label_field.vocab.itos[prediction]

with open("deep_model.txt", 'w') as f:
    for idx in range(len(predicted)):
        f.write("{}\n".format(predict_inference(test['premise'][idx], test['hypothesis'][idx], TEXT, LABEL, model, device)))
