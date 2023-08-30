import argparse
import csv
import os
import sys
import json
import pickle
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from model import *
from gensim.models import KeyedVectors
from itertools import cycle
from torch.utils.data import TensorDataset, DataLoader
# os.environ['CUDA_VISIBLE_DEVICE'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0, help='CUDA device')
parser.add_argument('--model', type=str, help='model (CNN)')
parser.add_argument('--task', type=str, help='task name (UNTEdu))')
parser.add_argument('--max_seq_length', type=int, default=256, help='max sequence length')
parser.add_argument('--ckpt_path', type=str, help='model checkpoint path')
parser.add_argument('--output_path', type=str, help='model output path')
parser.add_argument('--train_path', type=str, help='train dataset path')
parser.add_argument('--dev_path', type=str, help='dev dataset path')
parser.add_argument('--test_path', type=str, help='test dataset path')
parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='learning rate')
parser.add_argument('--do_train', action='store_true', default=False, help='enable training')
parser.add_argument('--do_evaluate', action='store_true', default=False, help='enable evaluation')
args = parser.parse_args()
print(args)

# Loading word2vec model
embed_lookup = KeyedVectors.load_word2vec_format('word2vec_model/GoogleNews-vectors-negative300.bin', binary= True)

assert args.task in ('UNTEdu')
assert args.model in ('CNN')

if args.task in ('UNTEdu'):
    n_classes = 2

def cuda(tensor):
    """Places tensor on CUDA device."""
    return tensor.to(args.device)

label_map = {'1.ForRepo': 1, '2.NotForRepo': 0}

train_path = args.train_path
val_path = args.dev_path
test_path = args.test_path
train_file = open(train_path,"r")
val_file = open(val_path,"r")
test_file = open(test_path,"r")
train = train_file.read().strip().split("\n")
val = val_file.read().strip().split('\n')
test = test_file.read().strip().split("\n")
train_text,train_label = [], []
val_text,val_label = [], []
test_text,test_label = [], []
all_text = []
for item in train:
    item = item.split("\t")
    train_text.append(item[-2])
    train_label.append(label_map[item[2]])
    all_text.append(item[-2])
for item in val:
    item = item.split("\t")
    val_text.append(item[-2])
    val_label.append(label_map[item[2]])
    all_text.append(item[-2])
for item in test:
    item = item.split("\t")
    test_text.append(item[-2])
    test_label.append(label_map[item[2]])
    all_text.append(item[-2])

pretrained_words = []
for word in embed_lookup.key_to_index:
    pretrained_words.append(word)
row_idx = 1
# get word/embedding in that row
word = pretrained_words[row_idx] # get words by index
embedding = embed_lookup[word] # embeddings by word

# vocab and embedding info
print("Size of Vocab: {}\n".format(len(pretrained_words)))
print('Word in vocab: {}\n'.format(word))
print('Length of embedding: {}\n'.format(len(embedding)))
#print('Associated embedding: \n', embedding)


def pad_features(tokenized_sentences, seq_length):
    ''' Return features of tokenized_sentences, where each sentence is padded with 0's 
        or truncated to the input seq_length.
    '''
    
    # getting the correct rows x cols shape
    features = np.zeros((len(tokenized_sentences), seq_length), dtype=int)

    # for each review, I grab that review and 
    for i, row in enumerate(tokenized_sentences):
        features[i, -len(row):] = np.array(row)[:seq_length]
    
    return features

def tokenize_all_sentences(embed_lookup, list_split):
    # split each sentence into a list of words
    words = [sentence.split() for sentence in list_split]

    tokenized_sentences = []
    for word in words:
        ints = []
        for w in word:
            try:
                idx = embed_lookup.vocab[w].index
            except: 
                idx = 0
            ints.append(idx)
        tokenized_sentences.append(ints)
    
    return tokenized_sentences

train_x = tokenize_all_sentences(embed_lookup,train_text)
train_x = pad_features(train_x,args.max_seq_length)
train_y = np.array(train_label)
val_x = tokenize_all_sentences(embed_lookup,val_text)
val_x = pad_features(val_x,args.max_seq_length)
val_y = np.array(val_label)
test_x = tokenize_all_sentences(embed_lookup,test_text)
test_x = pad_features(test_x,args.max_seq_length)
test_y = np.array(test_label)

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

# dataloaders
batch_size = args.batch_size

# shuffling and batching data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

vocab_size = len(pretrained_words)
output_size = n_classes # binary class (1 or 0)
embedding_dim = len(embed_lookup[pretrained_words[0]]) # 300-dim vectors
num_filters = 100
kernel_sizes = [3, 4, 5]

model = CNNTextClassifier(embed_lookup, vocab_size, output_size, embedding_dim,
                   num_filters, kernel_sizes)

lr=0.0005

# criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
print_every = 10
# training loop
def train(model, train_loader, epochs, print_every=100):

    # move model to GPU, if available
    model = cuda(model)
    counter = 0 # for printing
    
    # train for some number of epochs
    model.train()
    for e in range(epochs):
        # batch loop
        for inputs, labels in train_loader:
            counter += 1
            inputs, labels = cuda(inputs), cuda(labels)
            # zero accumulated gradients
            model.zero_grad()

            # get the output from the model
            output = model(inputs)
            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels)
            loss.backward()
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_losses = []
                model.eval()
                for inputs, labels in valid_loader:
                    inputs, labels = cuda(inputs), cuda(labels)
                    output = model(inputs)
                    val_loss = criterion(output.squeeze(), labels)
                    val_losses.append(val_loss.item())

                model.train()
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))

train(model, train_loader, args.epochs, print_every=print_every)

test_losses = [] # track loss
num_correct = 0
model.eval()
total_num = 0
y_true,y_pred = [], []
# iterate over test data
for inputs, labels in test_loader:
    inputs, labels = cuda(inputs), cuda(labels)
    # get predicted outputs
    output = model(inputs)
    
    # calculate loss
    test_loss = criterion(output.squeeze(), labels)
    test_losses.append(test_loss.item())
    
    # convert output probabilities to predicted class
    pred = F.softmax(output,dim=-1)
    pred = torch.argmax(pred,dim=-1)
    y_pred += pred.data.tolist()
    y_true += labels.data.tolist()

accuracy = accuracy_score(y_true, y_pred) * 100.
f1 = f1_score(y_true, y_pred, average='macro') * 100.

results_dict = {
    'accuracy': accuracy_score(y_true, y_pred) * 100.,
    'macro-F1': f1_score(y_true, y_pred, average='macro') * 100.,
}
for k, v in results_dict.items():
    print(f'{k} = {v}')
