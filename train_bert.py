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
import pyparsing as pp
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from transformers import AdamW, AutoModel, AutoTokenizer,get_linear_schedule_with_warmup
from torch.distributions.distribution import Distribution
from tqdm import tqdm
from torch.distributions import Categorical
from model import *
from itertools import cycle

# csv.field_size_limit(sys.maxsize)

# os.environ['CUDA_VISIBLE_DEVICE'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0, help='CUDA device')
parser.add_argument('--model', type=str, help='pre-trained model (bert-base-uncased, roberta-base)')
parser.add_argument('--task', type=str, help='task name (SNLI, MNLI, QQP, TwitterPPDB, SWAG, HellaSWAG)')
parser.add_argument('--max_seq_length', type=int, default=256, help='max sequence length')
parser.add_argument('--ckpt_path', type=str, help='model checkpoint path')
parser.add_argument('--output_path', type=str, help='model output path')
parser.add_argument('--train_path', type=str, help='train dataset path')
parser.add_argument('--dev_path', type=str, help='dev dataset path')
parser.add_argument('--test_path', type=str, help='test dataset path')
parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0., help='weight decay')
parser.add_argument('--label_smoothing', type=float, default=-1., help='label smoothing \\alpha')
parser.add_argument('--max_grad_norm', type=float, default=1., help='gradient clip')
parser.add_argument('--do_train', action='store_true', default=False, help='enable training')
parser.add_argument('--do_evaluate', action='store_true', default=False, help='enable evaluation')
parser.add_argument('--warmup_steps',type=int, default=0)
parser.add_argument('--gradient_accumulation_steps',default=1)
args = parser.parse_args()
print(args)

assert args.task in ('UNTEdu')
assert args.model in ('bert-base-uncased')

if args.task in ('UNTEdu'):
    n_classes = 2

def cuda(tensor):
    """Places tensor on CUDA device."""

    return tensor.to(args.device)


def load(dataset, batch_size, shuffle):
    """Creates data loader with dataset and iterator options."""

    return DataLoader(dataset, batch_size, shuffle=shuffle)


def adamw_params(model):
    """Prepares pre-trained model parameters for AdamW optimizer."""

    no_decay = ['bias', 'LayerNorm.weight']
    params = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        },
    ]
    return params


def encode_sentence(sentence):
    """
    Encodes a single sentence for pre-trained models using the template
    [CLS] sentence1 [SEP]. 
    Returns input_ids, segment_ids, and attention_mask.
    """

    inputs = tokenizer.encode(
        sentence, max_length=args.max_seq_length
    )
    input_ids = inputs
    segment_ids = [0]*len(inputs)
    attention_mask = [1] * len(inputs) 
    padding_length = args.max_seq_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    segment_ids += [0] * padding_length
    attention_mask += [0] * padding_length
    for input_elem in (input_ids, segment_ids, attention_mask):
        assert len(input_elem) == args.max_seq_length
    return (
        cuda(torch.tensor(input_ids).long()),
        cuda(torch.tensor(segment_ids).long()),
        cuda(torch.tensor(attention_mask).long()),
    )


def encode_pair_inputs(sentence1, sentence2):
    """
    Encodes pair inputs for pre-trained models using the template
    [CLS] sentence1 [SEP] sentence2 [SEP]. Used for concatenating document and title
    Returns input_ids, segment_ids, and attention_mask.
    """

    inputs = tokenizer.encode_plus(
        sentence1, sentence2, add_special_tokens=True, max_length=args.max_seq_length
    )
    input_ids = inputs['input_ids']
    if args.model == 'bert-base-uncased' or args.model == 'bert-large-uncased':
        segment_ids = inputs['token_type_ids']
    else:
        segment_ids = [0]*len(inputs['input_ids'])
    attention_mask = inputs['attention_mask']
    padding_length = args.max_seq_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    segment_ids += [0] * padding_length
    attention_mask += [0] * padding_length
    for input_elem in (input_ids, segment_ids, attention_mask):
        assert len(input_elem) == args.max_seq_length
    return (
        cuda(torch.tensor(input_ids)).long(),
        cuda(torch.tensor(segment_ids)).long(),
        cuda(torch.tensor(attention_mask)).long(),
    )


def encode_label(label):
    """Wraps label in tensor."""

    return cuda(torch.tensor(label)).long()


class UNTEduProcessor:
    """Data loader for UNTEdu."""

    def __init__(self):
        self.label_map = {'1.ForRepo': 1, '2.NotForRepo': 0}

    def valid_inputs(self, sentence, label):
        return len(sentence.split()) > 0 and len(sentence.split()) < args.max_seq_length and label in self.label_map

    def load_samples(self, path, train=True):
        samples = []
        file = open(path,"r")
        data = file.read().strip().split("\n")
        header = True

        for row in data:
            if header:
                header = False
                continue
            row = row.split("\t")
            label = row[2]
            title = row[-1]
            text = row[-2]
            text = text.split()[:args.max_seq_length]
            text = " ".join(text)
            if self.valid_inputs(text,label):
                label = self.label_map[label]
                samples.append((text,label))
        return samples


def select_processor():
    """Selects data processor using task name."""

    return globals()[f'{args.task}Processor']()

class TextDataset(Dataset):
    """
    Task-specific dataset wrapper. Used for storing, retrieving, encoding,
    caching, and batching samples.
    """

    def __init__(self, path, processor, train=True):
        self.samples = processor.load_samples(path, train=train)
        self.cache = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        res = self.cache.get(i, None)
        if res is None:
            sample = self.samples[i]
            if args.task in ('UNTEdu'): 
                sentence, label = sample
                if args.model == 'bert-base-uncased':
                    input_ids, segment_ids, attention_mask = encode_sentence(
                        sentence
                    )
                label_id = encode_label(label)
                res = ((input_ids, segment_ids, attention_mask), label_id)
            self.cache[i] = res
        return res

def train(data,epoch=0):
    """Fine-tunes pre-trained model on training set."""

    model.train()
    train_loss = 0.
    train_loader = tqdm(load(data, args.batch_size, True))
    optimizer = AdamW(adamw_params(model), lr=args.learning_rate, eps=1e-8)
    for i, dataset in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, labels = dataset
        logit = model(inputs[0],inputs[1],inputs[2])
        loss = criterion(logit,labels)
        train_loss += loss.item()
        train_loader.set_description(f'train loss = {(train_loss / (i+1)):.6f}')
        loss.backward()
        if args.max_grad_norm > 0.:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
    return train_loss / len(train_loader)


def evaluate(dataset):
    """Evaluates pre-trained model on development set."""

    model.eval()
    eval_loss = 0.
    eval_acc = 0. 
    y_true, y_pred = [], []
    eval_loader = tqdm(load(dataset, args.batch_size, False))
    for i, dataset in enumerate(eval_loader):
        with torch.no_grad():
            inputs, labels = dataset
            output = model(inputs[0],inputs[1],inputs[2])
            for j in range(output.size(0)):
                y_pred.append(output[j].argmax().item())
                y_true.append(labels[j].item())
            loss = criterion(output,labels)
        eval_loss += loss.item()
        eval_loader.set_description(f'eval loss = {(eval_loss / (i+1)):.6f}')
    eval_acc = accuracy_score(y_true, y_pred) * 100.
    return eval_loss / len(eval_loader), eval_acc


model = cuda(PreTrainedModel(model_type=args.model, n_classes=n_classes))
tokenizer = AutoTokenizer.from_pretrained(args.model)
processor = select_processor()
criterion = nn.CrossEntropyLoss()

if args.train_path:
    train_dataset = TextDataset(args.train_path, processor)
    print(f'train samples = {len(train_dataset)}')
if args.dev_path:
    dev_dataset = TextDataset(args.dev_path, processor, train=False)
    print(f'dev samples = {len(dev_dataset)}')
if args.test_path:
    test_dataset = TextDataset(args.test_path, processor, train=False)
    print(f'test samples = {len(test_dataset)}')



if args.do_train:
    print()
    print('*** training ***')
    best_acc = -float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = train(data=train_dataset, epoch=epoch)
        eval_loss, eval_acc = evaluate(dev_dataset)
        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save(model.state_dict(), args.ckpt_path)
        print(
            f'epoch = {epoch} | '
            f'train loss = {train_loss:.6f} | '
            f'eval loss = {eval_loss:.6f} | '
            f'eval acc = {eval_acc:.6f} '
        )

if args.do_evaluate:
    if not os.path.exists(args.ckpt_path):
        raise RuntimeError(f'\'{args.ckpt_path}\' does not exist')
    
    print()
    print('*** evaluating ***')

    output_dicts = []
    model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()
        
    test_loader = tqdm(load(test_dataset, args.batch_size, False))

    for i, (inputs, label) in enumerate(test_loader):
        with torch.no_grad():
            logits = model(inputs[0],inputs[1],inputs[2])
            for j in range(logits.size(0)):
                probs = F.softmax(logits[j], -1)
                output_dict = {
                    'index': args.batch_size * i + j,
                    'true': label[j].item(),
                    'pred': logits[j].argmax().item(),
                    'conf': probs.max().item(),
                    'logits': logits[j].cpu().numpy().tolist(),
                    'probs': probs.cpu().numpy().tolist(),
                }
                output_dicts.append(output_dict)

    print(f'writing outputs to \'{args.output_path}\'')

    with open(args.output_path, 'w+') as f:
        for i, output_dict in enumerate(output_dicts):
            output_dict_str = json.dumps(output_dict)
            f.write(f'{output_dict_str}\n')

    y_true = [output_dict['true'] for output_dict in output_dicts]
    y_pred = [output_dict['pred'] for output_dict in output_dicts]
    y_conf = [output_dict['conf'] for output_dict in output_dicts]

    accuracy = accuracy_score(y_true, y_pred) * 100.
    f1 = f1_score(y_true, y_pred, average='macro') * 100.
    confidence = np.mean(y_conf) * 100.

    results_dict = {
        'accuracy': accuracy_score(y_true, y_pred) * 100.,
        'macro-F1': f1_score(y_true, y_pred, average='macro') * 100.,
        'confidence': np.mean(y_conf) * 100.,
    }
    for k, v in results_dict.items():
        print(f'{k} = {v}')
