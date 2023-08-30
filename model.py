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
# from gensim.models import KeyedVectors


class PreTrainedModel(nn.Module):
    """Pre-trained model for classification."""

    def __init__(self,model_type, n_classes):
        super().__init__()
        self.model_type = model_type
        self.model = AutoModel.from_pretrained(model_type)
        if model_type in ('bert-base-uncased', 'roberta-base'):
            self.classifier = nn.Linear(768, n_classes)
        elif model_type in ('bert-large-uncased', 'roberta-large-uncased'):
            self.classifier = nn.Linear(1024,n_classes)
        
    def forward(self, input_ids, segment_ids, attention_mask, unlabeled=False):
        transformer_params = {
            'input_ids': input_ids,
            'token_type_ids': (
                segment_ids if self.model_type in ('bert-base-uncased','bert-large-uncased') else None
            ),
            'attention_mask': attention_mask,
        }
        transformer_outputs = self.model(**transformer_params)
        cls_output = transformer_outputs[0][:, 0]
        logits = self.classifier(cls_output)
        return logits

class CNNTextClassifier(nn.Module):
    """
    The embedding layer + CNN model that will be used to perform text classification.
    """

    def __init__(self, embed_model, vocab_size, output_size, embedding_dim,
                 num_filters=200, kernel_sizes=[3, 4, 5], freeze_embeddings=False, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(CNNTextClassifier, self).__init__()

        # set class vars
        self.num_filters = num_filters
        self.embedding_dim = embedding_dim
        
        # 1. embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # set weights to pre-trained
        self.embedding.weight = nn.Parameter(torch.from_numpy(embed_model.vectors)) # all vectors
        # (optional) freeze embedding weights
        if freeze_embeddings:
            self.embedding.requires_grad = False
        
        # 2. convolutional layers
        self.convs_1d = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim), padding=(k-2,0)) 
            for k in kernel_sizes])
        
        # 3. final, fully-connected layer for classification
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, output_size) 
        
        # 4. dropout and sigmoid layers
        self.dropout = nn.Dropout(drop_prob)
        self.sig = nn.Sigmoid()
        
    
    def conv_and_pool(self, x, conv):
        """
        Convolutional + max pooling layer
        """
        # squeeze last dim to get size: (batch_size, num_filters, conv_seq_length)
        # conv_seq_length will be ~ 200
        x = F.relu(conv(x)).squeeze(3)
        
        # 1D pool over conv_seq_length
        # squeeze to get size: (batch_size, num_filters)
        x_max = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x_max

    def forward(self, x):
        """
        Defines how a batch of inputs, x, passes through the model layers.
        Returns a single, sigmoid-activated class score as output.
        """
        # embedded vectors
        embeds = self.embedding(x) # (batch_size, seq_length, embedding_dim)
        # embeds.unsqueeze(1) creates a channel dimension that conv layers expect
        embeds = embeds.unsqueeze(1)
        
        # get output of each conv-pool layer
        conv_results = [self.conv_and_pool(embeds, conv) for conv in self.convs_1d]
        
        # concatenate results and add dropout
        x = torch.cat(conv_results, 1)
        x = self.dropout(x)
        
        # final logit
        logit = self.fc(x) 
        
        # sigmoid-activated --> a class score
        return logit #self.sig(logit)
