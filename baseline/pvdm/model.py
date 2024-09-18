"""
This code is modified from cbowdon's repository.
https://github.com/cbowdon/doc2vec-pytorch

"""

import os
import json
import random
import math
from tqdm import tqdm
import argparse
import numpy as np
from collections import Counter, OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import torch.optim


class Vocab:
    def __init__(self, all_tokens, min_count=1):
        self.min_count = min_count
        counter = Counter(all_tokens)
        self.freqs = {t:n for t, n in counter.items() if n >= min_count}
        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
        # add UNK word
        ordered_dict = OrderedDict([('UNK', sum(counter.values()))] + list(ordered_dict.items()))
        self.words = list(ordered_dict.keys())
        self.word2idx = {w: i for i, w in enumerate(self.words)}


# get Negative Samples
class NoiseDistribution:
    def __init__(self, vocab):
        self.unk_idx = vocab.word2idx.get('UNK', 0)  # 获取 UNK 的索引
        self.probs = np.array([vocab.freqs[w] for w in vocab.words if vocab.word2idx[w] != self.unk_idx])
        self.probs = np.power(self.probs, 0.75)
        self.probs /= np.sum(self.probs)
        self.vocab_size = len(vocab.words)
        self.idx_to_word = [vocab.word2idx[w] for w in vocab.words if vocab.word2idx[w] != self.unk_idx]
        
    def sample(self, n):
        "Returns the indices of n words randomly sampled from the vocabulary."
        sampled_indices = np.random.choice(a=len(self.probs), size=n, p=self.probs)
        return [self.idx_to_word[idx] for idx in sampled_indices]

# loss
class NegativeSampling(nn.Module):
    def __init__(self):
        super(NegativeSampling, self).__init__()
        self.log_sigmoid = nn.LogSigmoid()
    def forward(self, scores):
        batch_size = scores.shape[0]
        positive = self.log_sigmoid(scores[:,0])
        negatives = torch.sum(self.log_sigmoid(-scores[:,1:]), dim=1)
        return -torch.sum(positive + negatives) / batch_size  # average for batch
    

# pytorch dataset
class NCEDataset(Dataset):
    def __init__(self, examples):
        self.doc_ids = examples['doc_ids']
        self.sample_ids = examples['sample_ids']
        self.context_ids = examples['context_ids']
    def __len__(self):
        return len(self.doc_ids)
    def __getitem__(self, index):
        doc_id = self.doc_ids[index]
        sample_id = self.sample_ids[index]
        context_id = self.context_ids[index]
        return {"doc_id": doc_id,
                "sample_id": sample_id,
                "context_id": context_id}


class DistributedMemory(nn.Module):
    def __init__(self, vec_dim, n_docs, n_words):
        super(DistributedMemory, self).__init__()
        self.paragraph_matrix = nn.Parameter(torch.randn(n_docs, vec_dim))
        self.word_matrix = nn.Parameter(torch.randn(n_words, vec_dim))
        self.outputs = nn.Parameter(torch.zeros(vec_dim, n_words))
    
    def forward(self, doc_ids, context_ids, sample_ids):
        inputs = torch.add(self.paragraph_matrix[doc_ids,:], torch.sum(self.word_matrix[context_ids,:], dim=1))
        outputs = self.outputs[:,sample_ids]
        logits = torch.bmm(inputs.unsqueeze(dim=1), outputs.permute(1, 0, 2)).squeeze()
        return logits
    
    def predict(self, doc_ids, context_ids):
        inputs = torch.add(self.paragraph_matrix[doc_ids,:], torch.sum(self.word_matrix[context_ids,:], dim=1))
        
        outputs = torch.matmul(inputs, self.outputs)
        predicted_ids = torch.argmax(outputs, dim=1)
        return predicted_ids
    
    def get_word_vector(self, word_id):
        return self.word_matrix[word_id, :]