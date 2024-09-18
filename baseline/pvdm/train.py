# by Guodong Zhou
# 2024.4.6

import os
import json
import random
from tqdm import tqdm
import argparse
import numpy as np
from collections import Counter, OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import torch.optim
from PVDM.model import *

# load data
def load_data(file_path):
    sentences = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            sentences.append(line)
    doc_id = [i[0] for i in sentences]
    data = [i[1:] for i in sentences]
    return doc_id, data        

# save samples  
def samples_generator(save_path, doc_ids, docs, context_size, noise, n_negative_samples, vocab):
    doc_id_list = []
    sample_ids_list = []
    context_ids_list = []
    unk_idx = vocab.word2idx.get('UNK', 0)
    for k in tqdm(range(len(doc_ids)), desc="generating examples"):
        doc_id, doc = doc_ids[k], docs[k]
        for i in range(context_size, len(doc) - context_size):
            positive_sample = vocab.word2idx.get(doc[i], unk_idx)
            # ensure negative samples don't accidentally include the positive
            sample_ids = []
            while(len(sample_ids) < n_negative_samples):
                sample_id = noise.sample(1)
                if sample_id != positive_sample:
                    sample_ids.extend(sample_id)

            sample_ids.insert(0, positive_sample)
            context = doc[i - context_size:i] + doc[i + 1:i + context_size + 1]
            context_ids = [vocab.word2idx.get(w, unk_idx) for w in context]

            doc_id_list.append(int(doc_id))     # 文档ID
            sample_ids_list.append(sample_ids)      # 负样本(中心词，n_neg)
            context_ids_list.append(context_ids)    # 正样本（中心词的背景词

            # temp1 = torch.tensor(doc_id_list)
            # temp2 = torch.tensor(sample_ids_list)
            # temp3 = torch.tensor(context_ids_list)
    data = {"doc_ids": torch.tensor(doc_id_list),
            "sample_ids": torch.tensor(sample_ids_list), 
            "context_ids": torch.tensor(context_ids_list)}
    
    with open(save_path, "wb+") as f:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        pickle.dump(data, f)
    
    print("sucuess saved!")

# load samples  
def load_samples(save_path):
    with open(save_path, "rb") as f:
        loaded_data = pickle.load(f)

    return loaded_data

# calculate acc
def cosine_similarity(vec1, vec2):
    return torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2))

def calculate_similarity_accuracy(predicted_vectors, actual_vectors, threshold=0.5):
    similarities = [cosine_similarity(pred_vec, act_vec) for pred_vec, act_vec in zip(predicted_vectors, actual_vectors)]
    correct = sum(similarity >= threshold for similarity in similarities)
    total = len(similarities)
    return correct / total


class pvdm_model:
    def __init__(self, vec_dim=512, epochs=100, batch_size=4096, lr=5e-4, is_save=True,
                 window_size=3, neg_count=5, rootpath=r"PDFObj2Vec/PVDM",
                 train_data_file = r"PDFObj2Vec/PVDM/dataset/unix20_train_data", 
                 test_data_file = r"PDFObj2Vec/PVDM/dataset/unix20_test_data", 
                 recorder_name=r"recorder_unix20_512.json", 
                 save_model_path = r"/home/dell/data/SD/PDFObj2Vec/PVDM/record_v2/save_model_unix20_512"):
        self.vec_dim = vec_dim
        self.epochs = epochs
        self.is_save = is_save
        self.recorder_name = recorder_name
        self.save_model_path = save_model_path
        self.rootpath = rootpath

        # load data
        self.train_doc_ids, self.train_docs = load_data(train_data_file)
        self.test_doc_ids, self.test_docs = load_data(test_data_file)

        # get vocab and transform
        self.vocab = Vocab([word for sentence in self.train_docs for word in sentence], min_count=1)
        print(f"Dataset comprises {len(self.train_docs)} documents and {len(self.vocab.words)} unique words (over the limit of {self.vocab.min_count} occurrences)")

        # get negative samples
        self.noise = NoiseDistribution(self.vocab)

        # get all examples
        train_samples_save_file = r'/home/dell/data/SD/PDFObj2Vec/PVDM/dataset/unix20_train_samples'
        test_samples_save_file = r'/home/dell/data/SD/PDFObj2Vec/PVDM/dataset/unix20_test_samples'
        # If you have samples, please use '#'
        samples_generator(train_samples_save_file, self.train_doc_ids, self.train_docs, context_size=window_size, noise=self.noise, n_negative_samples=neg_count, vocab=self.vocab)
        samples_generator(test_samples_save_file, self.test_doc_ids, self.test_docs, context_size=window_size, noise=self.noise, n_negative_samples=neg_count, vocab=self.vocab)
        self.train_data = load_samples(train_samples_save_file)
        self.test_data = load_samples(test_samples_save_file)
        
        # dataloader
        self.train_dataset = NCEDataset(self.train_data)
        self.test_dataset = NCEDataset(self.test_data)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

        self.model = DistributedMemory(self.vec_dim, n_docs=len(self.train_doc_ids), n_words=len(self.vocab.words))
        self.device = "cuda" if torch.cuda.is_available else "cpu"
        self.model.to(self.device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.loss = NegativeSampling()

    def train(self):
        self.model.train()
        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            print('Epoch: {}/{}'.format(epoch, self.epochs - 1))
            epoch_losses = []
            epoch_acc = []
            for batch in tqdm(self.train_dataloader, desc="Training", leave=False):
                self.model.zero_grad()

                doc_ids = batch["doc_id"].to(self.device)
                sample_ids = batch["sample_id"].to(self.device)
                context_ids = batch["context_id"].to(self.device)

                logits = self.model.forward(doc_ids, context_ids, sample_ids)
                batch_loss = self.loss(logits)
                batch_loss.backward()
                self.optimizer.step()

                # loss
                epoch_losses.append(batch_loss.item())
                # acc
                predicted_vectors = [self.model.get_word_vector(pid) for pid in self.model.predict(doc_ids, context_ids)]
                actual_vectors = [self.model.get_word_vector(aid) for aid in sample_ids[:, 0]]
                batch_acc = calculate_similarity_accuracy(predicted_vectors, actual_vectors)
                epoch_acc.append(batch_acc)

            train_loss = np.mean(epoch_losses)
            train_acc = np.mean([acc.cpu().item() for acc in epoch_acc])
            # train_acc = np.mean(epoch_acc)
            print(f"Train_loss:{train_loss}, Train_acc:{train_acc}")
            val_loss, val_acc = self.evaluate()

            # save
            if self.is_save:
                os.makedirs(self.save_model_path, exist_ok=True)
                # save recorder
                with open('{}/record_v2/{}'.format(self.rootpath, self.recorder_name), 'a+') as f:
                    record = {'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                                'val_loss':val_loss, 'val_acc':val_acc}
                    json.dump(record, f, indent=1)
                    f.write('\n')
                # save model
                torch.save(self.model.state_dict(), '{}/{}'.format(self.save_model_path, "pvdm_model_epoch" + str(epoch) + ".pth"))
        print("Train done!")

    def evaluate(self):
        self.model.eval()
        val_losses = []
        val_acc = []
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc="Evaluating", leave=False):
                doc_ids = batch["doc_id"].to(self.device)
                sample_ids = batch["sample_id"].to(self.device)
                context_ids = batch["context_id"].to(self.device)

                logits = self.model.forward(doc_ids, context_ids, sample_ids)
                batch_loss = self.loss(logits)
                
                # loss
                val_losses.append(batch_loss.item())
                # acc
                predicted_vectors = [self.model.get_word_vector(pid) for pid in self.model.predict(doc_ids, context_ids)]
                actual_vectors = [self.model.get_word_vector(aid) for aid in sample_ids[:, 0]]
                batch_acc = calculate_similarity_accuracy(predicted_vectors, actual_vectors)
                val_acc.append(batch_acc)

        ret_loss = np.mean(val_losses)
        ret_acc = np.mean([acc.cpu().item() for acc in val_acc])
        # ret_acc = np.mean(val_acc)
        print(f"Val_loss:{ret_loss}, Val_acc:{ret_acc}")

        return ret_loss,  ret_acc
    


if __name__ == '__main__':
    pvdm = pvdm_model()
    pvdm.train()