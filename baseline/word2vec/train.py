# by Guodong Zhou
# 2024.3.31

import os
import torch
import pickle
import torch.optim as optim
from word2vec.word2vec_data import Word2VecDatasetProcess, Word2VecDatasetCreator
from word2vec.model import SkipGramModel
from tqdm import tqdm
import json
import math
import numpy as np
from collections import Counter, OrderedDict

class Word2VecWithSkipGram:
    ''' 使用Skip-Gram模型, 使用负采样优化'''
    def __init__(self, is_save=False, lr_adjust=40960, 
                 record_name=r"recorder_unix20_512.json", 
                 train_data_file=r"PDFObj2Vec/word2vec/dataset/unix20_train_data", 
                 test_data_file=r"PDFObj2Vec/word2vec/dataset/unix20_test_data",
                 root_path=r'PDFObj2Vec/word2vec',
                 emb_dimension=512, batch_size=4096, epochs=100,
                 window_size=3, neg_count=5, initial_lr=1e-4, min_freq=1): # 5e-3
        '''
        Args:
            emb_dimention: 嵌入层输出的维数
            window_size: 取背景词的窗口大小
            neg_count: 对于每一个正样本生成负样本的个数
            min_freq: 最小词频
        '''
        self.train_data = Word2VecDatasetProcess(train_data_file, min_freq)
        self.test_data = Word2VecDatasetProcess(test_data_file, min_freq)
        self.root_path = root_path
        self.vocab_size = len(self.train_data.word_id)
        self.emb_dimension = emb_dimension
        self.neg_count = neg_count
        self.batch_size = batch_size
        self.epochs = epochs
        self.window_size = window_size
        self.initial_lr = initial_lr
        self.lr_adjust = lr_adjust
        self.is_save = is_save
        self.record_name = record_name

        self.model = SkipGramModel(self.vocab_size, self.emb_dimension, dropout=0.5)
        self.device = "cuda" if torch.cuda.is_available else "cpu"
        self.model.to(self.device)
        
        # self.optimizer = optim.SGD(
        #     self.model.parameters(), lr=self.initial_lr)
        self.optimizer = optim.SparseAdam(self.model.parameters(), lr=self.initial_lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def train_batch(self):
        self.model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pairs_count = self.train_data.get_pospair_count(self.window_size)
        batches = int(pairs_count / self.batch_size)
        for _ in tqdm(range(batches)):
            pos_pairs = self.train_data.get_pospair(self.batch_size, self.window_size)
            neg_v = self.train_data.get_negv(pos_pairs, self.neg_count) # 负样本背景词
            pos_u = [pair[0] for pair in pos_pairs] # 正样本中心词
            pos_v = [pair[1] for pair in pos_pairs] # 正样本背景词
            
            pos_u = torch.LongTensor(pos_u).to(self.device)
            pos_v = torch.LongTensor(pos_v).to(self.device)
            neg_v = torch.LongTensor(neg_v).to(self.device)

            self.optimizer.zero_grad()
            batch_loss, batch_correct, batch_total = self.model.forward(pos_u,pos_v,neg_v)
            batch_loss.backward()
            self.optimizer.step()

            train_loss += batch_loss.item()
            train_correct += batch_correct
            train_total += batch_total
        
        avg_loss = train_loss / batches
        accuracy = train_correct / train_total

        return avg_loss, accuracy
    

    def val_batch(self):
        self.model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        pairs_count = self.test_data.get_pospair_count(self.window_size)
        batches = int(pairs_count / self.batch_size)
        for _ in tqdm(range(batches)):
            val_pos_pairs = self.test_data.get_pospair(self.batch_size, self.window_size)
            val_neg_v = self.test_data.get_negv(val_pos_pairs, self.neg_count) # 负样本背景词
            val_pos_u = [pair[0] for pair in val_pos_pairs] # 正样本中心词
            val_pos_v = [pair[1] for pair in val_pos_pairs] # 正样本背景词

            val_pos_u = torch.LongTensor(val_pos_u).to(self.device)
            val_pos_v = torch.LongTensor(val_pos_v).to(self.device)
            val_neg_v = torch.LongTensor(val_neg_v).to(self.device)
            
            with torch.no_grad():
                batch_loss, batch_correct, batch_total = self.model.forward(val_pos_u, val_pos_v, val_neg_v)
                
                val_loss += batch_loss.item()           
                val_correct += batch_correct
                val_total += batch_total

        avg_loss = val_loss / batches
        accuracy = val_correct / val_total

        return avg_loss, accuracy 


    def save_record(self, i, train_loss, train_acc, val_loss, val_acc):
        save_model_path = r"/home/dell/data/SD/PDFObj2Vec/word2vec/record_v2/save_model_unix20_512"
        os.makedirs(save_model_path, exist_ok=True)
        if self.is_save:
            torch.save(self.model.state_dict(), '{}/record_v2/save_model_unix20_512/{}'.format(self.root_path, "word2vec_model_epoch" + str(i) + ".pth"))
            with open('{}/record_v2/{}'.format(self.root_path, self.record_name), 'a+') as f:
                record = {'epoch': i, 'train_loss': train_loss, 'train_acc': train_acc,
                          'val_loss':val_loss, 'val_acc':val_acc}
                json.dump(record, f, indent=1)
                f.write('\n')


    @staticmethod
    def load_model(model_path, vocab_size, emb_dimension):
        model = SkipGramModel(vocab_size, emb_dimension, dropout=0.5)
        # with open(model_path, 'rb') as f:
        #     model = pickle.load(f)
        model.load_state_dict(torch.load(model_path))
        return model
    

    @staticmethod
    def org_embedding(model, word_id, emb_dimension, org_file_path, tf, idf, device_id):
        # 句向量 = TF-IDF加权平均词向量, 返回org中所有obj的句向量字典
        org_after_model = {}
        device = torch.device(f'cuda:{device_id}')
        model = model.to(device)
        model.eval()
        with open(org_file_path, 'r') as f:
            cfg = json.load(f)

            for id1 in list(cfg.keys()):
                block = cfg[id1]
                seq_norm = Word2VecDatasetCreator.norm_insn(block['insn_list'])

                org_after_model[id1] = np.array([0] * emb_dimension)
                org_after_model[id1] = org_after_model[id1].astype(float)
                if len(seq_norm) == 0:
                    continue
                for ins in seq_norm:
                    if ins not in word_id.keys():
                        # 如果是没有见过的词汇，将其词向量置0(跳过)
                        continue
                    input_id = word_id[ins]
                    input_tensor = torch.LongTensor([input_id]).to(device)
                    with torch.no_grad():
                        word_vector = model.u_embedding(input_tensor).cpu().numpy()
                    org_after_model[id1] += tf[ins] * idf[ins] * word_vector.reshape(emb_dimension)

        return org_after_model


    @staticmethod
    def get_TF(words_freq):
        # 计算词频(TF)
        total = sum(words_freq.values())
        return {key: value / total for key, value in words_freq.items()}
    

    @staticmethod
    def get_IDF(data_path=r"PDFObj2Vec/word2vec/dataset/unix20_train_data"):
        # 计算逆文档频率(IDF)
        with open(data_path, 'r', encoding='utf-8') as file:
            sentences_list = [line.strip().split() for line in file]
        
        # 这里将每一个obj都视为一个文档
        doc_sum = len(sentences_list)
        doc_freq = {}
        for sentence in sentences_list:
            temp_freq = {}
            for word in sentence:
                if word not in temp_freq:
                    temp_freq[word] = 1
            for word in temp_freq.keys():
                if word in doc_freq:
                    doc_freq[word] += 1
                else:
                    doc_freq[word] = 1

        for key, value in doc_freq.items():
            doc_freq[key] = math.log(doc_sum) - math.log(value)
        
        return doc_freq




if __name__ == '__main__':
    word2vec = Word2VecWithSkipGram()
    for i in tqdm(range(0, word2vec.epochs)):
        print('Epoch: {}/{}'.format(i, word2vec.epochs - 1))
        train_loss, train_acc = word2vec.train_batch()
        val_loss, val_acc = word2vec.val_batch()
        print("Train: loss: {:.8f}, acc: {:.8f}\nVal: loss: {:.8f}, acc: {:.8f}".format(train_loss, train_acc, val_loss, val_acc))
        if word2vec.is_save:
            word2vec.save_record(i, train_loss, train_acc, val_loss, val_acc)
        word2vec.scheduler.step()
    print("train done!\n")