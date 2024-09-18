import csv
import json
import os
import re
import sys
import argparse
import torch
from tqdm import tqdm
import torch_geometric.data
from torch_geometric.data import Data, DataLoader, Dataset
from transformers import BertConfig
import torch.optim as optim
import math
import numpy as np
from collections import Counter, OrderedDict


_current_dir = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_current_dir, '..'))
sys.path.append(PROJECT_ROOT)

from PVDM.train import *
from PVDM.model import Vocab, DistributedMemory
from PVDM.pvdm_data import Doc2VecDatasetCreator
sys.path.append('../dataset/')

class BBSDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, segment_labels, attention_mask):
        self.inputs = inputs  # (total, seq_len)
        self.segment_labels = segment_labels  # (total, seq_len)
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_ = self.inputs[idx]
        segment_label_ = self.segment_labels[idx]
        attention_mask_ = self.attention_mask[idx]

        return {
            'inputs': torch.tensor(input_),
            'segment_labels': torch.tensor(segment_label_),
            'attention_mask': torch.tensor(attention_mask_)
        }


class ORGDataset(torch_geometric.data.Dataset):
    def __init__(self, root, label_path):
        self.number_of_classes = 2
        self.label_path = label_path
        
        # labels
        self.labels = {}
        with open(self.label_path, 'r') as f:
            data = csv.reader(f)
            for row in data:
                if row[0] == 'Id':
                    continue
                self.labels[row[0]] = int(row[1])
        super(ORGDataset, self).__init__(root, transform=None, pre_transform=None)
        # Dataset.__init__(self, root=root, transform=None, pre_transform=None)

    @property
    def num_classes(self):
        return self.number_of_classes

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'org') #zgd

    @property
    def raw_file_names(self):
        files_names = os.listdir(self.raw_dir)
        # exclude .DS_Store
        if '.DS_Store' in files_names:
            files_names.remove('.DS_Store')
        return files_names

    @property
    def processed_file_names(self):
        file_names = ['data_{}_{}.pt'.format(
            i, filename.split('.')[0]) for i, filename in enumerate(self.raw_file_names)]
        return file_names

    def raw_file_name_lookup(self, idx):
        return self.raw_file_names[idx]

    def train_val_split(self, k):
        ''' k fold train_val_split 
        k = 0, 1, 2, 3, 4
        '''
        # TODO: 5-fold
        # 0 0.2 0.4 0.6 0.8
        label_list = list()
        for filename in self.raw_file_names:
            label = self.labels[filename]
            label_list.append(int(label))

        groups = [[] for _ in range(2)]
        for i, label in enumerate(label_list):
            groups[label - 1].append(i)

        train_idx = [];
        val_idx = []

        for group in groups:
            group_len = len(group)
            slice_1 = int(group_len * 0.2 * k)
            slice_2 = int(group_len * 0.2 * (k + 1))

            train_idx.extend(group[0:slice_1])
            train_idx.extend(group[slice_2:])

            val_idx.extend(group[slice_1:slice_2])
            # train_idx.extend(group[int(len(group) * 0.15):])
            # val_idx.extend(group[:int(len(group) * 0.15)])

        return train_idx, val_idx

    def len(self):
        return len(self.processed_file_names)

    def process(self):
        raise NotImplementedError

    def get(self, idx):
        raw_file_name = self.raw_file_name_lookup(idx).split('.')[0]
        data = torch.load(os.path.join(self.processed_dir, 'data_{}_{}.pt'.format(idx, raw_file_name)))
        return data


class ORGDataset_Normalized_After_BERT(ORGDataset):
    def __init__(self, device_id, root=None, hidden_size=512, label_path=None, doc2vec_path=None):
        
        self.doc2vec_path = doc2vec_path
        self.hidden_size = hidden_size
        self.device_id = device_id
        super(ORGDataset_Normalized_After_BERT, self).__init__(root, label_path)

    def norm_insn(self, insn_list):
        insn_list_norm = []
        for insn in insn_list:
            # 分词2
            if insn[1] == 'STREAM':
                insn_norm = insn[1]
            else:
                insn_norm = insn[0] + '_' + insn[1]

            insn_list_norm.append(insn_norm)
        return insn_list_norm

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'org_after_doc2vec') #zgd

    def process(self):
        ''' process raw JSON ORGs '''

        # load doc2vec model
        file_path = r"PDFObj2Vec/PVDM/dataset/unix20_train_data"
        doc_ids, docs = load_data(file_path)
        print(len(doc_ids))

        vocab = Vocab([word for sentence in docs for word in sentence], min_count=1)
        print(f"Dataset comprises {len(docs)} documents and {len(vocab.words)} unique words (over the limit of {vocab.min_count} occurrences)")

        model = DistributedMemory(vec_dim=self.hidden_size, n_docs=len(doc_ids), n_words=len(vocab.words))
        model.load_state_dict(torch.load(self.doc2vec_path))
        device = torch.device(f'cuda:{self.device_id}')
        model = model.to(device)
        print('Loaded doc2vec model.')

        # TF-IDF
        # 计算词频(TF)
        total = sum(vocab.freqs.values())
        tf = {key: value / total for key, value in vocab.freqs.items()}

        # 计算逆文档频率(IDF)
        doc_sum = len(docs)
        idf = {}
        for sentence in docs:
            temp_freq = {}
            for word in sentence:
                if word not in temp_freq:
                    temp_freq[word] = 1
            for word in temp_freq.keys():
                if word in idf:
                    idf[word] += 1
                else:
                    idf[word] = 1
        for key, value in idf.items():
            idf[key] = math.log(doc_sum) - math.log(value)
        
        ## get data
        senkeys = ['/OpenAction', '/Action', '/JavaScript', '/JS', '/S']
        idx = 0
        print ('raw files counts: ', len(self.raw_file_names))
        for raw_path in tqdm(self.raw_file_names):
            fullpath = os.path.join(self.raw_dir, raw_path)

            # node attributes
            org_after_model = {}
            model.eval()
            with open(fullpath, 'r') as f:
                cfg = json.load(f)
                
                # 获取每个节点的嵌入
                for id1 in list(cfg.keys()):
                    block = cfg[id1]
                    seq_norm = Doc2VecDatasetCreator.norm_insn(block['insn_list'])
                    # 新文档的文档嵌入信息置0
                    org_after_model[id1] = np.array([0] * self.hidden_size)
                    org_after_model[id1] = org_after_model[id1].astype(float)

                    if len(seq_norm) == 0:
                        continue
                    for ins in seq_norm:
                        if ins not in vocab.word2idx.keys():
                            # 如果是没有见过的词汇，将其词向量置0(跳过)
                            continue
                        word_tensor = torch.tensor(vocab.word2idx[ins]).to(device)
                        with torch.no_grad():
                            word_vector = model.word_matrix[word_tensor,:].cpu().numpy()
                        # 整个文档的嵌入 = TF-IDF(每个词汇的嵌入)
                        org_after_model[id1] += tf[ins] * idf[ins] * word_vector.reshape(self.hidden_size)
                    # org_after_model[id1] = org_after_model[id1].detach()
            # word2vec_out = np.array([tensor.numpy() if tensor.nelement() > 1 else tensor.item() for tensor in org_after_model.values()])
            word2vec_out = np.array([i for i in org_after_model.values()])

            with open(fullpath, 'r', encoding='utf-8') as f:
                cfg = json.load(f)

            addr_to_id = dict()  # {str: int}
            current_node_id = -1
            for addr, block in cfg.items():  # addr is 'str
                current_node_id += 1
                addr_to_id[addr] = current_node_id

                
            ## y (label)
            y = int(self.labels[raw_path])
            

            # get sparse adjacent matrix
            edge_index = list()
            edge_attr = list()
            for addr, block in cfg.items():  # addr is `str`
                start_nid = addr_to_id[addr]
                isins = [num for elem in block['insn_list'] for num in elem]
                for out_edge in block['out_edge_list']:
                    if str(out_edge) in addr_to_id.keys():
                        end_nid = addr_to_id[str(out_edge)]
                        ## edge_index
                        edge_index.append([start_nid, end_nid])
                        intersection = set(senkeys) & set(isins)
                        if intersection:
                            edge_attr.append(1)
                        else:
                            edge_attr.append(0)
            

            # Data
            x = torch.tensor(word2vec_out)
            y = torch.tensor(y, dtype=torch.long).unsqueeze(0)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

            # save
            assert (self.raw_file_name_lookup(idx) == raw_path)
            save_path = 'data_{}_{}.pt'.format(idx, raw_path.split('.')[0])
            save_path = os.path.join(self.processed_dir, save_path)
            torch.save(data, save_path)

            idx += 1

def main():
    # parser = argparse.ArgumentParser(description='Generate torch geometric data')
    # parser.add_argument('-r', required=True, dest='base_dir', action='store', help='Base dir' )
    # parser.add_argument('-v', required=True, dest='vocab_path', action='store', help='Input vocab file path' )
    # parser.add_argument('-b', required=True, dest='bert_path', action='store', help='Input BERT model path' )
    # parser.add_argument('-label', required=True, dest='label_file', action='store', help='Input label file path' )
    #parser.add_argument('-o', required=True, dest='out_dir', action='store', help='Output ORG after prebert files dir')
    
    # args = parser.parse_args()

    # CIC-evasive
    base_dir = r"PDFObj2Vec/PVDM/dataset"
    # vocab_path = args.vocab_path
    doc2vec_path = r"PDFObj2Vec/PVDM/record/doc2vec_model.pth"
    label_path = r"PDFObj2Vec/PVDM/dataset/mal0406_label.csv"
    device_id = 1
    dataset = ORGDataset_Normalized_After_BERT(root=base_dir, doc2vec_path=doc2vec_path, 
                                               hidden_size=512, label_path=label_path, device_id=device_id)

    print(len(dataset))

if __name__ == '__main__':
    main()



