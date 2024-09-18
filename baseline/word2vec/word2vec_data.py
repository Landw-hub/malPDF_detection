# by Guodong Zhou
# 2024.3.24

import os
import json
import random
import tqdm
import argparse
import numpy as np
from collections import Counter, OrderedDict
# from torchtext.vocab import vocab
# from torchtext.transforms import VocabTransform


class Word2VecDatasetCreator(object):
    ''' 获取Word2Vec语料, 其中每个obj视为一个句子 '''

    def __init__(self, org_dir, out_dir):
        self.org_dir = org_dir
        self.out_dir = out_dir
        self.file_iters = os.listdir(self.org_dir)


    @staticmethod
    def norm_insn(insn_list):
        insn_list_norm = []
        for insn in insn_list:
            if insn[1] == 'STREAM':
                insn_norm = insn[1]
            else:
                insn_norm = insn[0] + '_' + insn[1]

            insn_list_norm.append(insn_norm)
        return insn_list_norm
    
    # 取有间接引用的obj，提高数据集的质量
    def normalized_format(self, pairs_count=50):
        out_path = self.out_dir + '/unix20_train_data'
        store_f = open(out_path, 'a+')

        for path in tqdm.tqdm(self.file_iters):
            fullpath = os.path.join(self.org_dir, path)
            with open(fullpath, 'r') as f:
                cfg = json.load(f)

                count = 0
                key_list = list(cfg.keys())
                random.shuffle(key_list)
                for id1 in key_list:
                    block = cfg[id1]
                    if len(block['out_edge_list']) == 0:
                        continue
    
                    # first block
                    first_seq_norm = self.norm_insn(block['insn_list'])
                    # second block
                    id2 = None
                    for out_edge in block['out_edge_list']:
                        if str(out_edge) != id1:
                            id2 = str(out_edge)
                            break
                    if id2 is None or str(id2) not in cfg.keys():
                        continue
                    second_seq_norm = self.norm_insn(cfg[str(id2)]['insn_list'])
                    
                    for ins in first_seq_norm:
                        store_f.write(ins + ' ')
                    store_f.write('\n')
                    for ins in second_seq_norm:
                        store_f.write(ins + ' ')
                    store_f.write('\n')

                    count += 1
                    if count == pairs_count:
                        break

        store_f.close()


class Word2VecDatasetProcess(object):
    ''' 构造词汇表, 获取正负样本等等'''

    def __init__(self, data_file, min_freq):
        ''' 
        Self_attr:
            self.datafile: 文件路径
            self.min_freq: 可接受的最小词频
            self.sentences_list: 语料库中的每个句子
            self.words_list: 语料库中的所有词汇(可重复)
            self.sentences_count: 语料库的句子数
            self.words_count: 语料库的词汇数(可重复)
            self.words_freq: 语料库的词频计数
            self.id_word: 词汇索引to词汇
            self.word_id: 词汇to词汇索引
        '''
        self.data_file = data_file
        self.min_freq = min_freq
        self.get_data()
        self.get_vocab_transform()


    def get_vocab_transform(self):
        counter = Counter(self.words_list)
        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
        self.words_freq = ordered_dict
        
        special_token = ["<UNK>"] # 未知单词返回<UNK>
        self.id_word = special_token + list(ordered_dict.keys())
        self.word_id = {}
        for i in range(len(self.id_word)):
            self.word_id[self.id_word[i]] = i  # 词汇与索引间的转化关系


    def get_data(self):
        with open(self.data_file, 'r', encoding='utf-8') as file:
            sentences_list = [line.strip().split() for line in file]
            random.shuffle(sentences_list) # 随机打乱
            words_list = [item for sublist in sentences_list for item in sublist]

        self.sentences_list = sentences_list
        self.words_list = words_list

        self.sentence_index = 0    # 保证每个batch取出的样本不同
        self.sentences_count = len(sentences_list)
        self.words_count = len(words_list)
        

    def get_pospair_count(self, window_size):
        # 计算整个语料库中包含的正样本对的个数
        return self.words_count * (2*window_size) - window_size * (window_size+1) * self.sentences_count


    def get_pospair(self, batch_size, window_size):
        # 获取正样本
        # sentence_index = 0
        pos_pairs = []
        while len(pos_pairs) < batch_size:
            sentence = self.sentences_list[self.sentence_index]
            sentence = [self.word_id[word] for word in sentence]
                
            for i, center_word in enumerate(sentence):
                # 获取中心词的背景词
                start = max(0, i - window_size)
                end = min(len(sentence) - 1, i + window_size)
                context_words = [sentence[j] for j in range(start, end+1) if j != i]

                for context_word in context_words:
                    pos_pairs.append((center_word, context_word))

            # 防止越界，循环使用语料库
            self.sentence_index = (self.sentence_index + 1) % self.sentences_count

        return pos_pairs[:batch_size]

    # def get_pospair(self, batch_size, window_size):
    #     # 获取正样本
    #     total_word_freq = sum(self.words_freq.values())
    #     pos_pairs = []
        
    #     while len(pos_pairs) < batch_size:
    #         sentence = self.sentences_list[self.sentence_index]
    #         sentence = [self.word_id[word] for word in sentence]
                    
    #         for i, center_word in enumerate(sentence):
    #             # 减少高频词的抽取率
    #             word_freq = self.words_freq[self.id_word[center_word]]
    #             z = word_freq / total_word_freq
    #             p = (np.sqrt(z / 0.001) + 1) * (0.001 / z)      # 参照word2vec的论文公式
                
    #             # 以概率p保留该中心词
    #             if np.random.rand() > p:
    #                 continue
                
    #             # 获取中心词的背景词
    #             start = max(0, i - window_size)
    #             end = min(len(sentence) - 1, i + window_size)
    #             context_words = [sentence[j] for j in range(start, end + 1) if j != i]

    #             for context_word in context_words:
    #                 pos_pairs.append((center_word, context_word))
    
    #         # 防止越界，循环使用语料库
    #         self.sentence_index = (self.sentence_index + 1) % self.sentences_count

    #     return pos_pairs[:batch_size]


    # # 负采样获取背景词
    # def get_negv(self, pos_pairs, neg_count):
    #     # 负采样频率设置为词频率的0.75次方，提高低频词的选中概率
    #     sam_frequency = np.array(list(self.words_freq.values()))**0.75
        
    #     # 归一化为采样概率
    #     sam_probability = sam_frequency / sum(sam_frequency)

    #     # 按词频来挑选负样本，注意避开<UNK>
    #     neg_v = np.random.choice(range(1, len(sam_probability)+1), replace=True, size=(len(pos_pairs), neg_count), p=sam_probability).tolist()
    #     return neg_v
    

    # # 负采样获取背景词（避免与正样本重复，太慢了
    # def get_negv(self, pos_pairs, neg_count):
    #     # 负采样频率设置为词频率的0.75次方，提高低频词的选中概率
    #     sam_frequency = np.array(list(self.words_freq.values()))**0.75
    
    #     # 归一化为采样概率
    #     sam_probability = sam_frequency / np.sum(sam_frequency)

    #     # 创建排除正样本词的候选词表
    #     vocab_indices = np.arange(1, len(sam_probability) + 1)
    
    #     neg_v = []
    #     for pos_u, pos_v in pos_pairs:
    #         exclude_indices = np.array([pos_u, pos_v])
    #         mask = np.isin(vocab_indices, exclude_indices, invert=True)
    #         filtered_indices = vocab_indices[mask]
    #         filtered_probs = sam_probability[mask]
    #         filtered_probs /= np.sum(filtered_probs)  # 归一化概率
        
    #         # 按词频来挑选负样本
    #         neg_samples = np.random.choice(filtered_indices, replace=True, size=neg_count, p=filtered_probs).tolist()
    #         neg_v.append(neg_samples)

    #     return neg_v

    # 直接避免所有的正样本词，加速
    def get_negv(self, pos_pairs, neg_count):
        sam_frequency = np.array(list(self.words_freq.values()))**0.75
        sam_probability = sam_frequency / np.sum(sam_frequency)

        # 创建排除正样本词的候选词表
        vocab_indices = np.arange(1, len(sam_probability) + 1)
        exclude_indices = set()
        for pos_u, pos_v in pos_pairs:
            exclude_indices.add(pos_u)
            exclude_indices.add(pos_v)

        mask = np.isin(vocab_indices, list(exclude_indices), invert=True)
        filtered_indices = vocab_indices[mask]
        filtered_probs = sam_probability[mask]
        filtered_probs /= np.sum(filtered_probs)

        neg_v = np.random.choice(filtered_indices, replace=True, size=(len(pos_pairs), neg_count), p=filtered_probs).tolist()

        return neg_v



if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Init Word2vec Data')
    # parser.add_argument('-i', required=True, dest='org_dir', action='store', help='Input ORG files dir' )
    # parser.add_argument('-o', required=True, dest='out_dir', action='store', help='Output Word2vec data file dir' )
    # args = parser.parse_args()

    # word2vec_dataset_creator = Word2VecDatasetCreator(args.org_dir, args.out_dir)
    word2vec_dataset_creator = Word2VecDatasetCreator(r"/home/dell/data/SD/PDFObj2Vec/word2vec/dataset/unix20_org_train", r"PDFObj2Vec/word2vec/dataset")
    word2vec_dataset_creator.normalized_format()
