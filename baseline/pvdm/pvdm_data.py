# by Guodong Zhou
# 2024.4.5

import os
import json
import random
import tqdm
import argparse
import numpy as np
from collections import Counter, OrderedDict


class Doc2VecDatasetCreator(object):
    ''' 获取Doc2Vec语料, 为了后续的节点嵌入, 我们将一个obj视为一个doc '''

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
    

    def normalized_format(self, pairs_count=50):
        out_path = self.out_dir + '/unix20_train_data'
        store_f = open(out_path, 'a+')

        # 文档id
        doc_id = 0
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
                    
                    store_f.write(str(doc_id) + ' ')
                    doc_id += 1
                    for ins in first_seq_norm:
                        store_f.write(ins + ' ')
                    store_f.write('\n')
                    store_f.write(str(doc_id) + ' ')
                    doc_id += 1
                    for ins in second_seq_norm:
                        store_f.write(ins + ' ')
                    store_f.write('\n')

                    count += 1
                    if count == pairs_count:
                        break
                
        store_f.close()


if __name__ == '__main__':
    word2vec_dataset_creator = Doc2VecDatasetCreator(r"/home/dell/data/SD/PDFObj2Vec/PVDM/dataset/unix20_org_train", r"PDFObj2Vec/PVDM/dataset")
    word2vec_dataset_creator.normalized_format()