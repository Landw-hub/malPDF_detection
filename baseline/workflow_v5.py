# by Side Liu
# 2024.07.24
# word2vec_unix20_512

import os
import sys
import dataset_word2vec
import shutil
import eval3_all

_current_dir = os.path.abspath(os.path.dirname(__file__)) # 获取当前脚本所在目录的绝对路径
PROJECT_ROOT = os.path.normpath(os.path.join(_current_dir, '..')) # 通过返回上一级目录来确定项目的根目录
sys.path.append(PROJECT_ROOT) # 将项目根目录添加到Python的模块搜索路径中

from word2vec.model import *
from word2vec.word2vec_data import *
from word2vec.train import *


# 生成org_after_prebert
def gen_org_after_prebert(base_dir, word2vec_path, hidden_size, label_path, device_id):
    orgdataset = dataset_word2vec.ORGDataset_Normalized_After_BERT(root=base_dir, word2vec_path=word2vec_path, hidden_size=hidden_size, label_path=label_path, device_id=device_id)

contagio_train_base = '/home/dell/data/SD/PDFObj2Vec/word2vec/dataset/unix20_train'
contagio_test_base = '/home/dell/data/SD/PDFObj2Vec/word2vec/dataset/unix20_test'
contagio_label_path = '/home/dell/data/SD/PDFObj2Vec/word2vec/dataset/usenix20_labels.csv'

newben_base = '/home/dell/data/SD/PDFObj2Vec/word2vec/dataset/ben0406'
newben_label_path = '/home/dell/data/SD/PDFObj2Vec/word2vec/dataset/benign0406_label.csv'

cicmal_base = '/home/dell/data/SD/PDFObj2Vec/word2vec/dataset/mal0406'
cicmal_label_path = '/home/dell/data/SD/PDFObj2Vec/word2vec/dataset/mal0406_label.csv'


adv_base = '/home/dell/data/SD/PDFObj2Vec/word2vec/dataset/adv'
adv_label_path = '/home/dell/data/SD/PDFObj2Vec/word2vec/dataset/adv_label.csv'

# gen_org_after_prebert(newben_base, vocab_path, bert_path, hidden_size, newben_label_path, out_dir, device_id=2)

def del_all(dir):
    for fn in os.listdir(dir):
        os.remove(os.path.join(dir, fn))

bert_path = '/home/dell/data/SD/PDFObj2Vec/word2vec/record_v2/save_model_unix20_512'
log_path = '/home/dell/data/SD/PDFObj2Vec/baseline/record/word2vec_unix20_512'
hidden_size = 512 # unix20 vec
out_dir = 'org_after_word2vec'  # 保存.ptPDF文件的文件夹
device_id = 0 # GPU

for id in range(0,100):
    print(f"Model is {id}:\n")
    bert_fp = os.path.join(bert_path, f"word2vec_model_epoch{id}.pth")
    log = f"word2vec_model_epoch{id}.log.txt"

    if os.path.exists(os.path.join(log_path, log)):
        continue
    # 清除历史文件
    try:
        del_all(os.path.join(contagio_train_base, out_dir))
        del_all(os.path.join(contagio_test_base, out_dir))
        del_all(os.path.join(adv_base, out_dir))
        del_all(os.path.join(cicmal_base, out_dir))
        del_all(os.path.join(newben_base, out_dir))
        print ('### Delete all ###')
    except:
        pass


    # 开始生成图文件
    gen_org_after_prebert(contagio_train_base, bert_fp, hidden_size, contagio_label_path, device_id)
    gen_org_after_prebert(contagio_test_base, bert_fp, hidden_size, contagio_label_path, device_id)
    gen_org_after_prebert(adv_base, bert_fp, hidden_size, adv_label_path, device_id)
    gen_org_after_prebert(cicmal_base, bert_fp, hidden_size, cicmal_label_path, device_id)
    gen_org_after_prebert(newben_base, bert_fp, hidden_size, newben_label_path, device_id)

    train_path = os.path.join(contagio_train_base, out_dir)
    test_path = os.path.join(contagio_test_base, out_dir)
    adv_path = os.path.join(adv_base, out_dir)
    cicmal_path = os.path.join(cicmal_base, out_dir)
    newben_path = os.path.join(newben_base, out_dir)

    try:
        # 开始评估
        cmdlines = f"python3 /home/dell/data/SD/PDFObj2Vec/baseline/eval3_all.py -tr {train_path} -te {test_path} -adv {adv_path} -cic {cicmal_path} -ben {newben_path} -r {log_path} -o {log} -id {device_id}"
        os.system(cmdlines)
    except:
        print (cmdlines)
        pass