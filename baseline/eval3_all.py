from ginmodel import load_pdf_org_dataset, GIN, train, train_baseline
import torch
import os
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import dgl
import argparse


# 设置随机数种子
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def collate(samples):
    # `samples` 是一个列表，其中包含了被 `Dataset.__getitem__` 方法返回的数据对
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_labels = default_collate(labels)  # 对于标签，可以使用default_collate
    return batched_graph, batched_labels









if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate BERT Data')
    parser.add_argument('-tr', required=True, dest='train_dir', action='store', help='Input train dataset' )
    parser.add_argument('-te', required=True, dest='test_dir', action='store', help='Input test dataset' )
    parser.add_argument('-adv', required=True, dest='adv_dir', action='store', help='Input adv dataset' )
    parser.add_argument('-cic', required=True, dest='cic_dir', action='store', help='Input adv dataset' )
    parser.add_argument('-ben', required=True, dest='ben_dir', action='store', help='Input adv dataset' )
    parser.add_argument('-r', required=True, dest='record_dir', action='store', help='record_dir' )
    parser.add_argument('-o', required=True, dest='log_file', action='store', help='output log' )
    parser.add_argument('-id', required=True, dest='device_id', action='store', help='device id' )
    args = parser.parse_args()

    train_dataset = args.train_dir
    test_dataset = args.test_dir
    adv_dataset = args.adv_dir
    cicmal_dataset = args.cic_dir
    newben_dataset = args.ben_dir
    log_file = args.log_file    
    record_dir = args.record_dir
    deviceid = args.device_id



    train_dataset, (feature_dim1,feature_dim2) = load_pdf_org_dataset(train_dataset)
    test_dataset, (feature_dim3,feature_dim4) = load_pdf_org_dataset(test_dataset)
    adv_dataset, (feature_dim5,feature_dim6) = load_pdf_org_dataset(adv_dataset)
    mal0406_dataset, (_1, _2) = load_pdf_org_dataset(cicmal_dataset)
    allben_pred0_dataset, (_1, _2) = load_pdf_org_dataset(newben_dataset)

    setup_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate)
    adv_loader = DataLoader(adv_dataset, batch_size=128, shuffle=False, collate_fn=collate)
    allben_pred0_loader = DataLoader(allben_pred0_dataset, batch_size=256, shuffle=False, collate_fn=collate)
    mal0406_loader = DataLoader(mal0406_dataset, batch_size=256, shuffle=False, collate_fn=collate)



    
    device = torch.device(f"cuda:{deviceid}")
    # 初始化模型、优化器和损失函数
    model = GIN(in_feats=512, hidden_feats=256, num_classes=2).to(device)


    epochs = 50
    # record_dir = './org_after_prebert_usenix20_768_prebert0'
    # log_file = 'prebert0_log.txt'
    train(model, device, train_loader, test_loader, adv_loader, allben_pred0_loader, mal0406_loader, epochs, record_dir,log_file)
    # train_baseline(model, device, train_loader, test_loader, adv_loader, epochs, record_dir,log_file)


     