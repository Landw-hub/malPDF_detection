#################################
##### by Side Liu 2024.3.17######
#################################

from torch_geometric.data import Data
from dgl.data import DGLDataset
import os
import copy
import torch
from torch import nn
import dgl
from dgl.data.utils import load_graphs
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dgl.nn import GraphConv, GINConv
import numpy as np
from torch.utils.data.dataloader import default_collate
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import random

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn import GraphConv, GINConv

def pyg_to_dgl(pyg_graph):
    # 创建DGL图
    dgl_graph = dgl.graph((pyg_graph.edge_index[0], pyg_graph.edge_index[1]))
    # 添加节点特征
    dgl_graph.ndata['x'] = pyg_graph.x
    # 添加边特征
    dgl_graph.edata['edge_attr'] = pyg_graph.edge_attr
    # 返回DGL图和图标签
    return dgl_graph, pyg_graph.y



class OrgDGLDataset(DGLDataset):
    def __init__(self, root):
        self.root = root
        super(OrgDGLDataset, self).__init__(name='org_dgl_dataset')
        self.load()  # 直接在__init__中调用load方法来处理数据

    def load(self):
        # 用于加载和处理数据的方法
        self.graphs = []
        self.labels = []
        for filename in os.listdir(self.root):
            if filename.endswith('.pt') and filename != 'pre_filter.pt' and filename != 'pre_transform.pt': # by zgd
                file_path = os.path.join(self.root, filename)
                pyg_graph = torch.load(file_path)
                
                # 以下是将PyG图转换为DGL图的逻辑
                # 请根据实际情况调整
                if pyg_graph.edge_index.size(0) > 0:
                    src, dst = pyg_graph.edge_index
                else:
                    src, dst = [], []

                
                dgl_graph = dgl.graph((src, dst), num_nodes=pyg_graph.x.shape[0])
                if hasattr(pyg_graph, 'x'):
                    dgl_graph.ndata['feat'] = pyg_graph.x
                if hasattr(pyg_graph, 'edge_attr'):
                    dgl_graph.edata['feat'] = pyg_graph.edge_attr
                
                self.graphs.append(dgl_graph)
                self.labels.append(pyg_graph.y)

    def __getitem__(self, idx):
        # 获取单个图和标签
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        # 数据集中图的数量
        return len(self.graphs)

def load_pdf_org_dataset(dataset_dir):
    dataset = OrgDGLDataset(root=dataset_dir)

    feature_dim = dataset[0][0].ndata['feat'].shape[1]

    labels = torch.tensor([x[1] for x in dataset])
    num_classes = torch.max(labels).item() + 1
    
    dataset = [(g.remove_self_loop().add_self_loop(), y) for g, y in dataset]

    print(f"******** # Num Graphs: {len(dataset)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")

    return dataset, (feature_dim, num_classes)

def collate(samples):
    # `samples` 是一个列表，其中包含了被 `Dataset.__getitem__` 方法返回的数据对
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_labels = default_collate(labels)  # 对于标签，可以使用default_collate
    return batched_graph, batched_labels



class GIN(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes):
        super(GIN, self).__init__()
        self.conv1 = GINConv(nn.Linear(in_feats, hidden_feats), 'mean')
        self.conv2 = GINConv(nn.Linear(hidden_feats, hidden_feats), 'mean')
        #self.conv3 = GINConv(nn.Linear(hidden_feats, hidden_feats), 'mean')
        self.classify = nn.Linear(hidden_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        #h = F.relu(h)
        #h = self.conv3(g, h)
        g.ndata['h'] = h
        hg = dgl.max_nodes(g, 'h')
        return self.classify(hg)
    

# 定义评估函数
def evaluate(model, data_loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batched_graph, labels in data_loader:
            batched_graph = batched_graph.to(device)
            labels = labels.to(device).squeeze()
            features = batched_graph.ndata['feat'].float().to(device)
            edge_weight = batched_graph.edata['feat'].to(device)
            logits = model(batched_graph, features)
            _, indices = torch.max(logits, dim=1)
            y_true.extend(labels.detach().cpu().numpy())
            y_pred.extend(indices.detach().cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='binary')
    precision = precision_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn)

    return acc, recall, precision, f1, fpr

def evaluate_adv(model, device, data_loader):
    model.eval()  # 设置模型为评估模式
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batched_graph, labels in data_loader:
            batched_graph = batched_graph.to(device)
            labels = labels.to(device).squeeze()
            features = batched_graph.ndata['feat'].float().to(device)
            edge_weight = batched_graph.edata['feat'].to(device)
            logits = model(batched_graph, features)
            _, indices = torch.max(logits, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(indices.detach().cpu().numpy())
    # 计算指标
    yp = np.array(y_pred)
    return (len(yp[yp==1])), len(yp)
    #print (len(yp[yp==1]), '/', len(yp))

def evaluate_ben(model, device, data_loader):
    model.eval()  # 设置模型为评估模式
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batched_graph, labels in data_loader:
            batched_graph = batched_graph.to(device)
            labels = labels.to(device).squeeze()
            features = batched_graph.ndata['feat'].float().to(device)
            edge_weight = batched_graph.edata['feat'].to(device)
            logits = model(batched_graph, features)
            _, indices = torch.max(logits, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(indices.detach().cpu().numpy())

    # 计算指标
    yp = np.array(y_pred)
    return (len(yp[yp==0])), len(yp)



# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# train_dataset, (feature_dim1,feature_dim2) = load_pdf_org_dataset('../preprocess_data/usenix20_train/org_after_prebert97')
# adv_train_dataset, (feature_dim1,feature_dim2) = load_pdf_org_dataset('../preprocess_data/usenix20_train/org_after_prebert97(includeadv)')
# test_dataset, (feature_dim3,feature_dim4) = load_pdf_org_dataset('../preprocess_data/usenix20_test/org_after_prebert97')
# adv_dataset, (feature_dim5,feature_dim6) = load_pdf_org_dataset('../preprocess_data/reverse_mimicry_wine08/org_after_prebert97')
# mal0406_dataset, (_1, _2) = load_pdf_org_dataset('../preprocess_data/mal0406/org_after_prebert97')
# allben_pred0_dataset, (_1, _2) = load_pdf_org_dataset('../preprocess_data/allben_pred0_by_previous/org_after_prebert97')

# def setup_seed(seed):
#      torch.manual_seed(seed)
#      torch.cuda.manual_seed_all(seed)
#      np.random.seed(seed)
#      random.seed(seed)
#      torch.backends.cudnn.deterministic = True
# # 设置随机数种子
# setup_seed(42)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate)
# advtrain_loader = DataLoader(adv_train_dataset, batch_size=128, shuffle=True, collate_fn=collate)
# test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate)
# adv_loader = DataLoader(adv_dataset, batch_size=128, shuffle=False, collate_fn=collate)
# allben_pred0_loader = DataLoader(allben_pred0_dataset, batch_size=256, shuffle=False, collate_fn=collate)
# mal0406_loader = DataLoader(mal0406_dataset, batch_size=256, shuffle=False, collate_fn=collate)

def train(model, device, data_loader, val_loader, adv_loader, newben_loader, cicall_loader, epochs, records_dir, log_file):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    records_dir = records_dir
    if not os.path.exists(records_dir):
        os.mkdir(records_dir)
    record_fd = open(os.path.join(records_dir, log_file), 'w+')
    model.train()
    for epoch in range(epochs):
        for batched_graph, labels in data_loader:
            batched_graph = batched_graph.to(device)
            features = batched_graph.ndata['feat'].float().to(device)
            edge_weight = batched_graph.edata['feat']
            logits = model(batched_graph, features).to(device)
            labels = labels.to(device).squeeze()  # 确保标签是正确的维度
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 在测试集上评估
        acc, recall, precision, f1, fpr = evaluate(model, val_loader, device)
        adv_hits = evaluate_adv(model, device, adv_loader)
        newben_hits = evaluate_ben(model, device, newben_loader)
        cic_hits = evaluate_adv(model, device, cicall_loader)

        print(f'Epoch {epoch} | Acc: {acc:.4f} | Rec: {recall:.4f} | Prec: {precision:.4f} | F1: {f1:.4f} | FPR: {fpr:.4f},  Adv: {adv_hits}, Newben: {newben_hits}-{newben_hits[0]/newben_hits[1]:.4f}, CIC: {cic_hits}-{cic_hits[0]/cic_hits[1]:.4f}')
        # model_dict = copy.deepcopy(model.state_dict())
        # fp = os.path.join(records_dir, '{}.pth'.format(epoch))
        record_fd.write(f'Epoch {epoch} | Acc: {acc:.4f} | Rec: {recall:.4f} | Prec: {precision:.4f} | F1: {f1:.4f} | FPR: {fpr:.4f},  Adv: {adv_hits}, Newben: {newben_hits}-{newben_hits[0]/newben_hits[1]:.4f}, CIC: {cic_hits}-{cic_hits[0]/cic_hits[1]:.4f} \n')
        # torch.save(model_dict, fp)
    record_fd.close()

def train_baseline(model, device, data_loader, val_loader, adv_loader, epochs, records_dir, log_file):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    records_dir = records_dir
    if not os.path.exists(records_dir):
        os.mkdir(records_dir)
    record_fd = open(os.path.join(records_dir, log_file), 'w+')
    model.train()
    for epoch in range(epochs):
        for batched_graph, labels in data_loader:
            batched_graph = batched_graph.to(device)
            features = batched_graph.ndata['feat'].to(device)
            edge_weight = batched_graph.edata['feat']
            logits = model(batched_graph, features).to(device)
            labels = labels.to(device).squeeze()  # 确保标签是正确的维度
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 在测试集上评估
        acc, recall, precision, f1, fpr = evaluate(model, val_loader, device)
        adv_hits = evaluate_adv(model, device, adv_loader)
       

        print(f'Epoch {epoch} | Acc: {acc:.4f} | Rec: {recall:.4f} | Prec: {precision:.4f} | F1: {f1:.4f} | FPR: {fpr:.4f},  Adv: {adv_hits}')
        # model_dict = copy.deepcopy(model.state_dict())
        # fp = os.path.join(records_dir, '{}.pth'.format(epoch))
        record_fd.write(f'Epoch {epoch} | Acc: {acc:.4f} | Rec: {recall:.4f} | Prec: {precision:.4f} | F1: {f1:.4f} | FPR: {fpr:.4f},  Adv: {adv_hits} \n')
        # torch.save(model_dict, fp)
    record_fd.close()



# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
# device = torch.device('cuda:3')
# # 初始化模型、优化器和损失函数
# model = GIN(in_feats=768, hidden_feats=256, num_classes=2).to(device)


# epochs = 50
# record_dir = './org_after_prebert_usenix20_768'
# train(model, device, train_loader, test_loader, adv_loader, allben_pred0_loader, mal0406_loader, epochs, record_dir)