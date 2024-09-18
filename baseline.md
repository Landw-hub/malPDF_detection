# 数据集生成

Word2Vec 和 PVDM 的生成语料库方法相同。在预训练阶段，将 PDF_org 文件按比例划分为训练集和测试集。对于训练集和测试集中的每个 PDF_org 文件，我们随机抽取 50 对 obj 对象，每对 obj 之间存在间接引用关系。我们将每个 obj 视为一个句子，该句子由多个词组成。

# Word2Vec 训练

Word2Vec 包含 CBOW 和 Skip-Gram 两种模型，这两者都是基于预测相邻词语实现的。CBOW 的方法是用背景词预测中心词，而 Skip-Gram 则是用中心词预测背景词。这里我们使用的是 Skip-Gram 模型。

首先，Word2Vec 会统计训练集中的所有词汇，最小词频设置为 1，得到一个 `word_to_id` 词典。由于训练集中只有 obj 句子，没有标签，因此需要自行构造正负样本。以下是一个简单的例子来展示如何生成正负样本：

例如，对于一个 obj 句子 `/Type_NAME /Parent_REF /MediaBox_NUM_LIST /Resources_DICT /Resources/ProcSet_NAME_LIST /Resources/XObject_DICT ...`，假设句子由 10 个词组成，表示为 `(1,2,3,4,5,6,7,8,9,10)`，设定 `window_size`（采样窗口）的大小为 3，`neg_count`（每个正样本生成的负样本数量）为 5。那么对于中心词 5，它的背景词是 2、3、4、6、7、8。因此可以得到这个句子的所有正样本，如 `(1,2)`、`(1,3)`、`(1,4)`、`(2,1)`、`(2,3)`、`(2,4)`、`(2,5)` 等。

对于负样本的生成，通常在同一语句中，对于那些不在中心词附近（即不在采样窗口内）的词汇，可以视为负背景词（例如对于词 1 来说，词 5、6、7、8、9、10 都是负背景词）。不过这种方式生成的样本数量太多，可能导致模型训练速度过慢，因此我们采用优化的负采样方法，即对每个中心词，在所有词汇中随机选择 `neg_count` 个负背景词。为了提高低频词的选中概率，我们将负采样频率设为词频的 0.75 次方，同时避开中心词。这样每个正样本都会对应 `neg_count` 个负样本。

Word2Vec 模型由两个嵌入层组成，分别用于学习中心词和背景词的嵌入，还包括一个 Dropout 层。优化目标是负对数似然损失函数，即让正样本的中心词和背景词的相似度尽可能大，而负样本的相似度尽可能小。模型训练完毕后，可以输出每个词的向量表示（维度为 512 或 1024）。

## Word2vec 模型结构
```python
batch_size: 4096
epochs: 100
lr: 1e-4
self.optimizer = optim.SparseAdam(self.model.parameters(), lr=self.initial_lr)
self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
```
```python
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, emb_dimension, dropout=0.5):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dimension = emb_dimension

        self.u_embedding = nn.Embedding(self.vocab_size, self.emb_dimension, sparse=True)
        self.v_embedding = nn.Embedding(self.vocab_size, self.emb_dimension, sparse=True)
        self.dropout = nn.Dropout(dropout)

        initrange = 0.5 / self.emb_dimension
        self.u_embedding.weight.data.uniform_(-initrange, initrange)
        self.v_embedding.weight.data.uniform_(0, 0)

    def forward(self, pos_u, pos_v, neg_v):
        pos_embu = self.u_embedding(pos_u)
        pos_embv = self.v_embedding(pos_v)
        pos_score = torch.sum(torch.mul(pos_embu, pos_embv).squeeze(), dim=1)
        pos_score2 = F.logsigmoid(pos_score)

        neg_embv = self.v_embedding(neg_v)
        neg_score = torch.bmm(neg_embv, pos_embu.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-1*neg_score)

        batch_loss = -1 * (torch.sum(pos_score2) + torch.sum(neg_score))
        pos_pred = torch.sigmoid(pos_score) > 0.5
        batch_correct = torch.sum(pos_pred).item()
        batch_total = pos_pred.size(0)

        return batch_loss, batch_correct, batch_total
```

## TF-IDF 加权平均

训练好的 Word2Vec 模型只能用于生成词嵌入，而在 GIN 中我们需要 obj 的嵌入。obj 由一个个词组成，常见的方法是将每个词的向量累加平均得到 obj 的向量表示。然而，这种做法可能会导致恶意词汇被稀释，所以我们不使用简单的平均加权处理，而是使用 TF-IDF 加权平均进行优化。TF-IDF 的原理是基于词频（每个词在整个训练集中出现的次数，即 `TF = word_count/sum(all_words_count)`）和文档频率（将一个 obj 视为一个文档，计算每个词在每个文档中的出现次数，即 `IDF = math.log(all_objs_count) - math.log(word_count / sum(all_objs_count))`），最后得到的每个词的权重为 `TF*IDF`，再做加权平均得到句向量。

PS：TF-IDF中提到了每个词在每个文档中的出现次数，比如对于(1,1,3)(1,4,5)(2,6)这三句话而言，1出现的次数为2。

# PVDM 训练

PVDM 与 Word2Vec 原理类似，都是基于相邻词汇的关系实现的。但与 Word2Vec 不同的是，PVDM 认为同一文档中每个词汇之间的关系比不同文档中的词汇之间的关系更紧密。

PVDM 的词汇表构成方法与 word2vec 一致。

在 PVDM 中，每个 obj 被视为一个文档，每个文档都有自己的文档 ID，文档 ID 从 0 开始累加直至遍历完训练集。每个文档由多个词组成。

PVDM 中设置的 `window_size` 和 `neg_count` 与 Word2Vec 一致，使用的正负采样方法也相同。最终得到的正负样本格式为 `(word 所在文档 ID，中心词，背景词)`，比 Word2Vec 多了一个文档 ID。

## PVDM 模型结构
```python
batch_size: 4096
epochs: 100
lr: 5e-4
self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
```
```python
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
```

## PVDM优化目标

```python
class NegativeSampling(nn.Module):
    def __init__(self):
        super(NegativeSampling, self).__init__()
        self.log_sigmoid = nn.LogSigmoid()

    def forward(self, scores):
        batch_size = scores.shape[0]
        positive = self.log_sigmoid(scores[:, 0])
        negatives = torch.sum(self.log_sigmoid(-scores[:, 1:]), dim=1)
        return -torch.sum(positive + negatives) / batch_size  # average for batch
```

训练完成后，PVDM 模型可以通过 `self.word_matrix` 层输出每个词汇的向量表示。最终的 obj 句向量通过与 Word2Vec 相同的 TF-IDF 加权平均方法得到。这些嵌入将用于分类任务。

# 词汇表
```
- 最小词频: 1
- Word2Vec (Unix20): 5331
- Word2Vec (Contagio++): 31522
- PVDM (Unix20): 5401
- PVDM (Contagio++): 32495
```