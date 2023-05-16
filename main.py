import numpy as np
import random
import time
import datetime

import sklearn
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# 设定超参数
SEED = 123
BATCH_SIZE = 16
learning_rate = 5e-5  # 0.00002,学习率设置过大，则结果不好
weight_decay = 1e-2  # 0.01
epsilon = 1e-8

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def readFile(filename):
    with open(filename, encoding='utf-8') as f: content = f.readlines()
    return content


pos_text = readFile(r'D:\Your-Path\pos.txt')
neg_text = readFile(r'D:\Your-Path\neg.txt')
sentences = pos_text + neg_text  # 将读取到的两个文件的数据合并成一个sentences

pos_targets = np.ones([len(pos_text)])  # (5000, ) 创建一个1*5000的一维单位矩阵
neg_targets = np.zeros([len(neg_text)])  # (5000, ) 创建一个1*5000的一维零矩阵
targets = np.concatenate((pos_targets, neg_targets), axis=0).reshape(-1, 1)  # (10000, 1) 变成1000*1矩阵，与上面拼接的矩阵匹配
# axis是拼接的方向为竖着拼，行数更改。reshape(-1,1):-1表示自动计算行/列数
total_targets = torch.tensor(targets)

model_name = 'bert-base-chinese'
cache_dir = './sample_data/'  # 若没有，则自动创建这个文件夹

tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)  # 加载预训练模型
print("pos_targets: ", pos_targets.ndim)
print("neg_targets", neg_targets)
print(len(pos_text))  # pos和neg都是各自5000条评论
print(pos_text[2])
print(tokenizer.tokenize(pos_text[2]))
print(tokenizer.encode(pos_text[2]))
print(tokenizer.convert_ids_to_tokens(tokenizer.encode(pos_text[2])))


# 将每一句话处理为等长，固定值为126，大于126做截断，小于126做 Padding，加上首位两个标识，长度总共等于128
def convert_text_to_token(tokenizer, sentence, limit_size=126):
    tokens = tokenizer.encode(sentence[:limit_size])
    if len(tokens) < limit_size + 2:  # +2是因为有首尾'CLS'和'SEP'
        tokens.extend([0] * (limit_size + 2 - len(tokens)))
    return tokens


input_ids = [convert_text_to_token(tokenizer, sen) for sen in sentences]
input_tokens = torch.tensor(input_ids)
print(input_tokens.shape)


# 建立mask
def attention_mask(input_ids):
    atten_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        atten_masks.append(seq_mask)
    return atten_masks


atten_masks = attention_mask(input_ids)
attention_tokens = torch.tensor(atten_masks)
print(attention_tokens.shape)

# 划分训练集和测试集
from sklearn.model_selection import train_test_split

train_inputs, test_inputs, train_labels, test_labels = \
    train_test_split(input_tokens, total_targets, random_state=666,
                     test_size=0.1)  # random_state：是随机数的种子，若填写0或者不填，则每次随机数数组是不一样的

train_masks, test_masks, _, _ = \
    train_test_split(attention_tokens, input_tokens, random_state=666, test_size=0.1)

print(train_inputs.shape, test_inputs.shape)  # [8000, 128]   [2000, 128]
print(train_masks.shape)  # [8000, 128]

print(train_inputs[0])
print(train_masks[0])

train_data = TensorDataset(train_inputs, train_masks, train_labels)
# TensorDataset 可以用来对 tensor 进行打包，就好像 python 中的 zip 功能。
train_sampler = RandomSampler(train_data)  # RandomSampler：对数据集随机采样。SequentialSampler：按顺序对数据集采样。
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

# 查看train_dataloader的内容
for i, (train, mask, label) in enumerate(train_dataloader):
    print(train.shape, mask.shape, label.shape)  # [16, 128]   [16, 128]   [16, 1]
    break

print('len(train_dataloager) = ', len(train_dataloader))  # 500

# 创建模型
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 表示分类的个数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  # 代表将模型加载到指定设备上。

# 定义优化器
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=epsilon)  # transformers 库实现了基于权重衰减的优化器，AdamW
# eps也是AdamW为了数值稳定的参数 epsilon超参：1e-8.  learning_rate  超参 = 1e-2

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': weight_decay
     },
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0
     }
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=epsilon)
epochs = 2
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


def binary_acc(preds, labels):
    correct = torch.eq(torch.max(preds, dim=1)[1], labels.float()).float()
    acc = correct.sum().item() / len(correct)
    return acc



def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train(model, optimizer):
    t0 = time.time()
    avg_loss, avg_acc = [], []

    model.train()  # 作用是启用batch normalization和drop out。 测试阶段使用 model.eval()
    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:  # 每隔40个batch 输出一下所用时间.
            elapsed = format_time(time.time() - t0)
            print('Batch {:>5,} of {:>5,}. Elapsed:{:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids, b_input_mask, b_labels = batch[0].long().to(device), batch[1].long().to(device), batch[2].long().to(device)
        output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss, logits = output[0], output[1]
        avg_loss.append(loss.item())
        acc = binary_acc(logits, b_labels)  ## (predict, label)
        avg_acc.append(acc)
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)  # 大于1的梯度将其设为1.0, 以防梯度爆炸
        optimizer.step()  # 更新模型参数
        scheduler.step()  # 更新learning rate

    avg_acc = np.array(avg_acc).mean()
    avg_loss = np.array(avg_loss).mean()
    return avg_loss, avg_acc


def evaluate(model):
    avg_acc = []
    model.eval()  # 表示进入测试模式

    with torch.no_grad():
        for batch in test_dataloader:
            b_input_ids, b_input_mask, b_labels = batch[0].long().to(device), batch[1].long().to(device), batch[2].long().to(device)
            output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            acc = binary_acc(output[0], b_labels)
            avg_acc.append(acc)

    avg_acc = np.array(avg_acc).mean()
    return avg_acc

for epoch in range(epochs):
    train_loss, train_acc = train(model, optimizer)
    print('epoch={},训练集准确率={}， 损失={}'.format(epoch, train_acc, train_loss))

    test_acc = evaluate(model)
    print('epoch = {},测试准确率={}'.format(epoch, test_acc))

def predict(sen):
    input_id = convert_text_to_token(tokenizer, sen)
    input_token = torch.tensor(input_id).long().to(device)

    atten_mask = [float(i > 0) for i in input_id]
    attention_token = torch.tensor(atten_mask).long().to(device)

    output = model(input_token.view(1, -1), token_type_ids=None, attention_mask=attention_token.view(1, -1))
    print(output[0])

    return torch.max(output[0], dim=1)[1]

label = print('酒店位置难找，环境不太好，隔音差，下次不会再来的。')
print('好评' if label == 1 else '差评')

label = print('酒店还可以，接待人员很热情，卫生合格，空间也比较大，不足的地方就是没有窗户')
print('好评' if label == 1 else '差评')

label = print('服务各方面没有不周到的地方, 各方面没有没想到的细节')
print('好评' if label == 1 else '差评')
