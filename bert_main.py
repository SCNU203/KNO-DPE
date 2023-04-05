import os
import sys
import pickle as pickle
import time
import sys
import numpy as np
import pandas as pd
import pretrainedmodels
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import transformers
from torch.optim import lr_scheduler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import math
import torch.nn.functional as F
import warnings
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore")
train_curve = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(3)

torch.cuda.empty_cache()
batch_size = 32
epoches = 100
hidden_size = 768
emb_size = 512
n_class = 76
All_test = []
maxlen = 250

file_path = '/random_mix_train.pk'
test_path = '/test.pk'
# points=["切线","垂径定理","勾股定理","同位角","平行线","三角形内角和","三角形中位线","平行四边形","相似三角形","正方形","圆周角","直角三角形","距离","邻补角","圆心角","圆锥的计算","三角函数","矩形","旋转","等腰三角形","外接圆",
#             "内错角","菱形","多边形","对顶角","三角形的外角","角平分线","对称","立体图形","三视图","圆内接四边形","垂直平分线","垂线","扇形面积","等边三角形","平移","含30度角的直角三角形","仰角","三角形的外接圆与外心","方向角","坡角","直角三角形斜边上的中线",
#             "位似","平行线分线段成比例","坐标与图形性质","圆柱的计算","俯角","射影定理","黄金分割","钟面角","多边形内角和","外接圆","弦长","长度","中垂线","相交线","全等三角形","梯形","锐角","补角","比例线段","比例角度","圆形","正多边形","同旁内角","余角","三角形的重心",
#             "旋转角","中心对称","三角形的内心","投影","对角线","弧长的计算","平移的性质","位似变换","菱形的性质","正方形的性质"]
points = ["切线","垂径定理","勾股定理","同位角","平行线","三角形内角和","三角形中位线","平行四边形","相似三角形",
        "正方形","圆周角","直角三角形","距离","邻补角","圆心角","圆锥的计算","三角函数","矩形","旋转","等腰三角形",
        "外接圆","内错角","菱形","多边形","对顶角","三角形的外角","角平分线","对称","立体图形","三视图","圆内接四边形",
        "垂直平分线","垂线","扇形面积","等边三角形","平移","含30度角的直角三角形","仰角","三角形的外接圆与外心","方向角",
        "坡角","直角三角形斜边上的中线","位似","平行线分线段成比例","坐标与图形性质","圆柱的计算","俯角","射影定理","黄金分割",
        "钟面角","多边形内角和","弦长","长度","中垂线","相交线","全等三角形","梯形","锐角","补角","比例线段","比例角度","圆形",
        "正多边形","同旁内角","余角","三角形的重心","旋转角","中心对称","三角形的内心","投影","对角线","弧长的计算",
        "平移的性质","位似变换","菱形的性质","正方形的性质"]

"""def read_data(file_path):
    f1 = open(file_path, 'rb')
    train_files = pickle.load(f1)
    train_subject = []
    train_label = []

    for i in train_files:
        subject = i['subject']
        train_subject.append(subject)

    for i in train_files:
        train_label.append(i['formal_point'])
    labels = []
    for item in train_label:
        item2=list(item)
        if len(item2)!=0:
            labels.append(points.index(item2[0]))
        else:
            labels.append(0)
    return train_subject,labels"""
single_zero = []
for i in range(0, 76):
    single_zero.append(0)


def read_data(file_path):
    f1 = open(file_path, 'rb')
    train_files = pickle.load(f1)
    train_subject = []
    train_label = []
    train_id = []
    train_image = []

    for i in train_files:
        subject = i['subject']
        train_subject.append(subject)
        train_id.append(i['id'])
        train_image.append(i['image'])

    for i in train_files:
        train_label.append(i['formal_point'])
    labels = []

    for item in train_label:
        item2 = list(item)
        if len(item2) != 0:
            single_label = []
            for i in range(0, 76):
                single_label.append(0)
            for j in item2:
                single_label[points.index(j)] = 1
            labels.append(single_label)
        else:
            labels.append(single_zero)
    return train_subject, labels


def get_test_labels(path):
    f = open(path, 'rb')
    data = pickle.load(f)
    test_label = []
    for item in data:
        test_label.append(item['formal_point'])
    labels = []
    for la in test_label:
        item2 = list(la)
        if len(item2) != 0:
            single_label = []
            for j in item2:
                single_label.append(points.index(j))
            labels.append(single_label)
        else:
            labels.append([0])
    return labels


sentences, labels = read_data(file_path)
labels = torch.tensor(labels)
labels = labels.to(torch.float)
test_sentences, test_labels = read_data(test_path)


# 数据加载及编码模块
class MyDataset(Data.Dataset):

    def __init__(self, sentences, labels=None, train_image=None, with_labels=True, ):
        self.tokenizer = AutoTokenizer.from_pretrained("./Roberta")
        self.with_labels = with_labels
        self.sentences = sentences
        self.labels = labels
        self.train_image = train_image

    def __len__(self):
        return len(sentences)

    def __getitem__(self, index):
        sent = self.sentences[index]

        encoded_pair = self.tokenizer(sent,
                                      padding='max_length',
                                      truncation=True,
                                      max_length=maxlen,
                                      return_tensors='pt')

        token_ids = encoded_pair['input_ids'].squeeze(0)
        attn_masks = encoded_pair['attention_mask'].squeeze(0)
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)

        if self.with_labels:
            label = self.labels[index]

            return token_ids, attn_masks, token_type_ids, label
        else:
            return token_ids, attn_masks, token_type_ids


train = Data.DataLoader(dataset=MyDataset(sentences, labels), shuffle=True, batch_size=batch_size, num_workers=10)


class Focal_Loss(nn.Module):
    def __init__(self, weight=0.25, gamma=2):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.i = 0

    def forward(self, preds, labels):
        """
        preds:softmax输出结果
        labels:真实值
        """
        eps = 1e-7
        y_pred = preds
        target = labels
        ce = -1 * torch.log(y_pred + eps) * target
        floss = torch.pow((1 - y_pred), self.gamma) * ce
        floss = torch.mul(floss, self.weight)
        floss = torch.sum(floss, dim=1)
        return torch.mean(floss)


# 模型定义模块
# class Nasnet():
#     # 返回不同的模型，后面是在EPOCH=10时，该模型的准确率
#     # net = Net()                            #64%
#     # net = torchvision.models.vgg11()      #73%
#     # net = torchvision.models.vgg11(pretrained=True)     #73%
#     # net = torchvision.models.googlenet()  #76%
#     # net = torchvision.models.resnet18()  # 77%
#     # net = torchvision.models.resnet18(pretrained=True)   #80%
#     # for param in net.parameters():#nn.Module有成员函数parameters()
#     # param.requires_grad = False
#     # Replace the last fully-connected layer
#     # Parameters of newly constructed modules have requires_grad=True by default
#     # net.fc = nn.Linear(512, 1000)#resnet18中有self.fc，作为前向过程的最后一层。
#     # net = torchvision.models.resnet50()   #71%
#     # net = torchvision.models.resnet50(pretrained=True)   #80%
#     model_name= 'nasnetalarge'
#     net=pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
#     net.avg_pool = nn.AdaptiveAvgPool2d(1)
#     net.last_linear=nn.Linear(in_features=4032,out_features=768,bias=True)
#
#     return net
class BertClassify(nn.Module):
    def __init__(self):
        super(BertClassify, self).__init__()
        self.bert = AutoModel.from_pretrained("./Roberta", output_hidden_states=True, return_dict=True)  # 构建bert层模型
        self.hidden_size = hidden_size
        self.emb_size=emb_size
        self.kno_pos = nn.Parameter(torch.randn((1, 1, 768)))
        self.kno_embedding = nn.Parameter(torch.randn((n_class, 768)))  # 全连接层分类输出
        self.lstm = torch.nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=2, dropout=0.5)
        self.lstm_embedding = nn.Embedding(22128, hidden_size, padding_idx=0)
        self.lstm_dropout = nn.Dropout(0.5)

        self.attn = nn.Linear(hidden_size * 3, hidden_size)
        # torch.nn.init.normal_(self.attn.weight, mean=0, std=1)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.norm = nn.LayerNorm(hidden_size)
        self.lstm_trans = nn.Linear(768, 512)
        self.albert_trans=nn.Linear(1024,768)

    def forward(self, X):

        # input_ids, attention_mask, token_type_ids ,image_input= X[0], X[1], X[2],X[3]
        input_ids, attention_mask = X[0], X[1]
        input_length = torch.sum(attention_mask, dim=1).long().view(-1, ).cpu()

        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        attention_mask = torch.cat([torch.ones((batch_size, 1)).cuda(), attention_mask], dim=-1)
        token_type = torch.cat(
            [torch.zeros((batch_size, 1)).cuda().long(), torch.ones((batch_size, seq_len)).cuda().long()], dim=-1)
        # embeds = self.bert.word_embedding(input_ids)
        embeds = self.bert.embeddings.word_embeddings(input_ids)
        embeds_ = torch.cat([self.kno_pos.repeat(batch_size, 1, 1), embeds], dim=1)

        outputs = self.bert(inputs_embeds=embeds_, attention_mask=attention_mask,
                            token_type_ids=token_type)  # 返回一个output字典
        # outputs = outputs.pooler_output.unsqueeze(1)
        outputs= outputs.last_hidden_state[:, 0, :]
        print(outputs.size())
        outputs=self.albert_trans(outputs)

        # lstm_embedding = self.lstm_dropout(self.lstm_embedding(input_ids))
        lstm_embedding = self.lstm_embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(lstm_embedding, input_length, batch_first=True, enforce_sorted=False)
        lstm_output, _ = self.lstm(packed)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
        lstm_output = lstm_output[:, :, :self.hidden_size] + lstm_output[:, :, self.hidden_size:]
        lstm_output = lstm_output[:, 0, :] + lstm_output[:, -1, :]
        lstm_output=self.lstm_trans(lstm_output)
        # embedding = self.embedding.unsqueeze(0).repeat(outputs.size()[0],1,1)
        # att_outputs, _ = self.attn(outputs.unsqueeze(1),embedding,embedding)
        # # nasnet_output=self.nasnet(image_input)  #用模型的最终输出来做connect
        # att_outputs = self.layer_norm(att_outputs).squeeze(1)
        # outputs = outputs + att_outputs                                                                     #   将attention结果与原始bert输出相加，再输入全连接层分类输出
        # # nasnet_output=self.con_norm(nasnet_output).squeeze(1)
        # mix_output=outputs+ 0.3*nasnet_output
        logits = (outputs + lstm_output).mm(self.kno_embedding.transpose(0, 1))
        print((outputs+lstm_output).size())
        print(self.kno_embedding.size())
        print(logits.size())

        sys.exit(0)
        return logits


bc = BertClassify().to(device)


def train_model():  # 训练函数

    bert_params = list(map(id, bc.bert.parameters()))
    base_params = filter(lambda p: id(p) not in bert_params, bc.parameters())

    optimizer = torch.optim.AdamW([
        {"params": bc.bert.parameters(), "lr": 2e-5},
        {"params": base_params},
    ], lr=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    loss_fn = nn.BCEWithLogitsLoss()
    sum_loss = 0
    total_step = len(train)
    for epoch in range(epoches):
        scheduler.step()
        print(epoch, "   ")
        t = tqdm(enumerate(train))
        for i, batch in t:
            bc.train()
            optimizer.zero_grad()
            batch = tuple(p.to(device) for p in batch)
            # pred = bc([batch[0], batch[1], batch[2],batch[4]])
            pred = bc([batch[0], batch[1]])
            # m=nn.Softmax(dim=1)
            # pred = m(pred)
            # labe=torch.zeros(batch[3].size()[0],77)
            # labe=labe.to(device)
            # teni = 0
            # for item in batch[3]:
            #     labe[teni, int(item)] = 1
            #     teni = teni + 1

            loss = loss_fn(pred, batch[3])
            # loss = loss_fn(pred, labe)
            sum_loss += loss.item()
            loss.backward()
            optimizer.step()
            t.set_postfix(loss=loss.item())
        train_curve.append(sum_loss)
        sum_loss = 0
        bc.eval()
        test_model(epoch)  # 测试函数


def test_model(epoch: int):
    bc.eval()
    save_index = []
    with torch.no_grad():
        test_text = test_sentences
        # test_label=test_labels # 测试方式1
        test_label = get_test_labels(test_path)  # 测试方式二

        # image_test=test_image
        test = MyDataset(test_text, labels=None, with_labels=False)
        count = 0
        for i in range(0, len(test_text)):
            x = test.__getitem__(i)
            x = tuple(p.unsqueeze(0).to(device) for p in x)
            pred = bc([x[0], x[1]])
            pred = torch.sigmoid(pred)
            pred = pred.tolist()
            temp = pred[0]
            index = []

            for item in range(0, len(temp)):
                if temp[item] >= 0.8:
                    index.append(item)

            item =test_label[i]
            if item==index:
                count = count + 1

        All_test.append(count / 755)
        print('ACC:', count / 755)


        if epoch == 50:
            with open('/home1/caojie/knowledgepoints/Albert/with_token_epoch50.pickle', 'wb') as op:
                pickle.dump(bc, op)


if __name__ == '__main__':
    time1 = time.time()
    train_model()
    print("********train finished!!*********")
    print("The Best Acc:", max(All_test), "Best epoch:", All_test.index(max(All_test)))
    time2 = time.time()
    print(time2 - time1)
