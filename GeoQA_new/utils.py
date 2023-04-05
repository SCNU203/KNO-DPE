import sys
from allennlp.data.fields import *
from allennlp.data.tokenizers import Token
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from transformers import AutoModel, AutoTokenizer
import math
from typing import *
from overrides import overrides
import numpy as np
import pickle
import torch
import torch.nn as nn
import cv2 as cv
# torch.manual_seed(123)
# torch.cuda.set_device(0)
# device = torch.device('cuda:0')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
maxlen = 252
hidden_size = 768
n_class = 76
tokenizer = AutoTokenizer.from_pretrained("data/pretrain/Roberta")
points=["切线","垂径定理","勾股定理","同位角","平行线","三角形内角和","三角形中位线","平行四边形","相似三角形",
        "正方形","圆周角","直角三角形","距离","邻补角","圆心角","圆锥的计算","三角函数","矩形","旋转","等腰三角形",
        "外接圆","内错角","菱形","多边形","对顶角","三角形的外角","角平分线","对称","立体图形","三视图","圆内接四边形",
        "垂直平分线","垂线","扇形面积","等边三角形","平移","含30度角的直角三角形","仰角","三角形的外接圆与外心","方向角",
        "坡角","直角三角形斜边上的中线","位似","平行线分线段成比例","坐标与图形性质","圆柱的计算","俯角","射影定理","黄金分割",
        "钟面角","多边形内角和","弦长","长度","中垂线","相交线","全等三角形","梯形","锐角","补角","比例线段","比例角度","圆形",
        "正多边形","同旁内角","余角","三角形的重心","旋转角","中心对称","三角形的内心","投影","对角线","弧长的计算",
        "平移的性质","位似变换","菱形的性质","正方形的性质"]
knowledge_list=['圆的切线垂直于过其切点的半径，切线和圆心的距离等于圆的半径。','在一个圆中，垂直于弦的直径平分这条弦，并且平分弦所对的两条弧。','在平面上的一个直角三角形中，两个直角边边长的平方加起来等于斜边长的平方。','两条直线被第三条直线所截在第三条直线同侧形成的两个角称为同位角，两直线平行同位角相等。',
                '两直线平行同位角相等，内错角相等，同旁内角互补。','三角形内角和等于C_3。','三角形的中位线平行于第三边，并且等于第三边的一半。','平行四边形内角和为C_4，对边相等且平行，对角相等，邻角互补。','相似三角形对应角相等，对应线段和周长的比等于相似比，对应面积比等于相似比的平方。',
                '正方形四个角都等于C_2, 四条边长度相等。','一条弧所对圆周角等于它所对圆心角的一半。','直角三角形其中一个角为C_2，另外两个角的和也等于C_2。','假设A点坐标为(a,b), B点坐标为(c,d)，则AB两点之间的距离为√{(a²-c²)+(b²-d²)}。','两个角有一条公共边，它们的另一边互为反向延长线，具有这种关系的两个角，叫做邻补角，互为邻补角两个角相加等于C_3。',
                '圆心角所对应的顶点是圆心，两条边为圆的半径，圆心角的大小是同一条弧对应的圆周角大小的两倍。','假设圆锥底面半径为R，母线长为L，圆锥体的高为H，那么圆锥的侧面积=C_5*R*L，圆锥的体积为frac{1}{3}*C_5*R*2*H。','在直角三角形ABC中，∠B为锐角，则sinB等于∠B所对的直角边与斜边的比，cosB等于∠B的邻边与斜边的比，tanB等于直角边与邻边的比。',
                '矩形四个内角均为C_2，矩形的对边相等。','图形旋转前后对应点到旋转中心的距离相等，旋转前后图形全等。','等腰三角形两个底角相等，底角对应的两条边相等，是轴对称图形。','多边形的各顶点都在某个圆上，则该圆为多边形的外接圆，圆心称为多边形的外心。','两条平行直线被第三条直线所截,内错角相等。',
                '菱形四边相等，对边平行且相等，对角线互相垂直。','正n边形的内角和为(n-2)*C_3','一个角的两边是另一个角的反向延长线，那么这两个角称为对顶角，互为对顶角的两个角相等。','三角形的外角与它相邻的内角互补，三角形的外角等于和它不相邻的两个内角的和。','角平分线把角平分为两个完全相等的角，角平分线上的点到角的两边的距离相等。',
                '如果两个图形成轴对称那么对称轴是对应点连接线的垂直平分线。','常见的立体几何图形有正方体、长方体、圆柱、圆锥等。','三视图的主视图和俯视图长相等、主视图和左视图高相等、左视图和俯视图的宽相等。','圆内接四边形的对角互补，其任意一个外角等于他的内对角。',
                '线段的垂直平分线垂直且平分其所在的线段，垂直平分线上的任意一点到两端点的距离相等。','两条互相垂直的线段互为对方的垂线。','假设扇形的半径为R，n是扇形的圆心角度数，L是扇形对应的弧长，则扇形的面积= frac{(n*π*R²)}{C_4}。','等边三角形三条边相等，三个内角都等于C_1。','图形沿直线平移后对应角和边不变。',
                '直角三角形中，C_0角所对的直角边是斜边长的一半，斜边上的中线是斜边的一半。','当观察者抬头望一物件时，其视线与水平线的夹角称为仰角。','三角形的外接圆圆心称为外心，三角形的外心到三角形的各边距离相等。','方向角指的是采用某坐标轴方向作为标准方向所确定的方位角，常用方向角描述有东北、西北、东南、西南。',
                '坡面与水平面的家教叫做坡角，坡角表示了坡度的大小。','直角三角形斜边上的中线长度为斜边的一半，中线把三角形分为面积相等的连个三角形，斜边上的中点是三角形的外心。','两个位似图形具有相似图形的一切性质，此外位似图形的对应顶点均在同一直线上。','平行线分线段成比例定理指的是两条直线被一组平行线（不少于3条）所截，截得的对应线段的长度成比例。',
                '可以通过图形的顶点坐标信息求解图形的面积，相关边长等信息。','设圆柱的底面半径为R，高为H，则圆柱的体积为C_5*R²*H，圆柱的侧面积为2*C_5*R*H。','当观察者低头望一物件时，其视线与水平线的夹角称为俯角。','在直角三角形ABC中，AD是斜边AC上的高，那么根据射影定理有BD²=AD*CD，AB²=AC*AD，BC²=CD*AC。','把一个线段切割为两部分，其中较长部分与全长部分的比值约等于0.618，则这个线段分割比例即为黄金分割。',
                '将时钟表盘分割成12部分，其中每一部分大小为C_0,每n部分组成的角称为钟面角，钟面角的大小为n*C_0。','正n边形的内角和为(n-2)*C_3。','弦长为连接圆上任意两点的线段的长度。','线段、直线、曲线均具有长度属性。','垂直平分线垂直且平分其所在线段，垂直平分线上任意一点到线段端点的距离相等。','两条不平行的直线经过延长后总能相交于一点。','两个三角形全等时，其对应边和对应角相等。',
                '若梯形的上边长为a, 底边长为b，梯形的高为h,那么梯形的面积等于：(a+b)*h*0.5。','小于C_2的角称为锐角。','若两个角的大小和等于C_3，则两个角互为补角。','如果四条线段的长度分别为abcd,且frac{a}{b}=frac{c}{d}，则四条线段称为比例线段。','两个三角形相似，对应角度相等。',
                '设圆的半径为R，则圆的周长为2*π*R，直径为2*R，面积为π*R²。','正多边形的每条边都相等，每个内角都相等，正n边形内角和为(n-2)*C_3。','两条平行线被第三条直线所截，平行线内形成两个夹角，称为同旁内角，同旁内角互补。','若两个角大小和等于C_2，则两个角互为余角。','三角形三条边的中线交点为重心，重心到顶点的距离为重心到对边距离的两倍，重心和顶点组成的三个三角形面积相等。',
                '平面中图形绕着某点旋转一个角度，这个角度称为旋转角。','将图形绕着某点旋转C_3度，如果旋转后的图形与原图像重合，则该图形是中心对称图形。','三角形的内心到各边的距离相等，若三角形三边长分别为a、b、c，S为三角形的面积，则三角形的内切圆半径等于2*S/(a+b+c)。','令投射线通过点或其他物体，向选定的投影面投射，并在该面上得到图形的方法称为投影法。','正多边形的对角线平分该多边形，菱形和正方形的对角线互相垂直。',
                '假设某圆弧对应的半径为R，对应的圆心角为α，则该圆弧长度为frac{α*π*R}{C_3}。','图形经过平移后所有边角不变。','将图形的所有边角按照等比例进行缩放，缩放得到的图形的边角与原图形的边角成比例。','菱形四边相等，对边平行，对角线互相垂直，菱形的面积计算方法与平行四边形一样。','正方形四个角都等于C_2, 四条边长度相等，正方形的面积等于边长的平方。']
# knowledge_list=["切线","垂径定理","勾股定理","同位角","平行线","三角形内角和","三角形中位线","平行四边形","相似三角形","正方形","圆周角","直角三角形","距离","邻补角","圆心角","圆锥的计算","三角函数","矩形","旋转","等腰三角形","外接圆",
#             "内错角","菱形","多边形","对顶角","三角形的外角","角平分线","对称","立体图形","三视图","圆内接四边形","垂直平分线","垂线","扇形面积","等边三角形","平移","含30度角的直角三角形","仰角","三角形的外接圆与外心","方向角","坡角","直角三角形斜边上的中线",
#             "位似","平行线分线段成比例","坐标与图形性质","圆柱的计算","俯角","射影定理","黄金分割","钟面角","多边形内角和","弦长","长度","中垂线","相交线","全等三角形","梯形","锐角","补角","比例线段","比例角度","圆形","正多边形","同旁内角","余角","三角形的重心",
#             "旋转角","中心对称","三角形的内心","投影","对角线","弧长的计算","平移的性质","位似变换","菱形的性质","正方形的性质"]
def read_data(file_path):
    f1 = open(file_path, 'rb')
    train_files = pickle.load(f1)
    train_subject = []
    train_label = []
    train_id=[]
    train_image=[]

    for i in train_files:
        subject = i['subject']
        train_subject.append(subject)
        train_id.append(i['id'])
        train_image.append(i['kno_image'])

    for i in train_files:
        train_label.append(i['formal_point'])
    labels = []

    for item in train_label:
        item2=list(item)
        if len(item2)!=0:
            labels.append(points.index(item2[0]))
        else:
            labels.append(0)
    return train_subject,labels,train_image

def get_test_labels(path):
    f=open(path,'rb')
    data=pickle.load(f)
    test_label=[]
    for item in data:
        test_label.append(item['formal_point'])
    labels=[]
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

def process_image(img, min_side=224):  # 等比例缩放与填充
    size = img.shape
    h, w = size[0], size[1]
    # 长边缩放为min_side
    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w/scale), int(h/scale)
    resize_img = cv.resize(img, (new_w, new_h))
    # 填充至min_side * min_side
    # 下右填充
    top, bottom, left, right = 0, min_side-new_h, 0, min_side-new_w
    pad_img = cv.copyMakeBorder(resize_img, int(top), int(bottom), int(left), int(right),
                                cv.BORDER_CONSTANT, value=[255,255,255]) # 从图像边界向上,下,左,右扩的像素数目
    return pad_img
# def get_mask(sent):
#     encoded_pair = tokenizer(sent,
#                                       padding='max_length',
#                                       truncation=True,
#                                       max_length=maxlen,
#                                       return_tensors='pt')
#
#     token_ids = encoded_pair['input_ids'].squeeze(0)
#     attn_masks = encoded_pair['attention_mask'].squeeze(0)
#     token_type_ids = encoded_pair['token_type_ids'].squeeze(0)
#     return token_ids,attn_masks,token_type_ids


# with open('/home1/caojie/knowledgepoints/epoch80_gpu3.pickle', 'rb') as f:
#     bc = pickle.load(f)
@DatasetReader.register("s2s_manual_reader")
class SeqReader(DatasetReader):

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 source_token_indexer: Dict[str, TokenIndexer] = None,
                 target_token_indexer: Dict[str, TokenIndexer] = None,
                 model_name: str = None) -> None:
        super().__init__(lazy=False)
        self._tokenizer = tokenizer
        self._source_token_indexer = source_token_indexer
        self._target_token_indexer = target_token_indexer
        self._model_name = model_name

        sub_dict_path = "data/sub_dataset_dict.pk"  # problems type
        with open(sub_dict_path, 'rb') as file:
            subset_dict = pickle.load(file)
        self.subset_dict = subset_dict
        self.all_points = ['切线', '垂径定理', '勾股定理', '同位角', '平行线', '三角形内角和', '三角形中位线', '平行四边形',
                  '相似三角形', '正方形', '圆周角', '直角三角形', '距离', '邻补角', '圆心角', '圆锥的计算', '三角函数',
                  '矩形', '旋转', '等腰三角形', '外接圆', '内错角', '菱形', '多边形', '对顶角', '三角形的外角', '角平分线',
                  '对称', '立体图形', '三视图', '圆内接四边形', '垂直平分线', '垂线', '扇形面积', '等边三角形', '平移',
                  '含30度角的直角三角形', '仰角', '三角形的外接圆与外心', '方向角', '坡角', '直角三角形斜边上的中线', '位似',
                  '平行线分线段成比例', '坐标与图形性质', '圆柱的计算', '俯角', '射影定理', '黄金分割', '钟面角', '多边形内角和', '弦长', '长度', '中垂线',
                  '相交线', '全等三角形', '梯形', '锐角', '补角', '比例线段', '比例角度', '圆形', '正多边形', '同旁内角', '余角', '三角形的重心', '旋转角', '中心对称',
                  '三角形的内心', '投影', '对角线','弧长的计算' , '平移的性质' , '位似变换' ,'菱形的性质' ,'正方形的性质']

    @overrides
    def _read(self, file_path: str):
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
            for sample in dataset:
                yield self.text_to_instance(sample)
    @overrides
    def text_to_instance(self, sample) -> Instance:

        # token_ids,att_masks,token_type_ids=get_mask(sample['subject'])
        #
        # with torch.no_grad():
        #     subject=sample['subject']
        #     token_list=sample['token_list']
        #     x=(token_ids,att_masks,token_type_ids,sample['kno_image'])
        #     x = tuple(p.unsqueeze(0).to(device) for p in x)
        #     pred=bc([x[0],x[1],x[2],x[3]])
        #     m = nn.Softmax(dim=1)
        #     pred = m(pred)
        #     pred = pred.tolist()
        #     temp = pred[0]
        #     index = temp.index(max(temp))
        #     knowledge_point = knowledge_list[index]
        #     sample['subject']=yizhi+knowledge_point+subject
        #     sample['token_list']=list(yizhi)+list(knowledge_point)+token_list
        fields = {}
        image = sample['image']
        image = process_image(image)
        image = image/255
        img_rgb = np.zeros((3, image.shape[0], image.shape[1]))
        for i in range(3):
            img_rgb[i, :, :] = image

        fields['image'] = ArrayField(img_rgb)
        # kno_image=sample['kno_image']
        # kno_image_numpy=ArrayField(kno_image)
        # fields['kno_image']=kno_image_numpy

        s_token = self._tokenizer.tokenize(' '.join(sample['token_list']))
        fields['source_tokens'] = TextField(s_token, self._source_token_indexer)
        t_token = self._tokenizer.tokenize(' '.join(sample['manual_program']))
        t_token.insert(0, Token(START_SYMBOL))
        t_token.append(Token(END_SYMBOL))
        fields['target_tokens'] = TextField(t_token, self._target_token_indexer)
        fields['subject'] = MetadataField(sample['subject'])
        fields['source_nums'] = MetadataField(sample['numbers'])
        fields['choice_nums'] = MetadataField(sample['choice_nums'])
        fields['formal_point'] = MetadataField(sample['formal_point'])
        fields['label'] = MetadataField(sample['label'])

        type = self.subset_dict[sample['id']]
        fields['type'] = MetadataField(type)
        fields['data_id'] = MetadataField(sample['id'])
        equ_list = []

        equ = sample['manual_program']
        equ_token = self._tokenizer.tokenize(' '.join(equ))
        equ_token.insert(0, Token(START_SYMBOL))
        equ_token.append(Token(END_SYMBOL))
        equ_token = TextField(equ_token, self._source_token_indexer)
        equ_list.append(equ_token)

        fields['equ_list'] = ListField(equ_list)
        fields['manual_program'] = MetadataField(sample['manual_program'])

        point_label = np.zeros(76, np.float32)
        exam_points = sample['formal_point']

        for point in exam_points:
            point_id = self.all_points.index(point)
            point_label[point_id] = 1
        fields['point_label'] = ArrayField(np.array(point_label))
        return Instance(fields)

