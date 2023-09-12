# -*- coding:utf-8 -*-
import sys

from pytorch_transformers import BertModel
import torch
import torch.nn as nn
from model.base_model import base_model
from torchcrf import CRF
sys.path.append("../")
from layers.utensil import _generate_mask
import pickle

nn.Module
#注意pretrain_output_size这个参数，是bert的维度加上word2vec，这里bert-base的维度是768，word2vec维度是200，所以是968，规范的话可以分开写
class bert_bilstm_crf(base_model):
    def __init__(self, pretrain_model_path = None, pretrain_output_size = 400, lstm_hidden_size = 384,
                 num_layers = 1, dropout_ratio = 0.5, batch_first = True, bidirectional = True, lable_num = 4, device = "cpu"):
        super(bert_bilstm_crf, self).__init__()

        # self.bert = BertModel.from_pretrained("./bert-base-uncased/bert-base-uncased/")
        self.w2v_emb = pickle.load(open("./word2vec/w2v_nocls.pkl", "rb"))#读取之前训练的word2vec向量，
        self.w2v_emb = torch.from_numpy(self.w2v_emb)#把向量转成torch tensor
        self.w2v_emb = nn.Embedding.from_pretrained(self.w2v_emb)#建立查询表，加载事先训练好的word2vec权重
        self.bilstm = nn.LSTM(
                input_size=pretrain_output_size,
                hidden_size=lstm_hidden_size,
                num_layers=num_layers,
                dropout=dropout_ratio,
                batch_first=batch_first,
                bidirectional=bidirectional
            )

        # self.dropout = SpatialDropout(drop_p)
        # self.layer_norm = LayerNorm(hidden_size * 2)

        self.dropout = nn.Dropout(p=dropout_ratio)

        self.linear = nn.Linear(lstm_hidden_size * 2, lable_num)

        self.crf = CRF(lable_num, batch_first = batch_first)
#         self.crf = CRF(lable_num)

        self.device = device

        print("模型加载完成")

    def forward(self, x):

        x = x.to(self.device).long()
        # z = z.to(self.device).long()
        # segments_ids = torch.zeros(x.shape, dtype=torch.long).to(self.device)
        #
        # emb_outputs = self.bert(input_ids=x, attention_mask=z, token_type_ids=segments_ids) #
        #
        # bert_emb = emb_outputs[0]#bert的最后一层输出

        w2v_emb = self.w2v_emb(x)#word2vec的输出

        # bert_w2v = torch.cat((bert_emb, w2v_emb), 2)#将二者拼接

        # bilstm_output, _ = self.bilstm(bert_w2v)
        bilstm_output, _ = self.bilstm(w2v_emb)

        drop_out = self.dropout(bilstm_output)

        linear_output = self.linear(drop_out)

        return linear_output

    def get_loss(self, linear_output, max_len, sen_len, y = None, use_cuda = True):
        y = y.to(self.device).long()
        log_likelihood = self.crf(linear_output, y,
                                  mask=_generate_mask(sen_len, max_len, use_cuda),
                                  reduction='mean')
        
        return -log_likelihood

    @torch.no_grad()
    def decode(self, dev_x, max_len = None, sen_len = None, use_cuda = None, dev_y = None):
        dev_x = torch.tensor(dev_x, dtype=torch.long).to(self.device)

        output = self.forward(x=dev_x)
        loss = None
        if dev_y != None:
            dev_y = torch.tensor(dev_y, dtype=torch.long).to(self.device)
            loss = self.get_loss(output, max_len, sen_len, y = dev_y, use_cuda = use_cuda)
        return self.crf.decode(output, mask=_generate_mask(sen_len, max_len, use_cuda)), loss
