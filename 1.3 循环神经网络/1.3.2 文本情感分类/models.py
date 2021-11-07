from torch import nn
from lib import ws
import torch
import torch.nn.functional as F


class IMDBLstmmodel(nn.Module):
    def __init__(self):
        super(IMDBLstmmodel, self).__init__()
        self.hidden_size = 64 # 隐藏层神经元数量
        self.num_layer = 2  # 层数
        self.embedding_dim = 200 # 每个词语的维度
        self.bidriectional = True # 是否双向的lstm
        self.bi_num = 2 if self.bidriectional else 1
        self.dropout = 0.4 # dropout的比例，默认值为0。dropout是一种训练过程中让部分参数随机失活的一种方式，能够提高训练速度
        # 以上部分为超参数
        self.embedding = nn.Embedding(len(ws.dict),self.embedding_dim)
        self.lctm = nn.LSTM(self.embedding_dim,self.hidden_size,self.num_layer,bidirectional=True,dropout=self.dropout)
        self.fc = nn.Linear(self.hidden_size*self.num_layer,2)
        # self.fc2 = nn.Linear(20,2)


    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1,0,2)
        h_0,c_0 = self.init_hidden_state(x.size(1))
        out,(h_n,c_n) = self.lctm(x,(h_0,c_0))

        out = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=-1)
        out = self.fc(out)
        # out = F.relu(out)
        # out = self.fc2(out)

        return F.log_softmax(out,dim=-1)



    def init_hidden_state(self,batch_size):
        h_0 = torch.rand(self.num_layer*self.bi_num,batch_size,self.hidden_size)
        c_0 = torch.rand(self.num_layer*self.bi_num,batch_size,self.hidden_size)
        return h_0,c_0


