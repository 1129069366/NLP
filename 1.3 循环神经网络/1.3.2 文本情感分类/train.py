from models import *
from torch import optim
import torch.nn as nn
from dataset import *
import pandas


imdb_model = IMDBLstmmodel()   # 创建模型
optimizer = optim.Adam(imdb_model.parameters(),lr=0.0001)  # 创建优化器

train_batch_size = 128
test_batch_size= 5000

def train(epoch):
    mode = True
    train_dataloader = getDataLoader(mode,train_batch_size)
    for idx,(target,input) in enumerate(train_dataloader):
        # 梯度置为0
        optimizer.zero_grad()
        out = imdb_model(input)
        loss = F.nll_loss(out,target)
        loss.backward()
        optimizer.step()

        print("epoch:{},idx:{},loss:{}".format(epoch,idx,loss.item()))
        if idx % 100 == 0:
            torch.save(imdb_model.state_dict(), "./model/model.pkl")
            torch.save(optimizer.state_dict(), './model/optimizer.pkl')

if __name__ == '__main__':
    for i in range(10):
        train(i)






