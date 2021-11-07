import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as py


# 模拟数据
x = torch.rand([50,1])
y = x*3+0.8


class Lr(nn.Module):

    def __init__(self):
        super(Lr, self).__init__()
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        out = self.linear(x)
        return out

# 定义模型
model = Lr()
# 定义损失
criteria = nn.MSELoss()
# 定义优化器
optimizer = optim.SGD(model.parameters(),lr=0.001)

for i in range(30000):
    out = model(x)
    loss = criteria(y,out)

    # 梯度置为0
    optimizer.zero_grad()
    # 重新计算梯度 反向传播
    loss.backward()
    optimizer.step()

    if i % 500 == 0:
        params = list(model.parameters())
        # print('Epoch[{}/{}], loss: {:.6f}'.format(i, 30000, loss.data))
        print("loss:{},w = {},b = {}".format(loss.data,params[0],params[1]))

# 模型评估
model.eval()
y_predict = model(x).data.numpy()
plt.scatter(x.numpy(),y.numpy(),c='r')
plt.plot(x.numpy(),y_predict)
plt.show()



























