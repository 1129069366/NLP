import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn

# learning_rate = 0.01
#
# # 1. 准备数据
# # y = 3x +0.8
# x = torch.rand([500,1])
# y_true = x*0.3 + 0.8
#
#
# # 2. 通过模型计算y_predict
# w = torch.rand([1,1],requires_grad=True)
# b = torch.tensor(0,requires_grad=True,dtype=torch.float)
# print(b)
#
#
# # 4.通过循环 反向传播 更新参数
# for i in range(5000):
#     y_predict = torch.matmul(x, w) + b
#
#     # 3.计算损失
#     loss = (y_true - y_predict).pow(2).mean()
#     if w.grad is not None:
#         w.data.zero_()
#
#     if b.grad is not None:
#         b.data.zero_()
#
#     loss.backward()
#     w.data = w.data-w.grad*learning_rate
#     b.data = b.data-b.grad*learning_rate
#
#     print("w ,b,loss",w.item(),b.item(),loss.item())

# 1.准备数据
x = torch.rand(50)
y_true = x*0.3+0.8

# 2.初始化参数
w = torch.rand(1,requires_grad=True)
b = torch.rand(1,requires_grad=True)


def loss_fn(y_true,y_predict):
    """
    计算损失的函数
    :param y_true:
    :param y_predict:
    :return:
    """
    loss = (y_true-y_predict).pow(2).mean()
    # 反向传播
    for i in [w,b]:
        if i.grad is not None:
            i.grad.data.zero_()
    loss.backward()
    return loss.data


def optimize(learning_rate):
    """
    更新权重的函数
    :param learning_rate:
    :return:
    """
    w.data -= learning_rate*w.grad.data
    b.data -= learning_rate*b.grad.data


for i in range(3000):
    y_predict = x*w+b
    loss = loss_fn(y_true,y_predict)

    if i % 500 == 0:
        print(i,loss)
    optimize(0.01)

predict = x*w + b


plt.scatter(x.data.numpy(), y_true.data.numpy(),c = "r")
plt.plot(x.data.numpy(), predict.data.numpy())
plt.show()

print("w",w)
print("b",b)


































