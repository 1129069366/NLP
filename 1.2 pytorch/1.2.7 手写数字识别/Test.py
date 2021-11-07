from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
from torch import optim
import torch

train_batch_size = 64      # 训练批量的大小
test_batch_size = 1000    # 测试批量的大小
img_size = 28

# 返回数据集
def get_dataloader(train):
    dataset = torchvision.datasets.MNIST("./data",train=train,download=True,transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,),(0.3081,)
        )
    ]))
    batch_size = train_batch_size if train else test_batch_size
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# 构建模型的代码如下:
class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = nn.Linear(28*28*1, 28)
        self.fc2 = nn.Linear(28,10)

    def forward(self, x):
        x = x.view(-1,28*28*1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

# 创建模型实例
mnist_net = MnistNet()
# 创建优化器
optimizer = optim.Adam(mnist_net.parameters(),lr=0.001)
train_loss_list = []
train_count_list = []
def train(epoch):       # epoch:第几次训练
    mode = True
    mnist_net.train(mode=mode)
    train_loader = get_dataloader(train=mode)
    for index,(data,target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = mnist_net(data)
        loss = F.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        if index % 10 == 0:
            # {}/{}表示了批量的大小
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, index * len(data), len(train_loader.dataset),
                       100. * index / len(train_loader), loss.item()))

            train_loss_list.append(loss.item())
            train_count_list.append(index * train_batch_size + (epoch - 1) * len(train_loader))
            torch.save(mnist_net.state_dict(), "model/mnist_net.pkl")
            torch.save(optimizer.state_dict(), 'results/mnist_optimizer.pkl')

def test():
    test_loss = 0
    correct = 0
    mnist_net.eval()
    test_loader = get_dataloader(train=False)
    with torch.no_grad():
        for data,target in test_loader:
            output = mnist_net(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]  # 获取最大值的位置,[batch_size,1] 该位置值就												 是预测的数字
            correct += pred.eq(target.view_as(
                pred)).sum()  # target.view_as(pred)把																	target变成形状和pred相同的tensor
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':

    test()
    for i in range(5):  # 模型训练5轮
        train(i)
        test()
    # data_loader = get_dataloader(train=True)
    # print(len(data_loader.dataset))
    # print(len(data_loader))
    # for index,(data,target) in enumerate(data_loader):
    #     print(data)
    #     break


















