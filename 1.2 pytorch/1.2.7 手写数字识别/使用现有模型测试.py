from torch import  nn
from torch.utils.data import dataset,dataloader
import torchvision
from torchvision import transforms
import torch
from torch import optim
import torch.nn.functional as F

train_batch_size = 64      # 训练批量的大小
test_batch_size = 1000    # 测试批量的大小
img_size = 28

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

mnist_net = MnistNet()
optimizer = optim.Adam(mnist_net.parameters())
mnist_net.load_state_dict(torch.load("model/mnist_net.pkl"))
optimizer.load_state_dict(torch.load("results/mnist_optimizer.pkl"))

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