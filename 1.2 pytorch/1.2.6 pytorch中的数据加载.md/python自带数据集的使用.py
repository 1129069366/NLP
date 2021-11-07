import torchvision

dataset = torchvision.datasets.MNIST(root="./data",train=True,download=True,transform=None)

print(dataset)
print(len(dataset))

img = dataset[0][0]
print(img)




