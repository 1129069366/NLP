import torch

batch_size =10
seq_len = 20
embedding_dim = 30
word_vocab = 100
hidden_size = 18
num_layer = 2

#准备输入数据
input = torch.randint(low=0,high=100,size=(batch_size,seq_len))
#准备embedding
embedding  = torch.nn.Embedding(word_vocab,embedding_dim)
lstm = torch.nn.LSTM(embedding_dim,hidden_size,num_layer)

#进行mebed操作
embed = embedding(input) #[10,20,30]

#转化数据为batch_first=False
embed = embed.permute(1,0,2) #[20,10,30]

#初始化状态， 如果不初始化，torch默认初始值为全0
h_0 = torch.rand(num_layer,batch_size,hidden_size)
c_0 = torch.rand(num_layer,batch_size,hidden_size)
output,(h_n,c_n) = lstm(embed,(h_0,c_0))
#output [20,10,1*18]
#h_1 [2,10,18]
#c_1 [2,10,18]

print(output.size())
print(h_n.size())
print(c_n.size())
last_output = output[-1,:,:]
last_state = h_n[-1::]
print(last_output==last_state)