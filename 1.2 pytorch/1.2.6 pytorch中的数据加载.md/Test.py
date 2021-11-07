from torch.utils.data import Dataset,DataLoader
import pandas as pd

data_path = r"C:\Users\YBH\Desktop\Python\data\SMSSpamCollection"

class MyDataSet(Dataset):

    def __init__(self):
        lines = open(data_path,encoding="utf-8").readlines()
        lines = [[i[:4].strip(),i[4:].strip()] for i in lines]
        self.df = pd.DataFrame(lines,columns=["label","sms"])

    def __getitem__(self, i):

        single_item =  self.df.iloc[i,:];       # 返回i行的所有列数据
        return single_item.values[0],single_item.values[1]

    def __len__(self):
        return self.df.shape[0]    # 返回DataFrame.shape[0]  df的行数也就是记录条数了


if __name__ == '__main__':
    # d = MyDataSet()
    # for i in range(len(d)):
    #     print(i, d[i])
    d = MyDataSet()
    data_loader = DataLoader(d,batch_size=10,shuffle=True,num_workers=2)
    for index,(label,content) in enumerate(data_loader):
        print(index,label,content)







