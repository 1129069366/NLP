import torch
from torch.utils.data import DataLoader,Dataset
import os
import re
import lib
from lib import ws

data_base_path = r"C:\Users\YBH\Desktop\Python\data\aclImdb"   # 基础的数据路径

# 1.定义tokenize的方法  分词方法


def tokenize(text):
    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                '\?', '@'
        , '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”', '“', ]

    text = re.sub('<.*?>'," ",text,flags=re.S)
    test = re.sub('|'.join(fileters)," ",text,flags=re.S)
    return [i.strip() for i in text.split()]

# 2 .准备dataset


class ImdbDataset(Dataset):

    def __init__(self,mode):
        super(ImdbDataset, self).__init__()
        if mode == "train":
            text_path = [os.path.join(data_base_path,i) for i in ["train/pos","train/neg"]]
        else:
            text_path = [os.path.join(data_base_path,i) for i in ["test/pos","test/neg"]]

        # 所有文件的绝对路径
        self.total_file_path_list = []
        for i in text_path:
            self.total_file_path_list.extend([os.path.join(i,j) for j in os.listdir(i)])

    def __getitem__(self, idx):
        file_path = self.total_file_path_list[idx]    # C:\Users\YBH\Desktop\Python\data\aclImdb\test\neg\0_2.txt
        label = int(file_path.split("_")[-1].split(".")[0])-1
        label = 0 if label < 5 else 1
        text = tokenize(open(file_path,encoding="utf-8").read())
        return label,text

    def __len__(self):
        return len(self.total_file_path_list)

def getDataLoader(mode,batch_size):
    imdb_dataset = ImdbDataset(mode=mode)
    dataloader = DataLoader(imdb_dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)
    return dataloader


def collate_fn(batch):
    label,content = list(zip(*batch))   # batch内容  ([label,tokens],[label,tokens])
    label = torch.LongTensor(label)
    content = [ws.transform(i,max_len = lib.max_len) for i in content]  # 词语序列化为数字
    content = torch.LongTensor(content)
    return label,content


if __name__ == '__main__':
    imdb_dataset = ImdbDataset(mode="train")
    dataloader = DataLoader(imdb_dataset,batch_size=2,shuffle=True,collate_fn=collate_fn)
    for idx,(target,input) in enumerate(dataloader):
        print("idx:",idx)
        print("label:",target)
        print("text:",input)
        break


























