import numpy as np

class Word2Sequence():

    UNK_TAG = "UNK"
    PAD_TAG = "PAD"

    UNK = 0
    PAD = 1

    def __init__(self):
        self.dict = {
            self.UNK_TAG:self.UNK,
            self.PAD_TAG:self.PAD
        }

        self.count = {}   # 统计词频

    def fit(self,sentence):

        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self,min_count = 5, max_count = None, max_features=None):
        if min_count is not None:
            self.count = {k : v for k ,v in self.count.items() if v > min_count}

        if max_count is not None:
            self.count = {k:v for k,v in self.count.items() if v < max_count}

        if max_features is not None:
            self.count = sorted(self.count.items(),key=lambda x:x[-1],reverse=True)  # reverse=True 降序排列
            if len(self.count)>max_features:
                self.count = self.count[:max_features]

            for w in dict(self.count).keys():
                self.dict[w] = len(self.dict)

        for w in self.count:
            self.dict[w] = len(self.dict)


        self.inverse_dict = dict(zip(self.dict.values(),self.dict.keys()))

    def __len__(self):
        return len(self.dict)

    # 转化句子为序列
    def transform(self,sentences,max_len=None):
        """
        :param sentences:  [word1,word2]
        :return:
        """
        if max_len is not None:
            if len(sentences) > max_len:
                sentences = sentences[:max_len]
            else:
                sentences = sentences + [self.PAD_TAG] * (max_len - len(sentences))

        return [self.dict.get(word,self.UNK) for word in sentences]

    # 把数字数列转化为序列
    def inverse_transform(self,indices):

        return [self.inverse_dict.get(i) for i in indices]


if __name__ == '__main__':
    # ws = Word2Sequence()
    # ws.fit(["我","是","谁"])
    # ws.fit(["我","是","我"])
    # ws.build_vocab(min_count=0)
    #
    # print(ws.dict)
    # print(ws.inverse_dict)
    # ret = ws.transform(["我", "在", "哪"],max_len=10)
    # print(ret)
    # print(ws.inverse_transform(ret))
    pass
















