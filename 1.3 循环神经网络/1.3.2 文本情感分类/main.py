from Word2Sequence import Word2Sequence
import pickle
import os
from dataset import tokenize
from tqdm import tqdm

if __name__ == '__main__':

    ws = Word2Sequence()
    path = r"C:\Users\YBH\Desktop\Python\data\aclImdb\train"
    temp_data_path = [os.path.join(path,i) for i in ["pos","neg"]]
    for data_path in temp_data_path:
        filenames = os.listdir(data_path)
        filenames = [filename for filename in filenames if filename.endswith(".txt")]
        for i in tqdm(filenames):
            file_path = os.path.join(data_path,i)
            sentence = tokenize(open(file_path,encoding="utf-8").read())
            ws.fit(sentence)

    ws.build_vocab(min_count=10,max_features=10000)
    pickle.dump(ws,open("./model/ws.pkl","wb"))
    print(len(ws))