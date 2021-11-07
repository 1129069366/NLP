from torch.utils.data import Dataset,DataLoader
import pandas as pd


data_path = r"C:\Users\YBH\Desktop\Python\data\game\刑期预测赛道一期数据_训练集"

lines = open(data_path,encoding="utf-8",mode="r").readline()

print(lines)
