import json
import copy
import sys
from tqdm.auto import tqdm
import time
import numpy as np
import os.path as op
import pandas as pd
import argparse
import torch
from torch import nn,optim
import matplotlib.pyplot as plt
import copy
from data_analysis import data_analysis
import matplotlib.pyplot as plt
from net import torch_net,Bayes
# np.set_printoptions(threshold=np.inf)

class SD_Dataset(): # Staff Dimission data
    def __init__(self,data_path,sheet_name,explanation_path,non_num_col,missing_value):
        self.explanation = self.load_data(explanation_path,n=11)
        self.data = self.load_data(data_path,sheet_name=sheet_name)
        if non_num_col:
            self.non_num_col = non_num_col
            self.name2id_dict,self.id2name_dict = self.handle_non_number(self.data,non_num_col,missing_value)
    """
    function : load xls&txt file
    """
    def load_data(self,file_path,sheet_name='Sheet1',n=None):
        assert op.isfile(file_path)
        datatype = file_path.split('.')[-1]
        allow_data_type = ['xls','xlsx','txt']
        assert datatype in allow_data_type # 仅处理两种格式文件数据
        if datatype == 'txt':
            # convert key to name, for example{ID：员工编号}
            data = dict()
            with open(file_path,'r',encoding='gbk') as f:
                for line in f.readlines()[1:]:
                    temp_data = line.split('.')[1].split('：')
                    data[temp_data[0].strip()] = temp_data[1].strip()
                    if n != None:
                        n -= 1
                        if n == 0:
                            break
            return data
        elif datatype == 'xls' or datatype == 'xlsx':
            try:
                data = pd.read_excel(file_path, sheet_name=sheet_name)
            except Exception as e:
                print(e)
                print(file_path," 文件加载失败！")
                sys.exit(0)
            return data
    def handle_non_number(self,data,non_num_col,missing_value):
        convert_dict = dict()
        id2name_dict = dict()
        for col_name in non_num_col:
            col_dict = dict()
            id2name = []
            for i in range(data.shape[0]):
                str_data = str(data[col_name].iat[i])
                if str_data not in id2name and str_data not in missing_value:
                    col_dict[str_data] = len(id2name)
                    id2name.append(str_data)
            convert_dict[col_name] = col_dict
            id2name_dict[col_name] = id2name
        for col_name in non_num_col:
            for i in range(data.shape[0]):
                str_data = str(data[col_name].iat[i])
                if str_data not in missing_value:
                    data[col_name].iat[i]=convert_dict[col_name][str_data]
        return convert_dict,id2name_dict
    def get_demission_data(self):
        return self.data
    def get_explanation_data(self):
        return self.explanation
    def convert_id2name(self,col_name,index):
        return self.id2name_dict[col_name][index]
    def convert_name2id(self,col_name,index):
        return self.name2id_dict[col_name][index]

def process_mis_val(data,methord,mis_val,data_columns,mis_val_num,mis_row=None,mis_col_name=None):
    """
    function : process missing value
    :param data: total data
    :param methord: Methods to handle missing values
    :param mis_val: A list of missing values, which stores what is missing
    :param data_columns: process only these attributes
    :param mis_row: the row with missing values detected
    :param mis_col_name: the col with missing values detected
    """
    if methord == 'attr_mean':
        mis_col = []
        total_val = 0.0
        total_len = 0
        for col_name in data_columns:
            if str(data[col_name].iat[mis_row]) not in mis_val:
                total_val += float(data[col_name].iat[mis_row])
                total_len += 1
            else:
                mis_col.append(col_name)
        fill_val = total_val/total_len
        for col_name in mis_col:
            data[col_name].iat[mis_row] = fill_val
    if methord == 'col_mean':
        mis_r = []
        total_val = 0.0
        total_len = 0
        for row_index in range(data.shape[0]):
            if str(data[mis_col_name].iat[row_index]) not in mis_val:
                total_val += float(data[mis_col_name].iat[row_index])
                total_len += 1
            else:
                mis_r.append(row_index)
        fill_val = total_val / total_len
        mis_val_num[mis_col_name] = 0
        for row_index in mis_r:
            data[mis_col_name].iat[row_index] = fill_val
            mis_val_num[mis_col_name] += 1
        return mis_val_num
    if methord == 'min_dis':
        pass

def preprocess_data(data,mis_val,important_columns=[],ignore_columns=[],target=-1):
    """
    function :
    :param data:
    :param mis_val:
    :param important_columns: Missing values are not allowed in these columns
    :param ignore_columns:
    :param target:
    :return:
    """
    mis_val_num = dict()
    data_columns = list(data.columns) # Don't process columns
    target_col = data_columns[target]
    data_length = data.shape[0]
    important_columns.append(target_col)
    data_columns.remove(target_col)
    # Don't process the specified column if specified
    if ignore_columns:
        for col_name in ignore_columns:
            data_columns.remove(col_name)

    del_index = 0
    for col_name in important_columns:
        for index in range(data_length):
            str_data = str(data[col_name].iat[index])  # the origin type : np.float --> str, for comparison
            if str_data in mis_val:
                data.drop(index)
                del_index -= 1
            del_index += 1
    # handle missing value
    for col_name in data_columns:
        for index in range(data_length):
            str_data = str(data[col_name].iat[index]) # the origin type : np.float --> str, for comparison
            if str_data in mis_val:
                process_mis_val(data,'col_mean',mis_val,data_columns,mis_val_num,mis_col_name=col_name)
                break
    return mis_val_num

def sort_col(col,target_col):
    """
    function: sort col in the format of target_col
    """
    length = len(col)
    if length == 1:
        return
    num_dic = dict()
    num_list = []
    for i in range(len(target_col)):
        num_dic[target_col[i]] = i
    for i in col:
        num_list.append(num_dic[i])
    num_list.sort()
    for i in range(length):
        col[i] = target_col[num_list[i]]

def data_normalization(data):
    maximums, minimums, avgs = \
        data.max(axis=0), \
        data.min(axis=0), \
        data.sum(axis=0) / data.shape[0]
    for i in range(data.shape[1]):
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])

def split_data(demission_data,train_n,test_n,ignore_columns,target=-1):
    data_columns = list(demission_data.columns)  # Don't process columns
    data = np.array(demission_data)
    # Remove ignored attributes
    sort_col(ignore_columns,data_columns)
    index_f = 0
    for col_name in ignore_columns:
        index = data_columns.index(col_name)
        data = np.delete(data,index-index_f,1)
        index_f += 1

    train_i,test_i = int(data.shape[0]*train_n),int(data.shape[0]*test_n)
    train_data,train_label = data[:train_i,:target],data[:train_i,target]
    test_data,test_label = data[train_i:,:target], data[train_i:,target]

    # print(get_data_scope(train_data))
    # print(get_data_scope(test_data))
    # normalization
    data_normalization(train_data),data_normalization(test_data)
    train_data,test_data = np.c_[train_data,train_label],np.c_[test_data,test_label]
    return train_data,test_data

def FNN(train_data,test_data):
    # net = Network(train_data.shape[1]-1)
    # losses = net.train(train_data, num_epochs=50, batch_size=10, eta=0.1)
    model_path = "fnn_model.pt"
    model = torch_net(9,16,2,train_data,test_data)
    epochs = 30
    model.train(epochs,learning_rate=0.001)
    model.test()

    # torch.save(model, model_path)
    # model = torch.load('fnn_model.pt')
    # predict = model(test_data)

    # 画出损失函数的变化趋势
    # plot_x = np.arange(len(losses))
    # plot_y = np.array(losses)
    # plt.plot(plot_x, plot_y)
    # plt.show()

def bayes(train_data,test_data):
    model = Bayes(train_data)
    predict_label,target_label = model.predict(test_data)
    model.evaluate(predict_label,target_label)

def similar_ana(x,y):
    cor_res = np.corrcoef(x, y)
def get_data_scope(data):
    data = np.array(data)
    min_attr = np.min(data,axis=0)
    max_attr = np.max(data,axis=0)
    return np.c_[min_attr,max_attr]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default='bayes', type=str, required=False,
                        help="The method you want to use, support bayes and fnn")
    args = parser.parse_args()
    # global logger

    data_path = './员工离职数据.xls'
    sheet_name = '员工离职数据'
    explanation_path = './数据说明.txt'
    missing_value = ['nan','NAN'] # in pandas : 'NULL' --> 'nan' or 'NAN'
    non_num_col = ['sector','salary']
    ignore_columns = ['ID']
    train_n,test_n = 0.8,0.2 # 训练集、测试集 8:2

    # data preprocess
    dataset = SD_Dataset(data_path,sheet_name,explanation_path,non_num_col,missing_value)
    explanation_data = dataset.get_explanation_data()
    demission_data = dataset.get_demission_data()
    mis_num = preprocess_data(demission_data,missing_value,ignore_columns=ignore_columns)
    train_data,test_data = split_data(demission_data,train_n,test_n,ignore_columns)
    # cor = np.array(np.corrcoef(train_data))
    average_attr = np.average(train_data,axis=0)[:-1]
    # print(mis_num)
    if args.method == 'fnn':
        FNN(train_data,test_data)
    elif args.method == 'bayes':
        bayes(train_data,test_data)
    # data analyze
    # data_analysis(demission_data,missing_value)

if __name__ == '__main__':
    main()
