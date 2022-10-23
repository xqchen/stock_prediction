import pandas as pd
import os
import numpy as np
import torch.utils.data as Data


# 处理数据
def load_data(dir):
    # 获取目录下所有文件名字
    dir_4 = dir
    list_4 = os.listdir(dir_4)
    # 设置开始时间和结束时间
    start_time = 93000
    end_time = 145700
    for parquet_name in list_4:
        # 删除不在符合区间内的数据
        data_4 = pd.read_parquet(dir_4 + parquet_name)
        time_list = data_4['tradeTime'].tolist()
        time_index_list = []
        for time in time_list:
            if time < start_time:
                time_index_list.append(time_list.index(time))
            else:
                break
        for time in range(len(time_list) - 1, 0, -1):
            if time_list[time] > end_time:
                time_index_list.append(time)
            else:
                break
        # 获取特征列和中间交易量列，并通过中间交易量算出真实利润率
        data = np.array(data_4)[time_index_list[0] + 1:time_index_list[-1], :]
        mid_price_list = list(data[:, -1])
        data = data[:, 1:47]
        # profit_list = [0 for _ in range(20)]
        profit_list = []
        for price_index in range(20, len(mid_price_list)):
            profit = mid_price_list[price_index] / mid_price_list[price_index - 20] + 1
            profit_list.append(profit)
        # 将算出的真实利润率与特征拼接在一起作为新的数据
        profit_list += [0 for _ in range(20)]
        profit_np = np.array(profit_list).reshape(-1, 1)
        data = pd.DataFrame(np.concatenate([data, profit_np], axis=1))
        # 保存数据
        if data_4[-2] == 'l':
            new_name = '4/'
        elif data_4[-2] == 'y':
            new_name = '5/'
        elif data_4[-2] == 'e':
            new_name = '6/'
        if not os.path.exists(dir_4[-5:]):
            os.mkdir(dir_4[-5:])
        data.to_csv(new_name + parquet_name[: -8] + '.csv')
        print('over ' + parquet_name[: -8])

def main():
    dir_list = ['data_june/', 'data_april/', 'data_may/']
    for dir in dir_list:
        load_data(dir)

class MDataset(Data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


# 划分20步长的数据，并将4月和5月数据作为训练集，6月数据90%作为验证集，10%数据作为测试集
def data_create(TEST=False):
    # 获取新保存的csv文件
    train_seq, test_seq = [], []
    if TEST:
        name_list = ['6/']
    else:
        name_list = ['4/', '5/', '6/']

    for dir in name_list:
        if dir == '6/':
            TRAIN = False
        else:
            TRAIN = True
        dir_list = os.listdir(dir)
        # 将4、5、6月的数据分开处理
        for csv_dir in dir_list:
            data = np.array(pd.read_csv(dir + csv_dir))[:, 2:]
            nums, feature_nums = data.shape
            # 划分20的步长，获取特征与标签
            for index in range(0, nums-20):
                train_x_list, train_y_list = [], []
                test_x_list, test_y_list = [], []
                if TRAIN:
                    train_x_list.append(data[index:index+20, :])
                    train_y_list.append(data[index+20, -1])
                else:
                    test_x_list.append(data[index:index + 20, :])
                    test_y_list.append(data[index + 20, -1])
                if train_x_list != []:
                    train_seq.append((train_x_list, train_y_list))
                if test_y_list != []:
                    test_seq.append((test_x_list, test_y_list))
    val_seq = test_seq[0: int(len(test_seq) * 0.9)]
    test_seq = test_seq[int(len(test_seq) * 0.9):]

    # 生成Dataloader
    train_data = MDataset(train_seq)
    test_data = MDataset(test_seq)
    val_data = MDataset(val_seq)

    train_dataloader = Data.DataLoader(dataset=train_data, batch_size=128, shuffle=False)
    val_dataloader = Data.DataLoader(dataset=val_data, batch_size=128, shuffle=False)
    test_dataloader = Data.DataLoader(dataset=test_data, batch_size=128, shuffle=False)

    if len(train_dataloader) == 0:
        return test_dataloader
    return train_dataloader, val_dataloader


if __name__ == '__main__':
    train_dataloader, test_dataloader, val_dataloader = data_create()
    print(len(test_dataloader), len(val_dataloader))

    for step, (x, y) in enumerate(test_dataloader):
        print(x[0].shape)
        print(y[0].shape)

