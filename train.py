from lmodel import CNN_LSTM
from dataset import data_create
import torch
import torch.backends.cudnn as cudnn
from torch import optim
import time
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

'''

1. 使用是1层cnn+2层lstm模型，选用该模型的理由：
    （1）lstm是长短期记忆网络，相对于rnn而言，lstm通过细胞状态和隐层状态来存贮’长期记忆‘和’短期记忆‘。
        数据集明显和时间方向有关，因此选择lstm作为网络主体部分。
    （2）利润率只和过去的数据有关，而和未来的数据无关，因此选用单向lstm，而不是bilstm。
    （3）cnn提取特征能力十分强大，因此加入了1层cnn，增加网络提取特征的能力。
    ··
2. 通过dataset.py文件中的函数，将原始数据集划分为20步长，并将4月和5月数据作为训练集，6月数据90%作为
验证集，10%数据作为测试集。在训练和验证阶段，通过输出网络的loss，mse，mae来判断拟合的效果与网络的能力
，并输出了训练和验证阶段的loss曲线图。没有在训练和验证阶段输出r2和corr的原因是：内存占用过大。在测试阶
段输出了预测数据与真实数据的mae，mse，r2，corr与拟合结果图。通过损失图可以看出，网络确实在优化，并且未
发生过拟合情况。通过测试阶段的评价指标和拟合图可以看出，网络拟合能力比较优秀。

3. 该数据集是时间序列，近年来transformer以及transformer的变体在机器翻译等领域取得较好结果，并且在一
些图像处理领域也发挥了重要的作用。transformer主要是靠多头注意力将特征映射到不同的空间中发挥其作用，未来
可以将transformer的这种注意力机制与lstm结合起来来改进网络性能。

'''



# R方定义
def R2_0(y_test, y_pred):
    SStot = np.sum((y_test - np.mean(y_test)) ** 2)
    SSres = np.sum((y_test - y_pred) ** 2)
    r2 = 1 - SSres / SStot
    return r2

# 训练+验证
def train():
    # 设置cuda和种子参数
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.manual_seed(123)
    cudnn.benchmark = True

    # 获取训练集和验证集
    train_dataloader, val_dataloader = data_create()
    print('=' * 100)
    print('Data Over')
    print('=' * 100)

    # 定义网络，优化器，损失函数
    net = CNN_LSTM(in_channels=47, out_channels=32, hidden_size=32, num_layer=2, output_size=1)
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, mode='min', factor=0.9)
    MSE_loss = torch.nn.MSELoss()

    net = net.to(device)
    MSE_loss = MSE_loss.to(device)

    best_loss = float('inf')
    train_loss_list, val_loss_list = [], []
    for epoch in range(30):
        net.train()
        train_num = 0
        train_loss_all = 0.0
        train_strat = time.time()
        train_mse = 0.0
        train_mae = 0.0

        # 开始训练
        for step, (x, y) in enumerate(train_dataloader, 1):
            x = x[0].float().to(device)
            y = y[0].float().to(device)

            pre_y = net(x).reshape(-1)
            loss = MSE_loss(pre_y, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_all += loss.item()

            pre_y = pre_y.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            mse = np.mean((pre_y - y) ** 2)
            mae = np.mean(np.abs(pre_y - y))
            train_mae += mae
            train_mse += mse
            train_num += 1

        train_loss_list.append(train_loss_all / train_num)
        scheduler.step(train_loss_list[-1])
        train_spent = time.time() - train_strat

        print('_' * 150)
        print(
            'epoch:{}     train_loss:{:.6f}      train_spent_time:{:.4f}      mae:{:.6f}      mse:{:.6f}'.format(
                epoch + 1, train_loss_list[-1], train_spent, train_mae / train_num, train_mse / train_num))

        # 开始验证
        net.eval()
        val_num = 0
        val_loss_all = 0.0
        val_strat = time.time()
        val_mse = 0.0
        val_mae = 0.0
        for step, (x, y) in enumerate(val_dataloader, 1):
            x = x[0].float().to(device)
            y = y[0].float().to(device)

            pre_y = net(x).reshape(-1)
            loss = MSE_loss(pre_y, y)

            val_loss_all += loss.item()

            pre_y = pre_y.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            mse = np.mean((pre_y - y) ** 2)
            mae = np.mean(np.abs(pre_y - y))
            val_mae += mae
            val_mse += mse
            val_num += 1

        val_loss_list.append(val_loss_all / val_num)
        val_spent = time.time() - val_strat

        print('_' * 150)
        print(
            'epoch:{}     val_loss:{:.6f}      val_spent_time:{:.4f}      mae:{:.6f}      mse:{:.6f}'.format(
                epoch + 1, val_loss_list[-1], val_spent, val_mae / val_num, val_mse / val_num))

        # 保存模型
        if val_loss_list[-1] < best_loss:
            best_loss = val_loss_list[-1]
            if not os.path.exists('checkpoint/'):
                os.mkdir('checkpoint/')
            torch.save(net.state_dict(), 'checkpoint/result.pth')

    # 保存损失值
    if not os.path.exists('result/'):
        os.mkdir('result/')
    with open('result/train_loss.txt', 'w') as f:
        f.write(str(train_loss_list))
    with open('result/val_loss.txt', 'w') as f:
        f.write(str(val_loss_list))

# 可视化损失图
def visdom(train_loss_list, val_loss_list):
    assert len(train_loss_list) == len(val_loss_list)
    plt.figure(figsize=(10, 7))
    plt.plot(range(len(train_loss_list)), train_loss_list, label='Train Loss')
    plt.plot(range(len(val_loss_list)), val_loss_list, label='Val Loss')
    plt.title('Train and Val Loss')
    plt.legend()
    plt.show()

# 测试
def TEST():
    # 加载测试数据
    test_dataloader = data_create(TEST=True)
    print('Successful Load Data')
    # 加载保存的网络
    net = CNN_LSTM(in_channels=47, out_channels=32, hidden_size=32, num_layer=2, output_size=1)
    ckpt_dict = torch.load('checkpoint/result.pth')
    net.load_state_dict(ckpt_dict)
    print('Successful Load Model')

    net.eval()
    y_list = []
    pre_y_list = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    MSE_loss = torch.nn.MSELoss().to(device)
    test_mae, test_mse, test_r2 = 0.0, 0.0, 0.0
    test_num = 0
    test_loss = 0.0

    if not os.path.exists('result/'):
        os.makedirs('result/')

    # 开始测试
    for step, (x, y) in enumerate(test_dataloader, 1):
        x = x[0].float().to(device)
        y = y[0].float().to(device)

        pre_y = net(x).reshape(-1)
        loss = MSE_loss(pre_y, y)

        pre_y = pre_y.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        y_list += list(y.reshape(-1))
        pre_y_list += list(pre_y.reshape(-1))

        mse = np.mean((pre_y - y) ** 2)
        mae = np.mean(np.abs(pre_y - y))

        test_mae += mae
        test_mse += mse
        test_loss += loss.item()
        test_num += 1

    y_list = np.array(y_list).reshape(-1)
    pre_y_list = np.array(pre_y_list).reshape(-1)
    corr_pear, _ = pearsonr(y_list, pre_y_list)
    r2 = R2_0(y_list, pre_y_list)

    print('Test Loss:{:.6f}'.format(test_loss/test_num))
    print('Test MAE:{:.6f}'.format(test_mae/test_num))
    print('Test MSE:{:.6f}'.format(test_mse/test_num))
    print('Test R2:{:.6f}'.format(r2))
    print('Test Corr:{:.6f}'.format(corr_pear))

    # 保存拟合结果
    with open('result/test_y.txt', 'w') as f:
        f.write(str(list(y_list)))
    with open('result/pre_y.txt', 'w') as f:
        f.write(str(list(pre_y_list)))

# 可视化拟合结果
def visdom_test(pre_y, y):
    plt.figure(figsize=(10, 7))
    plt.plot(range(len(pre_y)), pre_y, label='Predict label')
    plt.plot(range(len(y)), y, label='True label')
    plt.title('True and Predict label')
    plt.xticks([])
    plt.legend()
    plt.show()

# 加载数据
def load_txt(dir_1, dir_2):
    data_1_list, data_2_list = [], []
    with open(dir_1, 'r') as f:
        data_1 = f.readline()
    data_1 = data_1.split(',')
    for data in data_1:
        data_1_list.append(float(data[1:-1]))
    with open(dir_2, 'r') as f:
        data_2 = f.readline()
    data_2 = data_2.split(',')
    for data in data_2:
        data_2_list.append(float(data[1:-1]))

    return data_1_list, data_2_list

def main():
    train()
    train_loss, val_loss = load_txt('result/train_loss.txt', 'result/val_loss.txt')
    visdom(train_loss, val_loss)

    TEST()
    true_y, predict_y = load_txt('result/test_y.txt','result/pre_y.txt')
    visdom_test(true_y, predict_y)


if __name__ == '__main__':
    TEST()




