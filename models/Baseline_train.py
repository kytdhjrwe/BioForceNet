import glob
import math
import time
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import Baseline_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

INPUT_DIMS = 8
TIME_STEPS = 80
startNum = 0
pre_len = 1
lstm_units = 64
stride = 10
epoch_nums = 3
learning_rate = 0.1

target = 'vgrf'
# self.target = 'tbf'



# Estimation of vGRF from Insole Data
def get_vGRF(path, TIME_STEPS, stride, pre_len, startNum):
    """
    Read data and generate training set and target values based on time steps and stride length
    """
    print('Estimating vertical ground reaction force')

    X, y = [], []
    df = pd.read_excel(path)

    All_lable_data = df.iloc[:, 15].tolist()
    length = len(All_lable_data)

    for i in range(startNum, length - TIME_STEPS + 1, stride):
        window_data = df.iloc[i:i + TIME_STEPS, :]
        if not window_data.isnull().values.any() and len(window_data) == TIME_STEPS:
            X.append(window_data.iloc[:, :8])
            y.append(window_data.iloc[-pre_len:, 15])

    len_train = len(X)
    print(f'{len_train} pieces of data in total')
    return np.array(X), np.array(y), len_train

# Estimation of TBF from Insole Data
def get_TBF(path, TIME_STEPS, stride, pre_len, startNum):
    """
    Read data and generate training set and target values based on time steps and stride length
    """
    print('Estimating tibia bone force')

    X, y = [], []
    df = pd.read_excel(path)

    All_lable_data = df.iloc[:, 16].tolist()
    length = len(All_lable_data)

    for i in range(startNum, length - TIME_STEPS + 1, stride):
        window_data = df.iloc[i:i + TIME_STEPS, :]
        if not window_data.isnull().values.any() and len(window_data) == TIME_STEPS:
            X.append(window_data.iloc[:,:8])
            y.append(window_data.iloc[-pre_len:, 16])

    len_train = len(X)
    print(f'{len_train} pieces of data in total')
    return np.array(X), np.array(y), len_train

def load_data(path, target, TIME_STEPS, stride, pre_len, startNum):
    """
    Load data and generate training set and label set.
    """
    train_X = []
    train_Y = []
    file_paths = glob.glob(f"{path}/*.xlsx")

    for path in file_paths:
        print(f'Loading data from: {path}')
        if target == 'vgrf':
            X, y, len_train = get_vGRF(path, TIME_STEPS, stride, pre_len, startNum)
        elif target == 'tbf':
            X, y, len_train = get_TBF(path, TIME_STEPS, stride, pre_len, startNum)
        else:
            print('Data reading exception')
            X, y, len_train = None, None, None

        train_X.append(X)
        train_Y.append(y)
        break

    # Combine all data
    train_X = np.vstack(train_X)
    train_Y = np.vstack(train_Y)

    return train_X, train_Y

def prepare_data(train_X, train_Y, device, split_ratio=0.8, batch_size=64):
    """
    Divide the data into training and testing sets, and prepare a data loader.
    """
    len_train = len(train_X)
    indices = torch.randperm(len_train)
    split_idx = int(len_train * split_ratio)

    X_shuffled = torch.tensor(train_X[indices], dtype=torch.float32).to(device)
    y_shuffled = torch.tensor(train_Y[indices], dtype=torch.float32).to(device)

    X_train, X_test = X_shuffled[:split_idx], X_shuffled[split_idx:]
    y_train, y_test = y_shuffled[:split_idx], y_shuffled[split_idx:]

    print(f'Training data size: {len(X_train)}')
    print(f'Testing data size: {len(X_test)}')

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def evaluate(model, test_loader, criterion):

    model = model.to(device)
    model.eval()
    test_loss = 0.0

    y_pred, y_true = [], []
    with torch.no_grad():  # 禁用梯度计算
        for X_batch, y_batch in test_loader:
            # print('X_batch',X_batch)
            # print('y_batch',y_batch)
            # print('X_batch',X_batch)
            # print('y_batch',y_batch)
            outputs = model(X_batch)
            y_batch = y_batch.unsqueeze(1)
            loss = criterion(outputs, y_batch)  # 将y_batch扩展一维以匹配输出
            test_loss += loss.item()
            y_pred.extend(outputs.cpu().numpy())
            y_true.extend(y_batch.cpu().numpy())

    avg_loss = test_loss / len(test_loader)

    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()

    mse = mean_squared_error(y_true, y_pred)


    correlation, p_value = pearsonr(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    return avg_loss,mse,correlation, p_value,rmse,mae,mape,r2

def plot_losses(train_loss_list, test_loss_list, model_name, target, TIME_STEPS, train_time):
    """
    Draw training and testing loss graphs and save them
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(test_loss_list, label="Test Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Train and Test Loss over Epochs ({model_name})')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(f"./Baseline_results/{model_name}_{target}_windows_{TIME_STEPS}_{train_time}.png")
    plt.close()

if __name__ == '__main__':

    path = 'Traindata'
    # 加载数据
    train_X, train_Y = load_data(path, target, TIME_STEPS, stride, pre_len, startNum)
    # 数据划分与准备
    train_loader, test_loader = prepare_data(train_X, train_Y, device)  # 使用 CPU

    # model_name_list = ['CNN3','CNN_LSTM','CNN_BiLSTM','BiLSTM_attention']
    model_name_list = ['BiLSTM_attention']
    for model_name in model_name_list:
        model = None
        print(f'Load model: {model_name}')
        # 创建模型
        if model_name == 'BiLSTM_attention':
            model = Baseline_model.BiLSTM_attention(INPUT_DIMS, TIME_STEPS, lstm_units).to(device)
        elif model_name == 'CNN_BiLSTM':
            model = Baseline_model.CNN_BiLSTM(INPUT_DIMS,lstm_units).to(device)
        elif model_name == 'CNN_LSTM':
            model = Baseline_model.CNN_LSTM(INPUT_DIMS,lstm_units).to(device)
        elif model_name == 'CNN3':
            model = Baseline_model.CNN(INPUT_DIMS).to(device)
        else:
            print("Model loading failed")

        # Define loss function and optimizer
        criterion_MSE = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Set to infinity
        min_epoch_loss = math.inf
        train_time = time.strftime('%Y_%m_%d-%H_%M_%S', time.localtime())

        log_write = open("./Baseline_results/" + model_name +"_"+ target + "__windows_"+ str(TIME_STEPS) + "__" + train_time + "__log" + ".txt", "w")

        train_loss_list = []
        test_loss_list = []

        for epoch in range(epoch_nums):
            if epoch > 100:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.01

            model.train()
            epoch_loss = 0.0

            # Training
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion_MSE(outputs, y_batch.unsqueeze(1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            train_loss = epoch_loss / len(train_loader)

            # Testing: Model performance evaluation after one round of training
            test_loss,mse,correlation, p_value,rmse,mae,mape,r2 = evaluate(model, test_loader, criterion_MSE)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            print(f"Epoch [{epoch + 1}/{epoch_nums}]    train_loss={train_loss:.4f}   Test_loss:{test_loss:.4f}    MSE:{mse:.4f}    RMSE:{rmse:.4f}    MAE:{mae:.4f}    MAPE:{mape:.4f}%     R2 Score:{r2:.4f}    Pearson correlation coefficient:{correlation:.4f}  P value：{p_value}")

            saveModel = ''
            if min_epoch_loss > train_loss:
                min_epoch_loss = train_loss
                if epoch >= epoch_nums - 100:
                    torch.save(model.state_dict(), "./Baseline_results/" + model_name +"_"+ target + "__windows_"+ str(TIME_STEPS) + "__" + train_time + '.pth')
                    print(f'save model : min_epoch_loss = {min_epoch_loss}')
                    saveModel = '_save_model'

            log_write.write(str(epoch) + "    Train Loss:" + str(train_loss) + ",Test Loss:" + str(test_loss) + ",mse:" + str(mse) + ",rmse:" + str(rmse) + ",mae:" + str(mae) + ",mape:"
                + str(mape) + ",r2:" + str(r2) + ",correlation:" + str(correlation) + ",p_value:" + str(p_value) + saveModel + "\n")

        plot_losses(train_loss_list, test_loss_list, model_name=model_name, target=target, TIME_STEPS=TIME_STEPS, train_time=train_time)


