import glob
from scipy.signal import find_peaks
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import random
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
import torch.utils.data as Data
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error, r2_score

import BioforceNet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Estimation of vGRF from Insole Data
def get_vGRF(path, TIME_STEPS, stride, pre_len, startNum, peak_weight):
    """
    Read data and generate training set and target values based on time steps and stride length
    """
    print('Estimating vertical ground reaction force')
    X, y, z = [], [], []
    df = pd.read_excel(path)
    label_data = df.iloc[:, 15].tolist()
    gate_weights = calculate_gate_weights_with_min_width(np.array(label_data), peak_radius=3, peak_weight=peak_weight,min_width=12)

    length = len(label_data)
    for i in range(startNum, length - TIME_STEPS + 1, stride):
        window_data = df.iloc[i:i + TIME_STEPS, :]
        weight = gate_weights[i:i + TIME_STEPS]
        if not window_data.isnull().values.any() and len(window_data) == TIME_STEPS:
            X.append(window_data.iloc[:, :8])
            y.append(window_data.iloc[-pre_len:, 15])
            z.append(weight[-pre_len:])

    len_train = len(X)
    print(f'{len_train} pieces of data in total')
    return np.array(X), np.array(y), np.array(z), len(X)

# Estimation of TBF from Insole Data
def get_TBF(path, TIME_STEPS, stride, pre_len, startNum, peak_weight):
    """
    Read data and generate training set and target values based on time steps and stride length
    """
    print('Estimating tibia bone force')

    X, y, z = [], [] ,[]
    df = pd.read_excel(path)

    All_lable_data = df.iloc[:, 16].tolist()
    gate_weights = calculate_gate_weights_with_min_width(np.array(All_lable_data), peak_radius=3, peak_weight=peak_weight,min_width = 8)

    length = len(All_lable_data)
    for i in range(startNum, length - TIME_STEPS + 1, stride):
        window_data = df.iloc[i:i + TIME_STEPS, :]
        weight = gate_weights[i:i + TIME_STEPS]
        if not window_data.isnull().values.any() and len(window_data) == TIME_STEPS:
            X.append(window_data.iloc[:,:8])
            y.append(window_data.iloc[-pre_len:, 16])
            z.append(weight[-pre_len:])

    len_train = len(X)
    print(f'{len_train} pieces of data in total')
    return np.array(X), np.array(y), np.array(z), len_train

# Gate weights for key signals in the sequence
def calculate_gate_weights_with_min_width(grf_signals, peak_radius=3, peak_weight=2.0, min_width=8):

    """
        Calculate the weight based on the local maximum of the signal and filter out the maximum with smaller width.
        :param grf_signals: signals (batch_2, time_steps)
        :param peak_radius: Range of influence near the maximum value (radius)
        :param peak_weight: Maximum and nearby weight values
        :param min_width: minimum peak width
        :Return: weights (batch_size, time_steps)
    """

    gate_weights = torch.ones(len(grf_signals))
    peaks, _ = find_peaks(grf_signals, width=min_width)
    for idx in peaks:
        start = max(0, idx - peak_radius)
        end = min(len(grf_signals), idx + peak_radius + 1)
        gate_weights[start:end] = peak_weight
    return gate_weights


# Custom loss with gating weights
class WeightMSELoss(nn.Module):
    def __init__(self):
        super(WeightMSELoss, self).__init__()

    def forward(self, predictions, targets, gate_weights):
        """
        : parampredictions: Model's predicted values (batch_size, *)
        : param targets: true values (batch_size, *)
        : paramgatew_weights: Gated weights (batch_size,)
        : Return: Calculated Gate MSE loss
        """
        mse_loss = (predictions - targets) ** 2
        gate_mse_loss = gate_weights * mse_loss
        return torch.mean(gate_mse_loss)


# Experiment configuration
class ExP():
    def __init__(self, seed_n):
        self.seed_n = seed_n

        self.batch_size = 64
        self.n_epochs = 3
        self.lr = 0.001  # vgrf     Change to 0.0001 after 50 rounds
        # self.lr = 0.01 # tbf      Change to 0.001 after 50 rounds
        self.b1 = 0.5
        self.b2 = 0.999

        self.pre_len = 1
        self.startNum = 0
        self.INPUT_DIMS = 8
        self.stride = 10

        self.train_time = time.strftime('%Y_%m_%d-%H_%M_%S', time.localtime())
        self.target = 'vgrf'
        # self.target = 'tbf'
        self.TIME_STEPS = 60
        self.depth = 1
        self.peak_weight = 2.0

        self.train_loss_list = []
        self.test_loss_list = []

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_MSE = torch.nn.MSELoss().to(device)  # 均方误差损失函数
        self.criterion_WeightMSE = WeightMSELoss().to(device)  # 均方误差损失函数
        self.model = BioforceNet.BioforceNet(pre_len=self.pre_len, depth=self.depth).to(device)

    def train(self):
        path = 'Traindata'
        file_paths = glob.glob(f"{path}/*.xlsx")

        # encoder_dep_list = [1,2,3,4,5,6,7,8,9,10]
        # windows_list = [30,40,50,60,70,80,90,100]
        peak_weight_list = [2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
        for peak_weight in peak_weight_list:
            # self.TIME_STEPS = windows
            # self.depth = encoder_dep
            self.peak_weight = peak_weight
            self.train_time = time.strftime('%Y_%m_%d-%H_%M_%S', time.localtime())
            self.model = BioforceNet.BioforceNet(pre_len=self.pre_len, depth=self.depth).to(device)

            self.train_loss_list = []
            self.test_loss_list = []
            self.log_write = open("./BioforeNet_results/" + self.target + "__windows_" + str(self.TIME_STEPS) + "__depth_" + str(self.depth) + "__" + self.train_time + "__" + str(self.peak_weight) + "__log" + ".txt", "w")
            self.log_write.write("seed is " + str(self.seed_n) + "\n")

            # Read all .xlsx files in folder
            train_X = []
            train_Y = []
            train_Z = []

            for path in file_paths:
                print('path', path)
                if self.target == 'vgrf':
                    X, y, z, len_train = get_vGRF(path, self.TIME_STEPS, self.stride, self.pre_len, self.startNum,self.peak_weight)
                elif self.target == 'tbf':
                    X, y, z, len_train = get_TBF(path, self.TIME_STEPS, self.stride, self.pre_len, self.startNum,self.peak_weight)
                else:
                    X, y, z, len_train = None,None,None,None
                    print('Data reading exception')
                train_X.append(X)
                train_Y.append(y)
                train_Z.append(z)
                break

            # Combine all data
            train_X = np.vstack(train_X)
            train_Y = np.vstack(train_Y)
            train_Z = np.vstack(train_Z)

            # Shuffle data
            N = train_X.shape[0]
            index = np.random.permutation(N)
            data = train_X[index, :, :]
            data = np.transpose(data, (0, 2, 1))
            data = np.expand_dims(data, axis=1)
            label = train_Y[index]
            weight = train_Z[index]

            # Split into train and test sets
            num_train = round(N * 0.8)
            train_data = data[:num_train]
            train_label = label[:num_train]
            train_weight = weight[:num_train]
            test_data = data[num_train:]
            test_label = label[num_train:]
            print(f'There are {len(train_data)} pieces of data available for training')
            print(f'There are {len(train_data)} pieces of data available for testing')

            # Create dataloaders
            train_data = torch.from_numpy(train_data).float().to(device)
            train_label = torch.from_numpy(train_label).float().to(device)
            train_weight = torch.from_numpy(train_weight).float().to(device)
            dataset = torch.utils.data.TensorDataset(train_data, train_label, train_weight)
            self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

            test_data = torch.from_numpy(test_data).float().to(device)
            test_label = torch.from_numpy(test_label).float().to(device)
            test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
            self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size,shuffle=True)

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

            # Set to infinity
            min_epoch_loss = float('inf')

            for e in range(self.n_epochs):
                if e == 50:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = 0.0001

                # Training
                self.model.train()
                train_all_loss = 0
                for insole_data, label, weight in self.dataloader:
                    insole_data, label, weight = (Variable(insole_data.to(device)), Variable(label.to(device)).squeeze(-1),
                                                  Variable(weight.to(device)).squeeze(-1))
                    _, outputs = self.model(insole_data)
                    loss = self.criterion_WeightMSE(outputs, label, weight)
                    train_all_loss += loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                train_loss_aver = train_all_loss / len(self.dataloader)
                self.train_loss_list.append(float(train_loss_aver))

                # Testing : Model performance evaluation after one round of training
                self.model.eval()
                test_all_loss = 0
                prediction_list, label_list = [], []
                with torch.no_grad():
                    for insole_data, label in self.test_dataloader:
                        insole_data, label = Variable(insole_data.to(device)), Variable(label.to(device)).squeeze(-1)
                        _, predictions = self.model(insole_data)
                        test_all_loss += self.criterion_MSE(predictions, label)

                        predictions = predictions.cpu().numpy()
                        prediction_list = prediction_list + list(predictions)
                        label = label.cpu().numpy()
                        label_list = label_list + list(label)

                loss_test_aver = test_all_loss / len(self.test_dataloader)
                self.test_loss_list.append(float(loss_test_aver))

                y_true, y_pred = np.array(prediction_list).flatten(), np.array(label_list).flatten()
                mse = mean_squared_error(y_true, y_pred)
                correlation, p_value = pearsonr(y_true, y_pred)
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                mae = np.mean(np.abs(y_true - y_pred))
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                r2 = r2_score(y_true, y_pred)

                print(f'Epoch {e + 1}, Train Loss: {train_loss_aver:.4f}, Test Loss: {loss_test_aver:.4f}, mse: {mse:.4f}, rmse: {rmse:.4f},'
                      f' mae: {mae:.4f}, mape: {mape:.4f}, r2: {r2:.4f}, correlation: {correlation:.4f}, p_value: {p_value:.4f}')

                # Save the model at minimum loss
                saveModel = ''
                if min_epoch_loss > train_loss_aver:
                    min_epoch_loss = train_loss_aver
                    # Only save the model in the last 100 rounds
                    if e >= self.n_epochs - 100:
                        torch.save(self.model.state_dict(),"./BioforeNet_results/" + self.target + "__windows_" + str(self.TIME_STEPS)
                                   + "__depth_" + str(self.depth) + "__" + self.train_time + '.pth')
                        saveModel = '_save_model'
                        print(f'save model : min_epoch_loss = {min_epoch_loss}')

                self.log_write.write(str(e) + "    Train Loss:" + str(train_loss_aver) + ",Test Loss:" + str(loss_test_aver) + ",mse:" + str(mse) + ",rmse:" + str(rmse) + ",mae:" + str(mae) + ",mape:" + str(
                    mape) + ",r2:" + str(r2) + ",Pearson correlation coefficient:" + str(correlation) + ",p_value:" + str(p_value) + saveModel + "\n")

            self.plot_losses()


    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_loss_list, label="Train Loss")
        plt.plot(self.test_loss_list, label="Test Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Train and Test Loss over Epochs')
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig("./BioforeNet_results/" + self.target + "__windows_"
                    + str(self.TIME_STEPS) + "__depth_" + str(self.depth) + "__" + self.train_time + '.png')
        plt.close()



if __name__ == "__main__":
    seed_n = np.random.randint(2024)
    print('Seed:', seed_n)
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)

    exp = ExP(seed_n)
    exp.train()
