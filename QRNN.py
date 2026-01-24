import math as m
import numpy as np
import pandas as pd
import time
from pyqpanda import *
from pyvqnet.nn.module import Module
from pyvqnet.qnn.quantumlayer import QuantumLayer
from pyvqnet.optim import adam
from pyvqnet.nn.loss import MeanSquaredError
from pyvqnet.tensor import tensor
from pyvqnet.tensor import QTensor

start = time.time() 

'''Quantum Circuit Definition'''


# Data input U_in matrix
def U_in(qubits, X_t):
    circuit = create_empty_circuit()
    theta_in = m.acos(X_t.item())
    circuit << RY(qubits[0], theta_in) \
    << RY(qubits[1], theta_in) \
    << RY(qubits[2], theta_in)
    return circuit


# Parameter matrix, 3*6 = 18 parameters
def U_theta(qubits, params):
    circuit = create_empty_circuit()
    for i in range(6):
        circuit << RX(qubits[i], params[3 * i]) \
        << RZ(qubits[i], params[3 * i + 1]) \
        << RX(qubits[i], params[3 * i + 2])
    return circuit


# First part of Hamiltonian simulation, 6 parameters
def H_X(qubits, params):
    circuit = create_empty_circuit()
    for i in range(6):
        circuit << RX(qubits[i], params[i])
    return circuit


# Second part of Hamiltonian simulation, 6 parameters
def H_ZZ(qubits, params):
    circuit = create_empty_circuit()
    for i in range(5):
        circuit << CNOT(qubits[i], qubits[i + 1]) \
        << RZ(qubits[i + 1], params[i]) \
        << CNOT(qubits[i], qubits[i + 1])
    circuit << CNOT(qubits[5], qubits[0]) \
    << RZ(qubits[0], params[5]) \
    << CNOT(qubits[5], qubits[0])
    return circuit


# Full parameterized circuit, total 18+6+6 = 30 parameters
def QRNN_VQC(qubits, params):
    params1 = params[0: 18]
    params2 = params[18: 18 + 6]
    params3 = params[18 + 6: 30]
    circuit = create_empty_circuit()
    circuit << U_theta(qubits, params1) \
    << H_X(qubits, params2) \
    << H_ZZ(qubits, params3)
    return circuit


# Batch data loader
def get_minibatch_data(x_data, true, batch_size):
    for i in range(0, x_data.shape[0] - batch_size + 1, batch_size):
        idxs = slice(i, i + batch_size)
        yield x_data[idxs], true[idxs]  # yeild  类似于return，返回后交出CPU使用权


# Build the complete quantum circuit
def QCircuit(input, weights, qlist, clist, machine):
    Amplitude = input[0:-1]
    x = input[-1]
    params = weights.squeeze()
    prog = QProg()
    circuit = create_empty_circuit()
    circuit << U_in(qlist, x)  # 数据输入
    circuit << amplitude_encode([qlist[3], qlist[4], qlist[5]], Amplitude, bool=False)
    circuit << QRNN_VQC(qlist, params[0: 30])
    prog << circuit
    qubit0_prob = machine.prob_run_dict(prog, qlist[0], -1)
    qubit1_prob = machine.prob_run_list(prog, qlist[1], -1)
    qubit2_prob = machine.prob_run_list(prog, qlist[2], -1)
    prob = list(qubit0_prob.values())
    return prob


# Amplitude calculation function. Amplitudes are not optimized parameters,
# they are only used to pass values between time steps
def Amplitude_Cacu(input, params):
    params = params.squeeze()
    Amplitude = input[0][0:-1]
    x = input[0][-1]
    qvm = CPUQVM()  # Create a local quantum virtual machine
    qvm.init_qvm()  # Initialize the quantum virtual machine
    qubits = qvm.qAlloc_many(6)
    prog = QProg()
    circuit = create_empty_circuit()
    circuit << U_in(qubits, x)   # Data input
    circuit << amplitude_encode([qubits[3], qubits[4], qubits[5]], Amplitude.to_numpy(), bool=False)  # 后三个比特的编码
    circuit << QRNN_VQC(qubits, params[0: 30])
    prog << circuit
    qubit0_prob = qvm.prob_run_list(prog, qubits[0], -1)
    qubit1_prob = qvm.prob_run_list(prog, qubits[1], -1)
    qubit2_prob = qvm.prob_run_list(prog, qubits[2], -1)
    Amplitude_2 = qvm.prob_run_list(prog, [qubits[3], qubits[4], qubits[5]], -1)
    Amplitude = np.sqrt(np.array(Amplitude_2))
    qvm.finalize()  # Release local virtual machine
    return Amplitude


'''Step2 定义一个继承于Module的机器学习模型类'''
param_num = 30  # Number of trainable parameters
qbit_num = 6  # Number of qubits in the quantum module



class QRNNModel(Module):
    def __init__(self):
        super(QRNNModel, self).__init__()
        # QuantumLayer allows the parameterized quantum circuit to be trained with auto-differentiation
        self.pqc = QuantumLayer(QCircuit, param_num, "cpu", qbit_num)
       # Initialize amplitude values
        self.Amplitude = np.array([1, 0, 0, 0, 0, 0, 0, 0])

    # Define forward pass
    def forward(self, X_t):
        xin = X_t[0]
        x_min = tensor.min(xin)
        x_max = tensor.max(xin)
        xin = (xin - x_min) / (x_max - x_min)
        for i in range(X_t.shape[1]):

            input = tensor.concatenate([QTensor(self.Amplitude), xin[i]], 0)
            input = tensor.unsqueeze(input)
            x = self.pqc(input)[0][1]
            param = np.array(self.pqc.parameters())[0].reshape((-1, 1)).to_numpy().squeeze()
            self.Amplitude = Amplitude_Cacu(input, param)
        return x


'''Step3 模型的训练'''


def train(data):
    # 1. 数据的读取
    data_P = data
    data_P_cha = []
    for i in range(len(data_P) - 1):
        data_P_cha.append(data_P[i + 1] - data_P[i])
    #  2. x_train y_train 的构建
    n = 16
    x_train = []
    y_train = []
    for i in range(len(data_P_cha) - n):
        x_train.append(data_P_cha[i:i + n])
        y_train.append(data_P_cha[i + n])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = x_train.reshape(-1, 16)
    # 3. 模型的训练
    batch_size = 1
    epoch = 1
    print("start training...........")
    for i in range(epoch):
        QRNNModel.train()
        count = 0
        loss = 0
        for data, true in get_minibatch_data(x_train, y_train, batch_size):
            data, true = QTensor(data), QTensor(true)
            # print(data, true)
            optimizer.zero_grad()  # 优化器中缓存梯度清零
            output = QRNNModel(data)  # 模型前向计算
            losss = MseLoss(true, output)  # 损失函数计算
            losss.backward()  # 损失反向传播
            optimizer._step()  # 优化器参数更新
            loss += losss.item()
            count += batch_size
        print(f"epoch:{i}, train_loss:{loss / count}\n")
    Param = np.array(QRNNModel.pqc.parameters())[0].reshape((-1, 1)).to_numpy().squeeze()
    return Param


'''Step3 性能测试'''


def Accuarcy(params, zhibiao, n):
    # 测试数据提取
    test_data = pd.read_csv("Test.csv")
    Data = list(test_data.loc[:, zhibiao])
    test_iterations = len(Data) - n - 1  # 测试次数，差分会再少一次
    Data_cha = []

    # 差分预处理
    for i in range(len(Data) - 1):
        Data_cha.append(Data[i + 1] - Data[i])

    #  这里两个用于记录测试集的真实值和预测值
    _ = []
    _ = []

    # 这个用于存放差值
    Y_pre_test_cha = []

    Ei_2_sum = 0  # 误差平方和初始化
    for j in range(test_iterations):
        zhenfu = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        X_t_cha = np.array(Data_cha[j: j + n + 1])
        X_t = np.array(Data[j: j + n + 2])

        # 数据预处理  注意，这里X_t[n]为预测数据真实值
        X_t_min = min(X_t_cha[0:n])
        X_t_max = max(X_t_cha[0:n])
        X_t_cha = (X_t_cha - X_t_min) / (X_t_max - X_t_min)
        xin = X_t_cha[0:-1].reshape(1, -1)
        Y_prediction = QRNNModel(xin)

        # 数据后处理
        Y_prediction = Y_prediction * (X_t_max - X_t_min) + X_t_min
        X_t_cha = (X_t_max - X_t_min) * X_t_cha + X_t_min

        # 这里得到的Y_prediciton为差值，需要进行处理
        Y_prediction = X_t[n] + Y_prediction
        Y_prediction = Y_prediction.item()
        # _.append(Y_prediction * sum + Y_tmin)
        _.append(Y_prediction)
        _.append(X_t[n + 1])

        # Ei = m.fabs(Y_t[n - 1] - Y_prediction * sum) / Y_t[n - 1]  # 计算误差
        if X_t[n] == 0:
            Ei = 0
        else:
            Ei = m.fabs(X_t[n] - Y_prediction) / X_t[n]  # 计算误差
        Ei_2 = Ei * Ei
        Ei_2_sum = Ei_2_sum + Ei_2

    accuarcy = 1 - m.sqrt(Ei_2_sum / test_iterations)
    return accuarcy, _, _


if __name__ == '__main__':
    df = pd.read_csv("datasets/Train_weather.csv")
    data_P = list(df.loc[:, 'Atmospheric Pressure'])
    data_Tmin = list(df.loc[:, 'Minimum Temperature'])
    data_Tmax = list(df.loc[:, 'Maximum Temperature'])
    data_RH = list(df.loc[:, 'Relative Humidity'])
    data_Ws = list(df.loc[:, 'Wind Speed'])
    QRNNModel = QRNNModel()
    optimizer = adam.Adam(QRNNModel.parameters(), lr=0.005)
    MseLoss = MeanSquaredError()

    param = train(data_P)
    accuarcy, _, _ = Accuarcy(param, 'Atmospheric Pressure', 16)
    print('气压精度为：' + str(accuarcy))
    np.savetxt(f"./QRNN_best_params/Atmospheric Pressure.txt", param)

    param = train(data_Tmin)
    accuarcy, _, _ = Accuarcy(param, 'Minimum Temperature', 16)
    print('最低温度精度为：' + str(accuarcy))
    np.savetxt(f"./QRNN_best_params/Minimum Temperature_{accuarcy}.txt", param)

    param = train(data_Tmax)
    accuarcy, _, _ = Accuarcy(param, 'Maximum Temperature', 16)
    print('最高温度精度为：' + str(accuarcy))
    np.savetxt(f"./QRNN_best_params/Maximum Temperature.txt", param)

    param = train(data_RH)
    accuarcy, _, _ = Accuarcy(param, 'Relative Humidity', 16)
    print('相对湿度精度为：' + str(accuarcy))
    np.savetxt(f"./QRNN_best_params/Relative Humidity.txt", param)
    #
    param = train(data_Ws)
    accuarcy, _, _ = Accuarcy(param, 'Wind Speed', 16)
    print('风速精度为：' + str(accuarcy))
    np.savetxt(f"./QRNN_best_params/Wind Speed.txt", param)

