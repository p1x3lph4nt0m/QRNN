import math as m
import numpy as np
import pandas as pd
import time
import os
from datetime import datetime
from pyqpanda import *
from pyvqnet.nn.module import Module
from pyvqnet.qnn.quantumlayer import QuantumLayer
from pyvqnet.optim import adam
from pyvqnet.nn.loss import MeanSquaredError
from pyvqnet.tensor import tensor
from pyvqnet.tensor import QTensor

start = time.time()
param_num = int()
qbit_num = int()
epoch = int()
n = int()
batch_size = int()
weight_log_interval = int()
df = pd.read_csv("datasets/train_weather.csv")
data_list = {
    'Atmospheric Pressure': list(df.loc[:, 'Atmospheric Pressure']),
    'Minimum Temperature': list(df.loc[:, 'Minimum Temperature']),
    'Maximum Temperature': list(df.loc[:, 'Maximum Temperature']),
    'Relative Humidity': list(df.loc[:, 'Relative Humidity']),
    'Wind Speed': list(df.loc[:, 'Wind Speed'])
}

def tweakable_parameters():
    global param_num, qbit_num, epoch, n, batch_size, weight_log_interval
    param_num = 30
    qbit_num = 6
    epoch = 3  # Reduced from 100 for testing
    n = 16
    batch_size = 1
    weight_log_interval = 1  # Log every epoch for visibility
tweakable_parameters()

def scalar(x):
    return float(x.item()) if hasattr(x, "item") else float(x)

def log(msg):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")

'''Quantum Circuit Definition'''

def U_in(qubits, X_t):
    circuit = create_empty_circuit()
    theta_in = m.acos(X_t.item())
    circuit << RY(qubits[0], theta_in) \
            << RY(qubits[1], theta_in) \
            << RY(qubits[2], theta_in)
    return circuit

def U_theta(qubits, params):
    circuit = create_empty_circuit()
    for i in range(6):
        circuit << RX(qubits[i], scalar(params[3 * i])) \
                << RZ(qubits[i], scalar(params[3 * i + 1])) \
                << RX(qubits[i], scalar(params[3 * i + 2]))
    return circuit

def H_X(qubits, params):
    circuit = create_empty_circuit()
    for i in range(6):
        circuit << RX(qubits[i], scalar(params[i]))
    return circuit

def H_ZZ(qubits, params):
    circuit = create_empty_circuit()
    for i in range(5):
        circuit << CNOT(qubits[i], qubits[i + 1]) \
                << RZ(qubits[i + 1], scalar(params[i])) \
                << CNOT(qubits[i], qubits[i + 1])
    circuit << CNOT(qubits[5], qubits[0]) \
            << RZ(qubits[0], scalar(params[5])) \
            << CNOT(qubits[5], qubits[0])
    return circuit

def QRNN_VQC(qubits, params):
    params1 = params[0: 18]
    params2 = params[18: 24]
    params3 = params[24: 30]
    circuit = create_empty_circuit()
    circuit << U_theta(qubits, params1) \
            << H_X(qubits, params2) \
            << H_ZZ(qubits, params3)
    return circuit

def get_minibatch_data(x_data, true, batch_size):
    for i in range(0, x_data.shape[0] - batch_size + 1, batch_size):
        idxs = slice(i, i + batch_size)
        yield x_data[idxs], true[idxs]

def QCircuit(input, weights, qlist, clist, machine):
    Amplitude = input[0:-1]
    x = input[-1]
    params = weights.squeeze()
    prog = QProg()
    circuit = create_empty_circuit()
    circuit << U_in(qlist, x)
    qvec = QVec()
    qvec.append(qlist[3])
    qvec.append(qlist[4])
    qvec.append(qlist[5])
    circuit << amplitude_encode(qvec, Amplitude.tolist(), False)

    # circuit << amplitude_encode([qlist[3], qlist[4], qlist[5]], Amplitude, bool=False)
    circuit << QRNN_VQC(qlist, params[0:30])
    prog << circuit
    prob = list(machine.prob_run_dict(prog, qlist[0], -1).values())
    return prob

def Amplitude_Cacu(input, params):
    params = params.squeeze()
    Amplitude = input[0][0:-1]
    x = input[0][-1]
    qvm = CPUQVM()
    qvm.init_qvm()
    qubits = qvm.qAlloc_many(6)
    prog = QProg()
    circuit = create_empty_circuit()
    circuit << U_in(qubits, x)
    qvec = QVec()
    qvec.append(qubits[3])
    qvec.append(qubits[4])
    qvec.append(qubits[5])
    circuit << amplitude_encode(qvec, Amplitude.to_numpy(), False)
    circuit << QRNN_VQC(qubits, params[0:30])
    prog << circuit
    Amplitude_2 = qvm.prob_run_list(prog, [qubits[3], qubits[4], qubits[5]], -1)
    qvm.finalize()
    return np.sqrt(np.array(Amplitude_2), dtype=np.float32)

class QRNNModel(Module):
    def __init__(self):
        super(QRNNModel, self).__init__()
        self.pqc = QuantumLayer(QCircuit, param_num, "cpu", qbit_num)
        self.Amplitude = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    def forward(self, X_t):
        log("Forward pass started")
        xin = X_t[0]
        x_min = tensor.min(xin)
        x_max = tensor.max(xin)
        xin = (xin - x_min) / (x_max - x_min)
        for i in range(X_t.shape[1]):
            log(f"Time step {i+1}/{X_t.shape[1]}")
            amp_tensor = QTensor(self.Amplitude.astype(np.float32))
            x_step = QTensor(xin[i].to_numpy().astype(np.float32))
            input = tensor.concatenate([amp_tensor, x_step], 0)
            # input = tensor.concatenate([QTensor(self.Amplitude), xin[i]], 0)
            input = tensor.unsqueeze(input)
            x = self.pqc(input)[0][1]
            param = np.array(self.pqc.parameters())[0].reshape((-1, 1)).squeeze()
            self.Amplitude = Amplitude_Cacu(input, param)
        log("Forward pass completed")
        return tensor.unsqueeze(x, 0)

def train(data):
    log("Preparing training data")
    data_P_cha = [data[i+1] - data[i] for i in range(len(data)-1)]

    x_train = np.array([data_P_cha[i:i+n] for i in range(len(data_P_cha)-n)]).reshape(-1,n)
    y_train = np.array([data_P_cha[i+n] for i in range(len(data_P_cha)-n)])

    log(f"Training samples: {len(x_train)}")

    for ep in range(epoch):
        log(f"Epoch {ep+1} started")
        QRNNModel.train()
        loss = 0
        count = 0

        for step, (data, true) in enumerate(get_minibatch_data(x_train, y_train, batch_size)):
            data = QTensor(data.astype(np.float32))
            true = QTensor(true.astype(np.float32)).reshape([1,1])

            optimizer.zero_grad()
            output = QRNNModel(data)
            losss = MseLoss(true, output)
            losss.backward()
            optimizer._step()

            loss += losss.item()
            count += batch_size

        log(f"Epoch {ep+1} completed. Avg Loss {loss/count:.6f}")

        if (ep+1) % weight_log_interval == 0:
            raw_params = QRNNModel.pqc.parameters()[0].reshape((-1,))
            Param = np.array([scalar(p) for p in raw_params], dtype=np.float32)
            print(f"--- Weights at epoch {ep+1} ---")
            print(Param)

    raw_params = QRNNModel.pqc.parameters()[0].reshape((-1,))
    Param = np.array([scalar(p) for p in raw_params], dtype=np.float32)

    return Param


# def Accuarcy(params, zhibiao, n):
#     log(f"Evaluating {zhibiao}")
#     test_data = pd.read_csv("datasets/test_weather.csv")
#     Data = list(test_data.loc[:, zhibiao])
#     Data_cha = [Data[i+1]-Data[i] for i in range(len(Data)-1)]
#     test_iterations = len(Data)-n-1
#     Ei_2_sum = 0
#     for j in range(test_iterations):
#         if j % 20 == 0:
#             log(f"Testing step {j}/{test_iterations}")
#         X_t_cha = np.array(Data_cha[j:j+n+1])
#         X_t = np.array(Data[j:j+n+2])
#         X_t_min = min(X_t_cha[:n])
#         X_t_max = max(X_t_cha[:n])
#         X_t_cha = (X_t_cha - X_t_min) / (X_t_max - X_t_min)
#         xin = X_t_cha[:-1].reshape(1,-1)
#         Y_prediction = scalar(QRNNModel(xin))
#         Y_prediction = Y_prediction*(X_t_max-X_t_min)+X_t_min
#         Y_prediction = X_t[n] + scalar(Y_prediction)
#         Ei = 0 if X_t[n]==0 else m.fabs(X_t[n]-Y_prediction)/X_t[n]
#         Ei_2_sum += Ei*Ei
#     accuarcy = 1 - m.sqrt(Ei_2_sum/test_iterations)
#     log(f"Accuracy for {zhibiao}: {accuarcy:.4f}")
#     return accuarcy

def dump_predictions(params, zhibiao):
    log(f"Generating predictions for {zhibiao}")

    test_data = pd.read_csv("datasets/test_weather.csv")
    Data = list(test_data.loc[:, zhibiao])
    Data_cha = [Data[i+1]-Data[i] for i in range(len(Data)-1)]

    predictions = []

    for j in range(len(Data)-n-1):
        X_t_cha = np.array(Data_cha[j:j+n+1])
        X_t = np.array(Data[j:j+n+2])

        X_t_min = min(X_t_cha[:n])
        X_t_max = max(X_t_cha[:n])

        X_t_cha = (X_t_cha - X_t_min) / (X_t_max - X_t_min)
        xin = X_t_cha[:-1].reshape(1,-1)

        Y_prediction = scalar(QRNNModel(xin))
        Y_prediction = Y_prediction*(X_t_max-X_t_min)+X_t_min
        Y_prediction = X_t[n] + scalar(Y_prediction)

        predictions.append(Y_prediction)

    os.makedirs("predictions", exist_ok=True)
    safe_name = zhibiao.replace(" ", "_")
    np.savetxt(f"predictions/{safe_name}_predictions.txt", predictions)
    log(f"Predictions saved for {zhibiao}")


if __name__ == '__main__':
    # script_name = os.path.basename(__file__)
    # backup_name = f"backup_{script_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    # os.system(f"cp {script_name} {backup_name}")
    # log(f"Script backed up as {backup_name}")
    log("Program started")
    
    QRNNModel = QRNNModel()
    optimizer = adam.Adam(QRNNModel.parameters(), lr=0.005)
    MseLoss = MeanSquaredError()

    for name, data in data_list.items():
        log(f"Training {name}")
        param = train(data)
        dump_predictions(param, name)
        safe_name = name.replace(" ", "_")
        np.savetxt(f"./best_params/{safe_name}.txt", param)
        log(f"Parameters saved for {name}")
        log(f"✓ {name} COMPLETE - files written to ./predictions/ and ./best_params/")

    log(f"Total runtime: {time.time()-start:.2f} seconds")
    log("✓ PROGRAM FINISHED - All models trained and files saved")
