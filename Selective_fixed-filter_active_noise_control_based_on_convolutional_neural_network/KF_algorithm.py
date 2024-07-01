import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import progressbar

class KalmanFilter:
    def __init__(self, len, q):
        self.L = len
        self.q = q
        self.W = torch.zeros(len, 1, requires_grad=True, dtype=torch.float)
        self.P = torch.eye(len, dtype=torch.float)
        self.Xd = torch.zeros(len, 1, dtype=torch.float)

    def feedforward(self, Xf):
        self.Xd = torch.roll(self.Xd, 1, 0)
        self.Xd[0, 0] = Xf
        yt = self.Xd.t() @ self.W
        return yt

    def update(self, ek):
        K = self.P @ self.Xd / (self.Xd.t() @ self.P @ self.Xd + self.q)
        self.W = self.W + K * ek
        self.P = (torch.eye(self.L) - K @ self.Xd.t()) @ self.P

    def lossfunction(self, y, d):
        e = d - y
        return e ** 2, e

    def _get_coeff_(self):
        return self.W.detach().numpy().flatten()

def train_kalman_filter(Model, Ref, Disturbance):
    bar = progressbar.ProgressBar(maxval=2 * Disturbance.shape[0],\
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    
    bar.start()
    Error_signal = []
    len_data = Disturbance.shape[0]
    for iter in range(len_data):
        # feedforward
        xin = Ref[iter]
        dis = Disturbance[iter]
        y = Model.feedforward(xin)
        loss, e = Model.lossfunction(y, dis)
        
        bar.update(2 * iter + 1)
        
        # update Kalman filter
        Model.update(e)
        Error_signal.append(e.item())
        
        bar.update(2 * iter + 2)
    bar.finish()
    return Error_signal

if __name__ == "__main__":
    # 生成一些模拟数据（替换为实际数据）
    L = 128  # 滤波器长度
    q = 0.00005  # 观测噪声
    N = 1000  # 数据长度
    Ref = np.random.randn(N)
    Disturbance = np.random.randn(N)

    # 初始化Kalman滤波模型
    model = KalmanFilter(L, q)

    # 训练模型
    error_signal = train_kalman_filter(model, Ref, Disturbance)

    # 观察误差变化
    import matplotlib.pyplot as plt
    plt.plot(error_signal)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Error Signal')
    plt.show()