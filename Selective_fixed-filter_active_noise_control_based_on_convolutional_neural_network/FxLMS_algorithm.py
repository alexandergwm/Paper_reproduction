import torch
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import scipy.signal as signal
import progressbar


class FxLMS_algorithm():
    """
    This class is used to create a FxLMS class
    """
    def __init__(self, len):
        self.Wc = torch.zeros(1, len, requires_grad=True, dtype=torch.float)
        self.Xd = torch.zeros(1, len, dtype=torch.float)

    def feedforward(self, Xf):
        self.Xd = torch.roll(self.Xd, 1, 1)
        self.Xd[0, 0] = Xf
        yt = self.Wc @ self.Xd.t()
        return yt
    
    def lossfunction(self, y, d):
        e = d - y
        return e ** 2, e
    
    def _get_coeff_(self):
        return self.Wc.detach().numpy()
    

def train_fxlms_algorithm(Model, Ref, Disturbance, Stepsize=0.0001):
    bar = progressbar.ProgressBar(maxval=2*Disturbance.shape[0],\
                                  widgets = [progressbar.Bar('=','[',']'),' ', progressbar.Percentage()])
    
    optimizer = optim.SGD([Model.Wc], lr=Stepsize)
    bar.start()
    Error_signal = []
    len_data = Disturbance.shape[0]
    for iter in range(len_data):
        # feedforward
        xin = Ref[iter]
        dis = Disturbance[iter]
        y = Model.feedforward(xin)
        loss, e = Model.lossfunction(y, dis)

        bar.update(2*iter+1)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Error_signal.append(e.item())
        
        bar.update(2*iter+2)
    bar.finish()
    return Error_signal
