import numpy as np

def softmax(self, x):
        exp_x = np.exp(x)
        c = np.max(exp_x)
        a = exp_x - c
        softmax_y = (exp_x - c) / (np.sum(exp_x - c))
        return softmax_y
#二乗和誤差
'''def mean_squared_error(y, t):
    out = 0.5 * np.sum((y - t) ** 2)
    return out'''

def softmax_backward(self, dout):
    softmax_dx = dout * self.softmax_y * (1 - self.softmax)
    return softmax_dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
    
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = (self.y, self.t)

        return self.loss
    
    def backward(self, ):
        out = 0.5 * np.sum((self.y - self.t) ** 2)
        return out