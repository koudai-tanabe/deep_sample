import numpy as np
import sys
sys.path.append('Users/tanabekoudai/deep_sample')
import affine
import ReLu
import softmax_with_loss
from collections import OrderedDict



class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

class ReLu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

def softmax(x):
        exp_x = np.exp(x)
        c = np.max(exp_x)
        a = exp_x - c
        softmax_y = (exp_x - c) / (np.sum(exp_x - c))
        return softmax_y
#二乗和誤差
def mean_squared_error(y, t):
    out = 0.5 * np.sum((y - t) ** 2)
    return out

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
        self.loss = mean_squared_error(self.y, self.t)

        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size     #(self.y-self.t)って何？
        return dx

#勾配
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.shape[0]):
        tmp_val = x[idx]
        x[idx] = tmp_val + h    #f(x+h)の計算
        fxh1 = f(x)

        x[idx] = tmp_val - h    #f(x+h)の計算
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
    return grad


class NetWork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size1)
        self.params['b1'] = np.random.randn(hidden_size1,)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size1, hidden_size2)
        self.params['b2'] = np.random.randn(hidden_size2,)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size2, output_size)
        self.params['b3'] = np.random.randn(output_size)

    #レイヤの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['sigmoid'] = Sigmoid()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['ReLu'] = ReLu()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        self.lastlayer = SoftmaxWithLoss()

    #Affineのforwardに全て入れる
    def predict(self, x):
        for layer in self.layers.values():  #Affine(self.pa..)とか
            x = layer.forward(x)
        return x    #返したxを次のAffine.forwardに入る

    #predectの（最後に出力された）xを出力層にいれる
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastlayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    #数値計算によって勾配を求める
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        grads['W3'] = numerical_gradient(loss_W, self.params['W3'])
        grads['b3'] = numerical_gradient(loss_W, self.params['b3'])

        return grads

    #誤差逆伝播法によって勾配を求める
    def gradient(self, x, t):
        #forward
        self.loss(x, t)

        #backward
        dout = 1
        dout = self.lastlayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        #dout代入し求めた勾配をdic型に入れる
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        grads['W3'] = self.layers['Affine3'].dW
        grads['b3'] = self.layers['Affine3'].db

        return grads

#データの読み込み
import kmnist



#勾配確認
'''from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_train) = load_mnist(normalize=True, one_hot_label=True)

net = NetWork(input_size=784, hidden_size1=4, hidden_size2=5, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = net.numerical_gradient(x_batch, t_batch)
grad_backprop = net.gradient(x_batch, t_batch)

for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))'''