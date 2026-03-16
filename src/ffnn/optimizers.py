import numpy as np


class Optimizer:
    def update(self, layer, dW, dB):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, layer, dW, dB):
        layer._weights -= self.lr * dW
        layer._bias -= self.lr * dB


class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m_w = {}
        self.v_w = {}
        self.m_b = {}
        self.v_b = {}

    def update(self, layer, dW, dB):
        self.t += 1
        key = id(layer)

        if key not in self.m_w:
            self.m_w[key] = np.zeros_like(dW)
            self.v_w[key] = np.zeros_like(dW)
            self.m_b[key] = np.zeros_like(dB)
            self.v_b[key] = np.zeros_like(dB)

        self.m_w[key] = self.beta1 * self.m_w[key] + (1 - self.beta1) * dW
        self.v_w[key] = self.beta2 * self.v_w[key] + (1 - self.beta2) * (dW ** 2)

        self.m_b[key] = self.beta1 * self.m_b[key] + (1 - self.beta1) * dB
        self.v_b[key] = self.beta2 * self.v_b[key] + (1 - self.beta2) * (dB ** 2)

        m_w_hat = self.m_w[key] / (1 - self.beta1 ** self.t)
        v_w_hat = self.v_w[key] / (1 - self.beta2 ** self.t)

        m_b_hat = self.m_b[key] / (1 - self.beta1 ** self.t)
        v_b_hat = self.v_b[key] / (1 - self.beta2 ** self.t)

        layer._weights -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
        layer._bias -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)