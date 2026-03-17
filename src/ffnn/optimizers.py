import numpy as np


class Optimizer:
    def update(self, layer, dW, dB, dG):
        raise NotImplementedError
    def update_gamma(self, layer, dG):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, layer, dW, dB):
        layer._weights -= self.lr * dW
        layer._bias -= self.lr * dB

    def update_gamma(self, layer, dG):
        layer._gammas -= self.lr * dG

class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = {}
        self.m_w = {}
        self.v_w = {}
        self.m_b = {}
        self.v_b = {}
        self.m_g = {}
        self.v_g = {}

    def update(self, layer, dW, dB):
        key = id(layer)

        if key not in self.m_w:
            self.t[key] = 0
            self.m_w[key] = np.zeros_like(dW)
            self.v_w[key] = np.zeros_like(dW)
            self.m_b[key] = np.zeros_like(dB)
            self.v_b[key] = np.zeros_like(dB)
        
        self.t[key] += 1
        t_current = self.t[key]

        self.m_w[key] = self.beta1 * self.m_w[key] + (1 - self.beta1) * dW
        self.v_w[key] = self.beta2 * self.v_w[key] + (1 - self.beta2) * (dW ** 2)

        self.m_b[key] = self.beta1 * self.m_b[key] + (1 - self.beta1) * dB
        self.v_b[key] = self.beta2 * self.v_b[key] + (1 - self.beta2) * (dB ** 2)

        m_w_hat = self.m_w[key] / (1 - self.beta1 ** t_current)
        v_w_hat = self.v_w[key] / (1 - self.beta2 ** t_current)

        m_b_hat = self.m_b[key] / (1 - self.beta1 ** t_current)
        v_b_hat = self.v_b[key] / (1 - self.beta2 ** t_current)

        layer._weights -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
        layer._bias -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

    def update_gamma(self, layer, dG):
        key = id(layer)

        if key not in self.m_g:
            self.t[key] = 0
            self.m_g[key] = np.zeros_like(dG)
            self.v_g[key] = np.zeros_like(dG)
        
        self.t[key] += 1
        t_current = self.t[key]

        self.m_g[key] = self.beta1 * self.m_g[key] + (1 - self.beta1) * dG
        self.v_g[key] = self.beta2 * self.v_g[key] + (1 - self.beta2) * (dG ** 2)

        m_g_hat = self.m_g[key] / (1 - self.beta1 ** t_current)
        v_g_hat = self.v_g[key] / (1 - self.beta2 ** t_current)

        layer._gammas -= self.lr * m_g_hat / (np.sqrt(v_g_hat) + self.epsilon)
