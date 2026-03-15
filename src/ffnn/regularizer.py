import numpy as np

from .losses import l2_regularization, l2_regularization_derivative, l1_regularization, l1_regularization_derivative

class l2:
    def __init__(self, l2=0.01):
        self.l2 = l2
    
    def __call__(self, weights): # Menghitung penalti loss
        return l2_regularization(weights, self.l2)
    
    def gradient(self, weights): # Menghitung gradient tambahan
        return l2_regularization_derivative(weights, self.l2)
    
class l1:
    def __init__(self, l1=0.01):
        self.l1 = l1
    
    def __call__(self, weights): # Menghitung penalti loss
        return l1_regularization(weights, self.l1)
    
    def gradient(self, weights): # Menghitung gradient tambahan
        return l1_regularization_derivative(weights, self.l1)