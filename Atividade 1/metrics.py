"""File with evaluation metrics implementation"""
import numpy as np

class Metrics():
    def __init__(self):
        pass
    
    def RSS(self,y_true, y_predict):
         return ((y_true - y_predict) ** 2).sum()
    
    def TSS(self, y_true):
        aux = (y_true - np.mean(y_true)) ** 2
        return np.sum(aux)
    
    def R2(self, y_true, y_predict):
        return 1 - (self.RSS(y_true, y_predict) / self.TSS(y_true))
    
    def MSE(self, y_true, y_predict):
        return np.mean((y_true - y_predict) ** 2)