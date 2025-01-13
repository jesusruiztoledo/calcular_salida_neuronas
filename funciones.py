# Funciones de activación

import numpy as np

def Relu(weighted_sum):
    return max(0, weighted_sum)
def Sigmoid(weighted_sum):
    return 1 / (1 + np.exp(-weighted_sum))
def Tanh(weighted_sum):
    return np.tanh(weighted_sum)
def Binary(weighted_sum):
    return 1 if weighted_sum >= 0 else 0
