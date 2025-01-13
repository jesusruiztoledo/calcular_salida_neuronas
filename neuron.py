import numpy as np
import funciones as f
class Neuron:
    def __init__ (self, weights, bias, func):
        self.weights = weights
        self.bias = bias
        self.func = func
    def changeBias(self, bias):
        self.bias = bias
    def run(self, input_data):
        weighted_sum = sum(np.array(self.weights) * np.array(input_data)) + self.bias
        if self.func == "relu":
            return f.Relu(weighted_sum)
        elif self.func == "sigmoid":
            return f.Sigmoid(weighted_sum)
        elif self.func == "tanh":
            return f.Tanh(weighted_sum)
        elif self.func == "binary":
            return f.Binary(weighted_sum)
        else:
            return "Función no válida"
        

    
    