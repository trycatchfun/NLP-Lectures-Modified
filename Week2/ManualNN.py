import numpy as np

# Mini Dataset
input_data = np.array([5, 7, 8, 1])

weights = {'node0': np.array([1, 1]),
           'node1': np.array([-1, 1]),
           'output': np.array([2, -1])}


# Activation Function
def ReLu(x):
    out = max(0, x)
    return out


def predict_with_NN(i