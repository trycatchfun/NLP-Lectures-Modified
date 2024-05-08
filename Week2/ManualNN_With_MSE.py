
import numpy as np
from sklearn.metrics import mean_squared_error

# Mini Dataset

input_data = np.array([0, 3, 5, 7, 8, 2])

weights = {'node0': np.array([2, 1]),
           'node1': np.array([1, 2]),
           'output': np.array([1, 1])}

newWeights = {'node0': np.array([0, 1]),
              'node1': np.array([1, 0]),
              'output': np.array([1, 1])}

ActualTargets = [1, 2, 3, 3, 9, 3]


def ReLu(x):
    out = max(0, x)
    return out


def predict_with_network(inputd, weights):

    node0_input = (inputd * weights['node0']).sum()
    node0_output = ReLu(node0_input)

    node1_input = (inputd * weights['node1']).sum()
    node1_output = ReLu(node1_input)

    hidden_layer_values = np.array([node0_output, node1_output])