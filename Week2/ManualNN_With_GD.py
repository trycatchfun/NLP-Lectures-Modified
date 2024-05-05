import numpy as np

# Mini dataset
input_data = np.array([1, 2, 3])
weights = np.array([0, 2, 1])

learning_rate = 0.01
target = 0

predict = (input_data * weights).sum()

# print("predict: %s" % predict)

error = predict - target

# print("error: %s" % error)

# The derivation of loss function (MSE) would be the slope
slope = 2 * error 