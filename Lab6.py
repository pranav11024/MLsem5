#A1
import math

def summation_unit(inputs, weights):
    return sum(i * w for i, w in zip(inputs, weights))

def step_function(x):
    return 1 if x >= 0 else 0

def bipolar_step_function(x):
    return 1 if x >= 0 else -1

def sigmoid_function(x):
    return 1 / (1 + math.exp(-x))

def tanh_function(x):
    return math.tanh(x)

def relu_function(x):
    return max(0, x)

def leaky_relu_function(x):
    return x if x >= 0 else 0.01 * x

def comparator_unit(predicted, actual):
    return actual - predicted

#A2
def train_perceptron_and_gate(inputs, outputs, epochs=1000, lr=0.05):
    weights = [10, 0.2, -0.75]
    for epoch in range(epochs):
        total_error = 0
        for i in range(len(inputs)):
            summation = summation_unit([1] + inputs[i], weights)
            prediction = step_function(summation)
            error = comparator_unit(prediction, outputs[i])
            total_error += error ** 2
            for j in range(len(weights)):
                weights[j] += lr * error * ([1] + inputs[i])[j]
        if total_error <= 0.002:
            break
    return weights, epoch

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [0, 0, 0, 1]
weights, epochs = train_perceptron_and_gate(inputs, outputs)

#A3
def train_with_activation(inputs, outputs, activation_func, epochs=1000, lr=0.05):
    weights = [10, 0.2, -0.75]
    for epoch in range(epochs):
        total_error = 0
        for i in range(len(inputs)):
            summation = summation_unit([1] + inputs[i], weights)
            prediction = activation_func(summation)
            error = comparator_unit(prediction, outputs[i])
            total_error += error ** 2
            for j in range(len(weights)):
                weights[j] += lr * error * ([1] + inputs[i])[j]
        if total_error <= 0.002:
            break
    return weights, epoch

# Example using Sigmoid Function
weights, epochs = train_with_activation(inputs, outputs, sigmoid_function)

#A4
learning_rates = [0.1 * i for i in range(1, 11)]
iterations = []

for lr in learning_rates:
    _, epoch = train_perceptron_and_gate(inputs, outputs, lr=lr)
    iterations.append(epoch)

#A5
inputs_xor = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs_xor = [0, 1, 1, 0]
weights, epochs = train_perceptron_and_gate(inputs_xor, outputs_xor)

#A6
def train_perceptron_customers(data, labels, epochs=1000, lr=0.05):
    weights = [0.1, 0.2, 0.3, 0.4]  # Example initial weights
    for epoch in range(epochs):
        total_error = 0
        for i in range(len(data)):
            summation = summation_unit([1] + data[i], weights)
            prediction = sigmoid_function(summation)
            error = comparator_unit(prediction, labels[i])
            total_error += error ** 2
            for j in range(len(weights)):
                weights[j] += lr * error * ([1] + data[i])[j]
        if total_error <= 0.002:
            break
    return weights, epoch

customer_data = [
    [20, 6, 2],
    [16, 3, 6],
    [27, 6, 2],
    [19, 1, 2],
    [24, 4, 2],
    [22, 1, 5],
    [15, 4, 2],
    [18, 4, 2],
    [21, 1, 4],
    [16, 2, 4]
]

high_value_labels = [1, 1, 1, 0, 1, 0, 1, 1, 0, 0]

weights, epochs = train_perceptron_customers(customer_data, high_value_labels)

#A7
import numpy as np

def pseudo_inverse_solution(data, labels):
    X = np.array([[1] + d for d in data])
    Y = np.array(labels)
    pseudo_inv = np.linalg.pinv(X)
    weights = np.dot(pseudo_inv, Y)
    return weights

weights_pseudo = pseudo_inverse_solution(customer_data, high_value_labels)

#A8
def sigmoid_derivative(x):
    return x * (1 - x)

def train_nn_and_gate(inputs, outputs, epochs=1000, lr=0.05):
    weights_input_hidden = [0.5, -0.6, 0.2]
    weights_hidden_output = [0.4, -0.7]
    for epoch in range(epochs):
        total_error = 0
        for i in range(len(inputs)):
            hidden_input = summation_unit([1] + inputs[i], weights_input_hidden)
            hidden_output = sigmoid_function(hidden_input)
            output_input = summation_unit([1, hidden_output], weights_hidden_output)
            output = sigmoid_function(output_input)
            error = comparator_unit(output, outputs[i])
            total_error += error ** 2
            delta_output = error * sigmoid_derivative(output)
            delta_hidden = delta_output * sigmoid_derivative(hidden_output) * weights_hidden_output[1]
            weights_hidden_output[0] += lr * delta_output * 1
            weights_hidden_output[1] += lr * delta_output * hidden_output
            for j in range(len(weights_input_hidden)):
                weights_input_hidden[j] += lr * delta_hidden * ([1] + inputs[i])[j]
        if total_error <= 0.002:
            break
    return weights_input_hidden, weights_hidden_output, epoch

weights_input_hidden, weights_hidden_output, epochs = train_nn_and_gate(inputs, outputs)

#A9
weights_input_hidden, weights_hidden_output, epochs = train_nn_and_gate(inputs_xor, outputs_xor)

#A10
def train_perceptron_2_outputs(inputs, outputs, epochs=1000, lr=0.05):
    weights = [[0.1, 0.2, -0.1], [0.05, 0.3, -0.25]]
    for epoch in range(epochs):
        total_error = 0
        for i in range(len(inputs)):
            summation_1 = summation_unit([1] + inputs[i], weights[0])
            summation_2 = summation_unit([1] + inputs[i], weights[1])
            prediction_1 = step_function(summation_1)
            prediction_2 = step_function(summation_2)
            error_1 = comparator_unit(prediction_1, outputs[i][0])
            error_2 = comparator_unit(prediction_2, outputs[i][1])
            total_error += error_1 ** 2 + error_2 ** 2
            for j in range(len(weights[0])):
                weights[0][j] += lr * error_1 * ([1] + inputs[i])[j]
                weights[1][j] += lr * error_2 * ([1] + inputs[i])[j]
        if total_error <= 0.002:
            break
    return weights, epoch

outputs_2 = [[1, 0], [1, 0], [1, 0], [0, 1]]  # Example for AND Gate logic
weights, epochs = train_perceptron_2_outputs(inputs, outputs_2)

#A11
from sklearn.neural_network import MLPClassifier

# AND Gate using MLPClassifier
mlp_and = MLPClassifier(hidden_layer_sizes=(), activation='logistic', solver='sgd', max_iter=1000)
mlp_and.fit(inputs, outputs)
and_predictions = mlp_and.predict(inputs)

# XOR Gate using MLPClassifier
mlp_xor = MLPClassifier(hidden_layer_sizes=(), activation='logistic', solver='sgd', max_iter=1000)
mlp_xor.fit(inputs_xor, outputs_xor)
xor_predictions = mlp_xor.predict(inputs_xor)

#A12
mlp_project = MLPClassifier(hidden_layer_sizes=(5,), activation='logistic', solver='sgd', max_iter=1000)
mlp_project.fit(customer_data, high_value_labels)
project_predictions = mlp_project.predict(customer_data)
