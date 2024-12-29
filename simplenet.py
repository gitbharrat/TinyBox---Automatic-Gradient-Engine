from tinybox import Scalar
import random


"""
Defining the basic component of Neural Network
- Linear Equations
- Activation Functions

Since the building block are Scalar blocks, the backpropagation will be taken care of autonomously and internally.
"""


class Neuron:

    def __init__(self, fanin):
        self.W = [Scalar(random.uniform(-1, 1)) for _ in range(fanin)]
        self.b = Scalar(random.uniform(-1, 1))

    def __call__(self, x):
        result = sum([wi * xi for wi, xi in zip(self.W, x)]) + self.b
        out = result.tanh()
        return out

    def parameters(self):
        return self.W + [self.b]


# Defining a Layer of multiple neurons
class Layer:

    def __init__(self, fanin, fanout):
        self.neurons = [Neuron(fanin) for _ in range(fanout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


# Defining the MLP architecture
class MLP:

    def __init__(self, input_dim, neurons):
        neurons = [input_dim] + neurons
        self.layers = [
            Layer(neurons[i], neurons[i + 1]) for i in range(len(neurons) - 1)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def mse(self, ytrue, ypred):
        return sum([(yt - yp) ** 2 for yt, yp in zip(ytrue, ypred)])

    def predict(self, X):
        return [self(x) for x in X]

    def sgd(self, X, y, iter=100, lr=0.001):
        for _ in range(iter):
            ypred = self.predict(X)
            j = self.mse(y, ypred)

            for p in self.parameters():
                p.grad = 0

            j.backward()

            for p in self.parameters():
                p.data -= lr * p.grad
            print(j)
