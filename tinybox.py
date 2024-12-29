import math


class Scalar:

    def __init__(self, data, children=(), op="", label=""):
        self.data = data
        self.label = label
        self.grad = 0.0  # Initialize Gradients with 0
        self._prev = children
        self._op = op
        self._backward = lambda: None  # Initial Backpropagation set as Non

    def __repr__(self):
        return f"Scalar(data = {self.data})"

    """
    Defining granular mathematic operations, Key idea is to:
    - Mathematical operations self backpropagate autonomously.
    - Hence I'll define some base operations, adding complexity on aggregation.
    """

    # Basic Operations

    def __add__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data + other.data, children=(self, other), op="+", label="+")

        def _backward():
            # Defining backpropagation rule for addition
            self.grad += 1.0 * out.grad  # Gradients are accumulated
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        # Treating subtraction as application of addition
        return self + -(other)

    def __rsub__(self, other):
        return -self + other

    def __pow__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data**other.data, children=(self,), op="^", label="^")

        def _backward():
            self.grad += other.data * (self.data ** (other.data - 1)) * out.grad
            if self.data > 0:
                other.grad += (self.data**other.data) * math.log(self.data) * out.grad
            else:
                other.grad = 0

        out._backward = _backward
        return out

    def __log__(self):
        out = Scalar(math.log(self.data), children=(self,), op="log", label="log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward

    def exp(self):
        out = Scalar(math.exp(self.data), children=(self,), op="exp", label="exp")

        def _backward():
            self.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data * other.data, children=(self, other), op="*", label="*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** (-1)

    def __neg__(self):
        return self * (-1.0)

    # Autonomously Implementing Backward Propagation

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

    # Additional Activation Functions

    """
    Including only Scalar based activations(excluding Softmax)
    """

    def sigmoid(self):
        x = self.data
        sigma = 1.0 / (1 + math.exp(-x))
        out = Scalar(sigma, children=(self,), op="sigmoid", label="sigmoid")

        def _backward():
            self.grad += ((1.0 - sigma) * sigma) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
        out = Scalar(t, children=(self,), op="tanh", label="tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def relu(self, leaky=0.0):
        x = self.data if self.data >= 0 else leaky * self.data
        out = Scalar(x, children=(self,), op="relu", label="relu")

        def _backward():
            grad = 1 if self.data >= 0 else leaky
            self.grad += grad * out.grad

        out._backward = _backward
        return out
