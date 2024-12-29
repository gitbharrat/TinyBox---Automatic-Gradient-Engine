# TinyBox: Scalar - Based Autograd Engine

Tinybox is a simple autograd engine that lets you build neural networks from scratch. It includes scalars, neurons, layers and MLP's along with support for custom activation functions and loss calculations.

What this module is already loaded with?
- Enables forward and backward propagation.
- Includes Simple activation functions like sigmoid, tanh, ReLU
- Neural Network Components
- Includes Optimization algorithm - Stochastic Gradient Descent

### Quick Start

```python
from tinybox import Scalar
from simplenet import MLP
```

#### Training a Simple MLP
```python
# Inputs (X) and Outputs (y)
X = [[Scalar(1.0), Scalar(2.0)], [Scalar(-1.0), Scalar(-2.0)]]
y = [Scalar(1.0), Scalar(-1.0)]

# Create an MLP with 2 inputs, 1 hidden layer (2 neurons), and 1 output
model = MLP(input_dim=2, neurons=[2, 1])

# Train with SGD
model.sgd(X, y, iter=50, lr=0.01)

# Predictions
predictions = model.predict(X)
print([p.data for p in predictions])
```

### Custom Functions

#### Adding Custom Activation

```python
def leaky_relu(self, alpha=0.01):
    value = self.data if self.data > 0 else alpha * self.data
    out = Scalar(value, children=(self,), op="leaky_relu", label="leaky_relu")

    def _backward():
        grad = 1 if self.data > 0 else alpha
        self.grad += grad * out.grad

    out._backward = _backward
    return out

# Add to Scalar class
Scalar.leaky_relu = leaky_relu

# Use in a Neuron
class Neuron:
    def __call__(self, x):
        result = sum(w * xi for w, xi in zip(self.W, x)) + self.b
        return result.leaky_relu(alpha=0.01)
```

#### Adding Custom Loss
```python
def mae(y_true, y_pred):
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred))

# Example usage
X = [[Scalar(1.0), Scalar(2.0)], [Scalar(-1.0), Scalar(-2.0)]]
y = [Scalar(1.0), Scalar(-1.0)]

model = MLP(input_dim=2, neurons=[2, 1])
for _ in range(50):
    y_pred = model.predict(X)
    loss = mae(y, y_pred)

    for p in model.parameters():
        p.grad = 0

    loss.backward()

    for p in model.parameters():
        p.data -= 0.01 * p.grad

    print(loss.data)
```

### Getting Started
- Clone this repo.
- Play with `playground.ipynb` notebook to understand how the library works.

---
### License

MIT License.