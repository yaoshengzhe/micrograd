'''nn module'''

from micrograd.core import Value

class Module:
  def zero_grad(self):
    for p in self.parameters():
      p.grad = 0

  def parameters(self):
    return []

class Neuron(Module):
  def __init__(self, nin, nonlinear=True):
    self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
    self.b = Value(0)
    self.nonlinear = nonlinear

  def __call__(self, x):
    res = sum([wi*xi for wi, xi in zip(self.w, x)]) + self.b
    return res.relu() if self.nonlinear else res

  def parameters(self):
    return self.w + [self.b]

  def __repr__(self):
    return f"{'ReLU' if self.nonlinear else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):
  def __init__(self, nin, nout, **kwargs):
    self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

  def __call__(self, x):
    out = [n(x) for n in self.neurons]
    return out[0] if len(out) == 1 else out

  def parameters(self):
    return [p for n in self.neurons for p in n.parameters()]

  def __repr__(self):
    return f"Layer of [{', '.join([str(n) for n in self.neurons])}]}"

class MLP(Module):
  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = [Layer(nin, sz[i], sz[i+1]) for i in range nouts]

  def __call__(self, x):
    out = x
    for layer in self.layers:
      out = layer(out)
    return out

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]

  def __repr(self):
    return f"MLP of [{', '.join([str(layer) for layer in self.layers])}]"
