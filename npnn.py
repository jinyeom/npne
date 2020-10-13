import numpy as np
import numba

##############
# Functional #
##############

@numba.njit
def linear(x, W, b):
  return x @ W + b

@numba.njit
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

@numba.njit
def tanh(x):
  return np.tanh(x)

@numba.njit
def relu(x):
  return x * (x > 0)

@numba.njit
def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / np.sum(e_x, axis=0)

##############
# Containers #
##############

class Module:
  def __init__(self):
    self._children = []
    self._params = []

  @property
  def params(self):
    return np.concatenate([np.ravel(p) for p in self._params])

  def __len__(self):
    return sum(p.size for p in self._params)

  def register_module(self, name, module):
    assert isinstance(module, Module)
    setattr(self, name, module)
    self._children.append((name, module))
    self._params.extend(module._params)

  def register_param(self, name, shape):
    param = np.empty(shape)
    setattr(self, name, param)
    self._params.append(param)

  def set_params(self, params):
    assert params.size == len(self)
    for dst in self._params:
      src, params = params[:dst.size], params[dst.size:]
      src = np.array(src).reshape(dst.shape)
      np.copyto(dst, src)

  def reset(self):
    return None

  def __call__(self):
    raise NotImplementedError

class _Modules(Module):
  def __init__(self, *modules):
    super().__init__()
    for i, module in enumerate(modules):
      self.register_module(str(i), module)

  def reset(self):
    hiddens = {}
    for name, module in self._children:
      hiddens[name] = module.reset()
    return hiddens

  def __getitem__(self, i):
    return self._children[i][1]

class Stack(_Modules):
  def __call__(self, x):
    for _, module in self._children:
      x = module(x)
    return x

class Group(_Modules):
  def __call__(self, x):
    return np.concatenate([m(x) for m in self._children])

###############
# Activations #
###############

class Tanh(Module):
  def __call__(self, x):
    return tanh(x)

class Sigmoid(Module):
  def __call__(self, x):
    return sigmoid(x)

class ReLU(Module):
  def __call__(self, x):
    return relu(x)

class Softmax(Module):
  def __call__(self, x):
    return softmax(x)

################
# Feed forward #
################

class Linear(Module):
  def __init__(self, N_i, N_o):
    super().__init__()
    self.N_i = N_i
    self.N_o = N_o
    self.register_param("W", (N_i, N_o))
    self.register_param("b", (N_o,))

  def __call__(self, x):
    return linear(x, self.W, self.b)

class Hebbian(Module):
  def __init__(self, N_i, N_h, η):
    super().__init__()
    self.N_i = N_i
    self.N_h = N_h
    self.η = η
    self.register_param("W", (N_i, N_h))
    self.register_param("A", (N_i, N_h))
    self.register_param("b", (N_h,))
    self.reset()

  def reset(self):
    self._Hebb = np.zeros((self.N_i, self.N_h))
    return None

  def __call__(self, x):
    W = self.W + self.A * self._Hebb
    y = tanh(linear(x, W, self.b))
    ΔHebb = self.η * np.outer(x, y)
    self._Hebb = (1 - self.η) * self._Hebb + ΔHebb
    return y

#############
# Recurrent #
#############

class RNN(Module):
  def __init__(self, N_i, N_h):
    super().__init__()
    self.N_i = N_i
    self.N_h = N_h
    self.register_param("W", (N_i + N_h, N_h))
    self.register_param("b", (N_h,))
    self.register_param("h_init", (N_h,))
    self.reset()

  def reset(self):
    self._h = np.array(self.h_init)
    return np.array(self._h)

  def __call__(self, x):
    xh = np.concatenate([x, self._h])
    self._h = tanh(linear(xh, self.W, self.b))
    return np.array(self._h)

class HebbianRNN(Module):
  def __init__(self, N_i, N_h, η):
    super().__init__()
    self.N_i = N_i
    self.N_h = N_h
    self.η = η
    self.register_param("W", (N_i, N_h))
    self.register_param("U", (N_h, N_h))
    self.register_param("A", (N_h, N_h))
    self.register_param("b", (N_h,))
    self.register_param("h_init", (N_h,))
    self.reset()

  def reset(self):
    self._Hebb = np.zeros((self.N_h, self.N_h))
    self._h = np.array(self.h_init)
    return np.array(self._h)

  def __call__(self, x):
    h0 = self._h
    U = self.U + self.A * self._Hebb
    h1 = tanh(linear(x, self.W, self.b) + linear(h0, U, 0))
    ΔHebb = self.η * np.outer(h0, h1)
    self._Hebb = (1 - self.η) * self._Hebb + ΔHebb
    self._h = h1
    return np.array(self._h)

class LSTM(Module):
  def __init__(self, N_i, N_h):
    super().__init__()
    self.N_i = N_i
    self.N_h = N_h
    self.register_param("W", (N_i + N_h, 4 * N_h))
    self.register_param("b", (4 * N_h,))
    self.register_param("c_init", (N_h,))
    self.register_param("h_init", (N_h,))
    self.reset()

  def reset(self):
    self._c = np.array(self.c_init)
    self._h = np.array(self.h_init)
    return np.array(self._h)

  def __call__(self, x):
    xh = np.concatenate([x, self._h])
    fioc = linear(xh, self.W, self.b)
    fioc = np.split(fioc, 4)
    f = sigmoid(fioc[0])
    i = sigmoid(fioc[1])
    o = sigmoid(fioc[2])
    c = np.tanh(fioc[3])
    self._c = f * self._c + i * c
    self._h = o * tanh(self._c)
    return np.array(self._h)
