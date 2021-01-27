import multiprocessing as mp
import numpy as np

#############
# Utilities #
#############

class Seeder:
  def __init__(self, seed=0):
    self.seed = seed
    self._rs = np.random.RandomState(seed=seed)

  def __call__(self, size):
    seeds = self._rs.randint(2 ** 31 - 1, size=size, dtype=int)
    return seeds.tolist()

class Evaluator:
  def __init__(self, num_workers=None):
    if num_workers is None:
      num_workers = mp.cpu_count()
    self.num_workers = num_workers

  def evaluate(self, solution, seed):
    raise NotImplementedError

  def __call__(self, solutions, seeds):
    results = []
    with mp.Pool(self.num_workers) as pool:
      for solution, seed in zip(solutions, seeds):
        func = self.evaluate
        args = (solution, seed)
        results.append(pool.apply_async(func, args=args))
      fitness = [r.get() for r in results]
    return np.array(fitness)

class DecayingFloat:
  def __init__(self, init, decay, limit):
    assert 0 < decay <= 1
    assert init >= limit
    self.init = init
    self.decay = decay
    self.limit = limit
    self.value = init

  def __float__(self): return float(self.value)
  def __repr__(self): return f"DecayingFloat({self.value})"
  def __add__(self, other): return float(self) + other
  def __radd__(self, other): return other + float(self)
  def __sub__(self, other): return float(self) - other
  def __rsub__(self, other): return other - float(self)
  def __mul__(self, other): return float(self) * other
  def __rmul__(self, other): return other * float(self)
  def __truediv__(self, other): return float(self) / other
  def __rtruediv__(self, other): return other / float(self)
  def __pow__(self, other): return float(self) ** other
  def __rpow__(self, other): return other ** float(self)

  def update(self):
    self.value = self.value * self.decay
    self.value = max(self.value, self.limit)
    return float(self.value)

##############
# Optimizers #
##############

class Optimizer:
  def __init__(self, theta):
    self.theta = theta
    self.t = 0

  def update(self, grad):
    self.t += 1
    self.theta += self._step(grad)
    return np.array(self.theta)

  def _step(self, grad):
    raise NotImplementedError

class SGD(Optimizer):
  def __init__(self, theta, alpha, beta=0.9):
    super().__init__(theta)
    self.alpha = alpha
    self.beta = beta
    self.v = np.zeros_like(theta)

  def _step(self, grad):
    self.v = self.beta * self.v + (1 - self.beta) * grad
    return -self.alpha * self.v

class Adam(Optimizer):
  def __init__(self, theta, alpha, beta1=0.9, beta2=0.999):
    super().__init__(theta)
    self.alpha = alpha
    self.beta1 = beta1
    self.beta2 = beta2
    self.m = np.zeros_like(theta)
    self.v = np.zeros_like(theta)

  def _step(self, grad):
    self.m = self.beta1 * self.m + (1 - self.beta1) * grad
    self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
    m_corr = 1 - self.beta1 ** self.t
    v_corr = np.sqrt(1 - self.beta2 ** self.t)
    alpha = self.alpha * v_corr / m_corr
    return -alpha * self.m / (np.sqrt(self.v) + 1e-8)

########################
# Evolution strategies #
########################

class ES:
  def __init__(self, optim, sigma):
    self.optim = optim
    self.mu = np.array(optim.theta)
    self.sigma = sigma
    self.epsilon = None

  def sample(self, popsize):
    assert popsize % 2 == 0
    eps_split = np.random.randn(popsize // 2, len(self.mu))
    self.epsilon = np.concatenate([eps_split, -eps_split], axis=0)
    return self.mu + self.sigma * self.epsilon

  def update(self, fitness):
    rank = np.empty_like(fitness, dtype=np.long)
    rank[np.argsort(fitness)] = np.arange(len(fitness))
    fitness = rank.astype(fitness.dtype) / (len(fitness) - 1) - 0.5
    fitness = (fitness - np.mean(fitness)) / (np.std(fitness) + 1e-8)
    grad = 1 / (len(fitness) * self.sigma) * (self.epsilon.T @ fitness)
    self.mu = self.optim.update(-grad)
