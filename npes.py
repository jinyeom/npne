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
    async_results = []
    with mp.Pool(self.num_workers) as pool:
      for solution, seed in zip(solutions, seeds):
        func = self.evaluate
        args = (solution, seed)
        result = pool.apply_async(func, args=args)
        async_results.append(result)
      fitness = [r.get() for r in async_results]
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
  def __init__(self, θ):
    self.θ = θ
    self.t = 0

  def update(self, grad):
    self.t += 1
    self.θ += self._step(grad)
    return np.array(self.θ)

  def _step(self, grad):
    raise NotImplementedError

class SGD(Optimizer):
  def __init__(self, θ, α, β=0.9):
    super().__init__(θ)
    self.α = α
    self.β = β
    self.v = np.zeros_like(θ)

  def _step(self, grad):
    self.v = self.β * self.v + (1 - self.β) * grad
    return -self.α * self.v

class Adam(Optimizer):
  def __init__(self, θ, α, β1=0.9, β2=0.999):
    super().__init__(θ)
    self.α = α
    self.β1 = β1
    self.β2 = β2
    self.m = np.zeros_like(θ)
    self.v = np.zeros_like(θ)

  def _step(self, grad):
    self.m = self.β1 * self.m + (1 - self.β1) * grad
    self.v = self.β2 * self.v + (1 - self.β2) * grad ** 2
    m_corr = 1 - self.β1 ** self.t
    v_corr = np.sqrt(1 - self.β2 ** self.t)
    α = self.α * v_corr / m_corr
    return -α * self.m / (np.sqrt(self.v) + 1e-8)

########################
# Evolution strategies #
########################

class ES:
  def __init__(self, optim, σ):
    self.optim = optim
    self.μ = np.array(optim.θ)
    self.σ = σ
    self.ε = None

  def sample(self, popsize):
    assert popsize % 2 == 0
    ε_split = np.random.randn(popsize // 2, len(self.μ))
    self.ε = np.concatenate([ε_split, -ε_split], axis=0)
    return self.μ + self.σ * self.ε

  def update(self, F):
    rank = np.empty_like(F, dtype=np.long)
    rank[np.argsort(F)] = np.arange(len(F))
    F = rank.astype(F.dtype) / (len(F) - 1) - 0.5
    F = (F - np.mean(F)) / (np.std(F) + 1e-8)
    grad = 1 / (len(F) * self.σ) * (self.ε.T @ F)
    self.μ = self.optim.update(-grad)
