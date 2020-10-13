import multiprocessing as mp
import numpy as np

##############
# Evaluation #
##############

class Evaluator:
  def __init__(self, num_workers, worker_size):
    self.num_workers = num_workers
    self.worker_size = worker_size
    self.pipes = []
    self.procs = []
    for rank in range(self.num_workers):
      parent_pipe, child_pipe = mp.Pipe()
      proc = mp.Process(
        target=self._worker,
        name=f"Worker{rank}",
        args=(rank, child_pipe, parent_pipe),
      )
      proc.daemon = True
      proc.start()
      child_pipe.close()
      self.pipes.append(parent_pipe)
      self.procs.append(proc)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    for pipe in self.pipes:
      pipe.send(("close", None))
    _, success = zip(*[pipe.recv() for pipe in self.pipes])
    assert all(success)

  def _build_agent(self):
    raise NotImplementedError

  def _build_env(self):
    raise NotImplementedError

  def _evaluate_once(self, agent, env):
    raise NotImplementedError

  def _evaluate(self, agent, env, num_evals):
    return np.mean([self._evaluate_once(agent, env) for _ in range(num_evals)])

  def _worker(self, rank, pipe, parent_pipe):
    parent_pipe.close()
    agents = [self._build_agent() for _ in range(self.worker_size)]
    envs = [self._build_env() for _ in range(self.worker_size)]
    while True:
      command, data = pipe.recv()
      if command == "evaluate":
        solutions, seeds, num_evals = data
        fitness = []
        for agent, env, solution, seed in zip(agents, envs, solutions, seeds):
          agent.set_params(solution)
          env.seed(int(seed))
          fitness.append(self._evaluate(agent, env, num_evals))
        fitness = np.array(fitness)
        pipe.send((fitness, True))
      elif command == "close":
        pipe.send((None, True))
        return True
      else:
        raise NotImplementedError

  def __len__(self):
    return self.num_workers * self.worker_size

  def __call__(self, solutions, seeds, num_evals):
    num_solutions = len(solutions)
    num_workers = int(np.ceil(num_solutions / self.worker_size))
    pipes = self.pipes[:num_workers]
    for i, pipe in enumerate(pipes):
      start = i * self.worker_size
      end = min(start + self.worker_size, num_solutions)
      pipe.send(("evaluate", (solutions[start:end], seeds[start:end], num_evals)))
    fitness, success = zip(*[pipe.recv() for pipe in pipes])
    assert all(success)
    return np.concatenate(fitness)

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
# Evolution Strategies #
########################

class OpenES:
  def __init__(self, optim, σ=0.1):
    self.optim = optim
    self.σ = σ
    self.ε = None

  def sample(self, popsize):
    assert popsize % 2 == 0
    θ = np.array(self.optim.θ)
    ε_split = np.random.randn(popsize // 2, len(θ))
    self.ε = np.concatenate([ε_split, -ε_split], axis=0)
    return θ + self.σ * self.ε

  def step(self, fitness):
    popsize = fitness.shape[0]
    rank = np.empty_like(fitness, dtype=np.long)
    rank[np.argsort(fitness)] = np.arange(popsize)
    rank = rank.astype(fitness.dtype) / (popsize - 1) - 0.5
    rank = (rank - np.mean(rank)) / (np.std(rank) + 1e-8)
    grad = 1 / (popsize * self.σ) * (self.ε.T @ rank)
    self.optim.update(grad)
