import multiprocessing as mp
import numpy as np

class Evaluator:
  def __init__(self, num_workers, agents_per_worker):
    self.num_workers = num_workers
    self.agents_per_worker = agents_per_worker
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
    agents = [self._build_agent() for _ in range(self.agents_per_worker)]
    envs = [self._build_env() for _ in range(self.agents_per_worker)]
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
    return self.num_workers * self.agents_per_worker

  def __call__(self, solutions, seeds, num_evals):
    num_solutions = len(solutions)
    num_workers = int(np.ceil(num_solutions / self.agents_per_worker))
    pipes = self.pipes[:num_workers]
    for i, pipe in enumerate(pipes):
      start = i * self.agents_per_worker
      end = min(start + self.agents_per_worker, num_solutions)
      pipe.send(("evaluate", (solutions[start:end], seeds[start:end], num_evals)))
    fitness, success = zip(*[pipe.recv() for pipe in pipes])
    assert all(success)
    return np.concatenate(fitness)

class OpenES:
  def __init__(
    self,
    num_params,
    sigma_init=0.1,
    sigma_decay=0.999,
    sigma_min=0.01,
    alpha_init=0.1,
    alpha_decay=0.995,
    alpha_min=0.001,
    gamma=0.9,
  ):
    self.num_params = num_params
    self.sigma_init = sigma_init
    self.sigma_decay = sigma_decay
    self.sigma_min = sigma_min
    self.alpha_init = alpha_init
    self.alpha_decay = alpha_decay
    self.alpha_min = alpha_min
    self.gamma = gamma
    self.mu = np.zeros(num_params)
    self.sigma = self.sigma_init
    self.alpha = self.alpha_init
    self.v = np.zeros_like(self.mu)

  def sample(self, popsize):
    assert popsize % 2 == 0
    eps_split = np.random.randn(popsize // 2, self.num_params)
    self.eps = np.concatenate([eps_split, -eps_split], axis=0)
    theta = self.mu + self.sigma * self.eps
    return theta

  def step(self, fitness):
    popsize = fitness.shape[0]
    rank = np.empty_like(fitness, dtype=np.long)
    rank[np.argsort(fitness)] = np.arange(popsize)
    rank = rank.astype(fitness.dtype) / (popsize - 1) - 0.5
    rank = (rank - np.mean(rank)) / np.std(rank)
    delta = 1 / (popsize * self.sigma) * (self.eps.T @ rank)
    self.v = self.gamma * self.v + (1 - self.gamma) * delta
    self.mu += self.alpha * self.v
    self.sigma = max(self.sigma * self.sigma_decay, self.sigma_min)
    self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)
