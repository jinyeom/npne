import logging
import sys
from time import sleep
import numpy as np
import npnn
import npes

class PrimMaze:
  def __init__(self, N=9, T=500):
    self.N = N
    self.T = T
    self.t = 0
    self.s = 0
    self.seed()

  def _generate(self):
    grid = np.ones((self.N, self.N), dtype=np.bool)
    frontiers = [((0, 0), (0, 0))]
    while frontiers:
      self.rng.shuffle(frontiers)
      cell = frontiers.pop()
      x, y = cell[1]
      if grid[x, y] == 1:
        grid[cell[0]] = 0
        grid[x, y] = 0
        if x >= 2 and grid[x - 2, y] == 1:
          frontiers.append(((x - 1, y), (x - 2, y)))
        if x < self.N - 2 and grid[x + 2, y] == 1:
          frontiers.append(((x + 1, y), (x + 2, y)))
        if y >= 2 and grid[x, y - 2] == 1:
          frontiers.append(((x, y - 1), (x, y - 2)))
        if y < self.N - 2 and grid[x, y + 2] == 1:
          frontiers.append(((x, y + 1), (x, y + 2)))
    grid = np.pad(grid, (2, 2), "constant", constant_values=1)
    return grid

  def _rand_pos(self):
    i, j = self.rng.randint(self.N + 4, size=2)
    while self.G[i, j]:
      i, j = self.rng.randint(self.N + 4, size=2)
    return i, j

  def seed(self, seed=None):
    self.rng = np.random.RandomState(seed)
  
  def reset(self):
    self.G = self._generate()
    self.t = 0
    self.s = 0
    self.goal = self._rand_pos()
    self.agent = self._rand_pos()
    i, j = self.agent
    obs = self.G[i-2:i+3, j-2:j+3]
    obs = np.ravel(obs).astype(np.float)
    return obs

  def render(self):
    i, j = self.agent
    G = [[("#" if w else ".") for w in r] for r in self.G]
    G[self.goal[0]][self.goal[1]] = "$"
    G[self.agent[0]][self.agent[1]] = "@"
    G_obs = [r[j-2:j+3] for r in G[i-2:i+3]]
    for r in range(5):
      G[r] += ["  "] + G_obs[r]
    G[6] += ["  ", f"t: {self.t}"]
    G[7] += ["  ", f"@: ({i},{j})"]
    G[8] += ["  ", f"$: {self.s:.1f}"]
    clear = "\x1B[2J\x1B[3J\x1B[H"
    view = "\n".join(["".join(l) for l in G])
    print(clear + view, flush=True)

  def step(self, action):
    reward = 0
    i, j = self.agent
    if   action == 0: i -= 1 # ^
    elif action == 1: i += 1 # v
    elif action == 2: j -= 1 # <
    elif action == 3: j += 1 # >
    else: raise ValueError
    if self.G[i, j]: reward = -0.1
    else: self.agent = (i, j)
    i, j = self.agent
    obs = self.G[i-2:i+3, j-2:j+3]
    obs = np.ravel(obs).astype(np.float)
    if self.agent == self.goal:
      self.agent = self._rand_pos()
      reward = 10
    self.t += 1
    self.s += reward
    done = self.t >= self.T
    return obs, reward, done

# if __name__ == "__main__":
#   from time import sleep
#   maze = PrimMaze()
#   maze.reset()
#   done = False
#   while not done:
#     maze.render()
#     action = np.random.randint(4)
#     _, _, done = maze.step(action)
#     sleep(0.1)
  
def build_agent():
  return npnn.Stack(
    npnn.HebbianRNN(30, 32, 0.1),
    npnn.Linear(32, 4),
  )

class PrimMazeEvaluator(npes.Evaluator):
  def _build_agent(self):
    return build_agent()

  def _build_env(self):
    return PrimMaze()

  def _evaluate_once(self, agent, env):
    agent.reset()
    obs = env.reset()
    prev_action = np.zeros(4)
    prev_reward = np.zeros(1)
    R = 0
    done = False
    while not done:
      x = np.concatenate([obs, prev_action, prev_reward])
      action = np.argmax(agent(x))
      obs, reward, done = env.step(action)
      prev_action = np.zeros(4)
      prev_action[action] = 1
      prev_reward[0] = reward
      R += reward
    return R

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.DEBUG)
fh = logging.FileHandler("progress.log")
fh.setLevel(logging.DEBUG)
logger.addHandler(ch)
logger.addHandler(fh)

np.random.seed(0)

num_params = len(build_agent())
sigma_init = 0.1
alpha_init = 0.1
num_workers = 10
agents_per_worker = 50
num_gens = 5000

mu = None
best_ind = None
max_score = -np.inf

es = npes.OpenES(num_params, sigma_init=sigma_init, alpha_init=alpha_init)
with PrimMazeEvaluator(num_workers, agents_per_worker) as evaluator:
  popsize = len(evaluator)

  for gen in range(num_gens):
    solutions = es.sample(popsize)
    seeds = np.random.randint(2 ** 31 - 1, size=popsize, dtype=int)
    fitness = evaluator(solutions, seeds, 3)
    es.step(fitness)

    gen_min_score = np.min(fitness)
    gen_max_score = np.max(fitness)
    gen_mean_score = np.mean(fitness)
    gen_std_score = np.std(fitness)
    logger.info(f"gen={gen}, min={gen_min_score:.3}, max={gen_max_score:.3}, "
                f"mean={gen_mean_score:.3}, std={gen_std_score:.3}")

    mu = np.array(es.mu)
    gen_best_ind = solutions[np.argmax(fitness)]
    if gen_max_score > max_score:
      logger.info(f"max_score {max_score:.3} -> {gen_max_score:.3}")
      best_ind = gen_best_ind
      max_score = gen_max_score
      np.save("best_agent.npy", best_ind)

    if gen % 50 == 0:
      env = PrimMaze()
      obs = env.reset()
      agent = build_agent()
      agent.set_params(gen_best_ind)
      agent.reset()
      prev_action = np.zeros(4)
      prev_reward = np.zeros(1)
      R = 0
      done = False
      while not done:
        env.render()
        x = np.concatenate([obs, prev_action, prev_reward])
        action = np.argmax(agent(x))
        obs, reward, done = env.step(action)
        prev_action = np.zeros(4)
        prev_action[action] = 1
        prev_reward[0] = reward
        R += reward
        sleep(0.1)
