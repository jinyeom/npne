import numpy as np

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
    obs = np.ravel(obs).astype(np.float32)
    return obs

  def render(self):
    i, j = self.agent
    G = [[("#" if w else ".") for w in r] for r in self.G]
    G[self.goal[0]][self.goal[1]] = "$"
    G[self.agent[0]][self.agent[1]] = "@"
    G_obs = [r[j-2:j+3] for r in G[i-2:i+3]]
    s1 = f"\x1B[2J\x1B[Ht:{self.t} @:({i},{j}) $:{self.s:.1f}\n"
    s2 = "\n".join(["".join(l) for l in G])
    s3 = "\n".join(["".join(l) for l in G_obs])
    s = "\n\n".join([s1, s2, s3])
    print(s, flush=True)

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
    obs = np.ravel(obs).astype(np.float32)
    if self.agent == self.goal:
      self.agent = self._rand_pos()
      reward = 10
    self.t += 1
    self.s += reward
    done = self.t >= self.T
    return obs, reward, done

if __name__ == "__main__":
  from time import sleep
  maze = PrimMaze()
  maze.reset()
  done = False
  while not done:
    maze.render()
    action = np.random.randint(4)
    _, _, done = maze.step(action)
    sleep(0.1)