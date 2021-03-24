from dataclasses import dataclass
import numpy as np
import numba

#############
# Utilities #
#############

@numba.njit
def cast_rays(pos, ori, n, fov, limit, prec, grid):
  colls = []
  dists = []
  start = ori - fov / 2
  stop = start + fov
  for a in np.linspace(start, stop, n):
    coll = None
    dist = -1
    for t in np.arange(0, limit, prec):
      cy = int(np.floor(pos[0] + t * np.sin(a)))
      cx = int(np.floor(pos[1] + t * np.cos(a)))
      if grid[cy, cx]:
        coll = (cy, cx)
        dist = t / limit
        break
    colls.append(coll)
    dists.append(dist)
  return colls, dists

##################
# Map generation #
##################

def four_rooms():
  return np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  ], dtype=np.bool)

def prim_maze(height, width, seed=None):
  np_random = np.random.RandomState(seed=seed)
  grid = np.ones((height, width), dtype=np.bool)
  frontiers = [((0, 0), (0, 0))]
  while frontiers:
    np_random.shuffle(frontiers)
    cell = frontiers.pop()
    r, c = cell[1]
    if grid[r, c] == 1:
      grid[cell[0]] = 0
      grid[r, c] = 0
      if r >= 2 and grid[r - 2, c] == 1:
        frontiers.append(((r - 1, c), (r - 2, c)))
      if r < height - 2 and grid[r + 2, c] == 1:
        frontiers.append(((r + 1, c), (r + 2, c)))
      if c >= 2 and grid[r, c - 2] == 1:
        frontiers.append(((r, c - 1), (r, c - 2)))
      if c < width - 2 and grid[r, c + 2] == 1:
        frontiers.append(((r, c + 1), (r, c + 2)))
  grid = np.pad(grid, (1, 1), constant_values=1)
  return grid

#############
# NumPy lab #
#############

@dataclass
class Robot:
  rays: int = 32
  fov: float = np.pi / 3
  limit: float = 10
  prec: float = 0.01
  vel: float = 0.2
  rot: float = np.pi / 32
  cost: float = 0.01
  dmg: float = 0.1

@dataclass
class Event:
  r: int
  c: int
  sym: str
  callback: callable
  reward: float

class NumPyLab:
  def __init__(self, robot):
    self._robot = robot
    self._seed = None
    self._np_random = None
    self._lab = None
    self._time_limit = None
    self._events = None
    self._agent_pos = None
    self._agent_ori = None
    self._curr_time = None
    self._curr_colls = None
    self._curr_dists = None
    self._curr_score = None
    self._triggered = None
    self._collision = None
    self.seed()

  def _build_lab(self):
    raise NotImplementedError

  def _build_lab_check(self):
    self._build_lab()
    assert self._lab is not None
    assert self._time_limit is not None
    assert self._events is not None

  def _rand_pos(self):
    height, width = self._lab.shape
    r = self._np_random.randint(height)
    c = self._np_random.randint(width)
    while self._lab[r, c]:
      r = self._np_random.randint(height)
      c = self._np_random.randint(width)
    y = r + self._np_random.rand()
    x = c + self._np_random.rand()
    return np.array([y, x])

  def _rand_ori(self):
    return 2 * np.pi * self._np_random.rand()

  def seed(self, seed=None):
    self._seed = seed
    self._np_random = np.random.RandomState(seed=seed)

  def observation(self):
    colls, dists = cast_rays(
      self._agent_pos,
      self._agent_ori,
      self._robot.rays,
      self._robot.fov,
      self._robot.limit,
      self._robot.prec,
      self._lab,
    )
    self._curr_colls = colls
    self._curr_dists = dists
    dists = np.array(dists, dtype=np.float32)
    return dists

  def reward(self, action):
    energy = np.sum(np.abs(action))
    reward = -self._robot.cost * energy
    if self._collision:
      reward -= self._robot.dmg
    r, c = np.floor(self._agent_pos)
    for event in self._triggered:
      reward += event.reward
    return reward

  def done(self):
    return self._curr_time >= self._time_limit

  def _render_hud(self):
    y, x = self._agent_pos
    ori = self._agent_ori
    return (
      f"T:{self._curr_time} "
      f"@:({y:.3f},{x:.3f}),{ori:.3f} "
      f"$:{self._curr_score:.3f} "
    )

  def _render_global_view(self):
    agent_y, agent_x = self._agent_pos
    agent_r, agent_c = int(agent_y), int(agent_x)
    grid = [[("#" if c else ".") for c in r] for r in self._lab]
    for coll in self._curr_colls:
      if coll is not None:
        coll_r, coll_c = coll
        grid[coll_r][coll_c] = "\x1B[32m#\x1B[0m"
    for event in self._events:
      grid[event.r][event.c] = event.sym[0]
    if np.pi / 4 <= self._agent_ori < 3 * np.pi / 4:
      grid[agent_r][agent_c] = "v"
    elif 3 * np.pi / 4 <= self._agent_ori < 5 * np.pi / 4:
      grid[agent_r][agent_c] = "<"
    elif 5 * np.pi / 4 <= self._agent_ori < 7 * np.pi / 4:
      grid[agent_r][agent_c] = "^"
    else:
      grid[agent_r][agent_c] = ">"
    grid = "\n".join(["".join(line) for line in grid])
    return grid

  def _render_agent_view(self):
    width = self._robot.rays
    height = self._robot.rays // 4
    fov = self._robot.fov
    ori = self._agent_ori
    grid = [[" "] * width for _ in range(height)]
    for c in range(width):
      if self._curr_colls[c] is None:
        continue
      a = ori - fov / 2 + c * fov / width
      t = self._curr_dists[c] * self._robot.limit
      col = height / (t * np.cos(a - ori))
      col = np.clip(col, 0, height)
      start = int(height / 2 - col / 2)
      end = int(start + col)
      for r in range(start, end):
        if   0 <= t < 1: grid[r][c] = "@"
        elif 1 <= t < 2: grid[r][c] = "#"
        elif 2 <= t < 3: grid[r][c] = "+"
        elif 3 <= t < 4: grid[r][c] = "-"
        elif 4 <= t < 5: grid[r][c] = ":"
        else:            grid[r][c] = "."
    grid = "\n".join(["".join(r) for r in grid])
    return grid

  def render(self):
    rendered = "\x1B[2J\x1B[3J\x1B[H"
    rendered += "\n\n".join([
      self._render_hud(),
      self._render_agent_view(),
      self._render_global_view(),
    ])
    print(rendered, flush=True)

  def reset(self):
    self._build_lab_check()
    self._agent_pos = self._rand_pos()
    self._agent_ori = self._rand_ori()
    self._curr_time = 0
    self._curr_score = 0
    self._collision = False
    return self.observation()

  def _turn_agent(self, angle):
    self._agent_ori += angle * self._robot.rot
    if self._agent_ori > 2 * np.pi:
      self._agent_ori -= 2 * np.pi
    if self._agent_ori < 0:
      self._agent_ori += 2 * np.pi

  def _move_agent(self, dist):
    dy = np.sin(self._agent_ori)
    dx = np.cos(self._agent_ori)
    pos_dy = dy * dist * self._robot.vel
    pos_dx = dx * dist * self._robot.vel
    self._agent_pos += (pos_dy, pos_dx)
    r, c = np.floor(self._agent_pos)
    if self._lab[int(r), int(c)]:
      self._collision = True
      self._agent_pos -= (pos_dy, pos_dx)
    else:
      self._collision = False

  def _trigger_events(self):
    self._triggered = []
    r, c = np.floor(self._agent_pos)
    for event in self._events:
      if r == event.r and c == event.c:
        self._triggered.append(event)
        event.callback()

  def step(self, action):
    angle = np.clip(action[0], -1, 1)
    dist = np.clip(action[1], -1, 1)

    self._turn_agent(angle)
    self._move_agent(dist)
    self._trigger_events()

    obs = self.observation()
    reward = self.reward(action)
    done = self.done()

    self._curr_score += reward
    self._curr_time += 1

    return obs, reward, done

if __name__ == "__main__":
  class TestMaze(NumPyLab):
    def _build_lab(self):
      self._time_limit = 500
      self._lab = prim_maze(10, 10, seed=self._seed)
      # self._lab = four_rooms()

      pos = self._rand_pos()
      r, c = np.floor(pos).astype(np.int)
      event = Event(r, c, "$", self._teleport_agent, 10.0)
      self._events = [event]

    def _teleport_agent(self):
      self._agent_pos = self._rand_pos()

  from time import sleep
  robot = Robot(rays=64)
  env = TestMaze(robot)
  obs = env.reset()
  while True:
    env.render()
    action = np.random.randn(2)
    # action = np.array([0, -0.1])
    obs, reward, done = env.step(action)
    sleep(1e-1)
