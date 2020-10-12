# [WIP] Reimplementation of pybullet_env with much clearner code.
# This implementation is also tailored for my own projects.
#
# Until this is being implemented, I will continue to use existing
# implementation of pybullet_env.

from collections import OrderedDict
import inspect
from functools import partial
import numpy as np
import pybullet

class _BulletClient:
  def __init__(self, connection_mode=None):
    if connection_mode is None:
      connection_mode = pybullet.SHARED_MEMORY
    client_id = pybullet.connect(connection_mode)
    if client_id < 0:
      client_id = pybullet.connect(pybullet.DIRECT)
    self._client_id = client_id

  @property
  def client_id(self):
    return self._client_id

  def __del__(self):
    if self._client_id >= 0:
      try:
        pybullet.disconnect(physicsClientId=self._client_id)
        self._client_id = -1
      except pybullet.error:
        pass

  def __getattr__(self, name):
    attr = getattr(pybullet, name)
    if inspect.isbuiltin(attr):
      attr = partial(attr, physicsClientId=self.client_id)
    if name == "disconnect":
      self._client_id = -1
    return attr

class BodyPart:
  def __init__(self, bullet, body_name, body_index, link_index):
    self.bullet = bullet
    self.body_name = body_name
    self.body_index = body_index
    self.link_index = link_index

  @property
  def state(self):
    if self.link_index == -1:
      xyz, abcd = self.bullet.getBasePositionAndOrientation(self.body_index)
      vxvyvz, _ = self.bullet.getBaseVelocity(self.body_index)
    else:
      link_state = self.bullet.getLinkState(self.body_index, self.link_index)
      xyz, abcd, vxvyvz = link_state[0], link_state[1], link_state[6]
    rpy = pybullet.getEulerFromQuaternion(abcd)
    return xyz, rpy, vxvyvz

  @property
  def contact_points(self):
    return self.bullet.getContactPoints(self.body, -1, self.link_index, -1)

  def reset_position(self, position):
    _, orientation = self.bullet.getBasePositionAndOrientation(self.body_index)
    self.bullet.resetBasePositionAndOrientation(self.body, position, orientation)

  def reset_orientation(self, orientation):
    position, _ = self.bullet.getBasePositionAndOrientation(self.body_index)
    self.bullet.resetBasePositionAndOrientation(self.body, position, orientation)

  def reset_velocity(self, linear_velocity=(0, 0, 0), angular_velocity=(0, 0, 0)):
    self.bullet.resetBaseVelocity(self.body, linear_velocity, angular_velocity)

  def reset_pose(self, position, orientation):
    self.bullet.resetBasePositionAndOrientation(self.body, position, orientation)

class Joint:
  def __init__(self, bullet, body_index, joint_name, joint_index):
    self.bullet = bullet
    self.body_index = body_index
    self.joint_name = joint_name
    self.joint_index = joint_index
    self.power_coeff = 0.0

    joint_info = self.bullet.getJointInfo(body_index, joint_index)
    self.lower_limit = joint_info[8]
    self.upper_limit = joint_info[9]

  @property
  def state(self):
    state = self.bullet.getJointState(self.body_index, self.joint_index)
    position, velocity, react_forces, app_torque = state
    del react_forces, app_torque
    center = (self.lower_limit + self.upper_limit) / 2
    joint_range = self.upper_limit - self.lower_limit
    position = 2 * (position - center) / joint_range
    velocity = 0.1 * velocity
    return position, velocity

  @property
  def position(self):
    state = self.bullet.getJointState(self.body_index, self.joint_index)
    return state[0]

  @property
  def velocity(self):
    state = self.bullet.getJointState(self.body_index, self.joint_index)
    return state[1]

  def set_state(self, position, velocity):
    self.bullet.resetJointState(
      bodyIndex=self.body_index,
      jointIndex=self.joint_index,
      targetValue=position,
      targetVelocity=velocity,
    )

  def set_position(self, position):
    self.bullet.setJointMotorControl2(
      bodyIndex=self.body_index,
      jointIndex=self.joint_index,
      controlMode=pybullet.POSITION_CONTROL,
      targetPosition=position,
    )

  def set_velocity(self, velocity):
    self.bullet.setJointMotorControl2(
      bodyIndex=self.body_index,
      jointIndex=self.joint_index,
      controlMode=pybullet.VELOCITY_CONTROL,
      targetVelocity=velocity,
    )

  def set_torque(self, torque):
    self.bullet.setJointMotorControl2(
      bodyIndex=self.body_index,
      jointIndex=self.joint_index,
      controlMode=pybullet.TORQUE_CONTROL,
      force=torque,
    )

  def disable_motor(self):
    self.bullet.setJointMotorControl2(
      bodyIndex=self.body_index,
      jointIndex=self.joint_index,
      controlMode=pybullet.POSITION_CONTROL,
      targetPosition=0,
      targetVelocity=0,
      force=0,
      positionGain=0.1,
      velocityGain=0.1,
    )

class Robot:
  def __init__(self, model_path, robot_name):
    self.model_path = model_path
    self.robot_name = robot_name
    self.bullet = None
    self.objects = None
    self.robot_body = None
    self.parts = None
    self.joints = None

  def add_to_scene(self, bodies):
    self.parts = OrderedDict()
    self.joints = OrderedDict()

    for body_index in bodies:
      num_joints = self.bullet.getNumJoints(body_index)

      if num_joints == 0:
        part_name, robot_name = self.bullet.getBodyInfo(body_index)
        self.robot_name = robot_name.decode("utf8")
        part_name = part_name.decode("utf8")
        part = BodyPart(self.bullet, part_name, body_index, -1)
        self.parts[part_name] = part

      for joint_index in range(num_joints):
        # TODO: what is this?
        # self.bullet.setJointMotorControl2(
        #   bodyIndex=body_index,
        #   jointIndex=joint_index,
        #   controlMode=pybullet.POSITION_CONTROL,
        #   positionGain=0.1,
        #   velocityGain=0.1,
        #   force=0,
        # )
        joint_info = self.bullet.getJointInfo(body_index, joint_index)
        joint_name = joint_info[1].decode("utf8")
        part_name = joint_info[12].decode("utf8")
        part = BodyPart(self.bullet, part_name, body_index, joint_index)
        self.parts[part_name] = part

        if part_name == self.robot_name:
          self.robot_body = part

        if self.robot_body is None:
          part = BodyPart(self.bullet, self.robot_name, body_index, -1)
          self.parts[self.robot_name] = part
          self.robot_body = part

        if joint_name.startswith("ignore"):
          Joint(self.bullet, body, joint_name, joint_index).disable_motor()
          continue

        if not joint_name.startswith("jointfix"):
          joint = Joint(self.bullet, body_index, joint_name, joint_index)
          joint.power_coeff = 100.0
          self.joints[joint_name] = joint

  def reset_pose(self, position, orientation):
    self.parts[self.robot_name].reset_pose(position, orientation)

  def reset(self, bullet):
    self.bullet = bullet
    if self.objects is None:
      flags = (pybullet.URDF_USE_SELF_COLLISION |
               pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS |
               pybullet.URDF_GOOGLEY_UNDEFINED_COLORS)
      self.objects = self.bullet.loadMJCF(self.model_path, flags=flags)
      self.add_to_scene(self.objects)

class RobotEnv:
  def __init__(self, robot):
    self.robot = robot
    self.scene = None
    self.bullet = None
    self.frame = None
    self.state = None
    self.reward = None
    self.done = None
    self.seed()

  def seed(self, seed=None):
    self.np_random = np.random.RandomState(seed=seed)
    self.robot.np_random = self.np_random

  def _build_scene(self):
    raise NotImplementedError

  def reset(self):
    if self.bullet is None:
      self.bullet = _BulletClient(connection_mode=pybullet.DIRECT)
      self.bullet.resetSimulation()
      self.bullet.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
      self.bullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
    if self.scene is None:
      self.scene = self._build_scene(self._p)
    self.scene.episode_restart(self._p)
    self.robot.scene = self.scene
    self.frame = 0
    self.state = self.robot.reset(self._p)
    self.reward = 0
    self.done = False
    return self.state

  def close(self):
    if self.bullet is not None:
      self.bullet.disconnect()
    self.bullet = None
