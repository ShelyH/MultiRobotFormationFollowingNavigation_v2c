import numpy as np
from numpy.linalg import norm
import abc
import logging
from utils.action import ActionXY
from utils.state import ObservableState, FullState

from utils.Loging import Log
logging = Log(__name__).getlog()

class Agent(object):
    def __init__(self):
        self.v_pref = 1.0
        self.radius = 0.3
        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.theta = None
        self.policy = None
        self.time_step = 0.25

    def set_policy(self, policy):
        self.policy = policy

    def sample_random_attributes(self):
        self.v_pref = np.random.uniform(0.5, 1.5)
        self.radius = np.random.uniform(0.3, 0.5)

    def set(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta
        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_pref = v_pref

    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)

    def get_next_observable_state(self, action):
        pos = self.compute_position(action, self.time_step)
        next_px, next_py = pos
        next_vx = action.vx
        next_vy = action.vy
        return ObservableState(next_px, next_py, next_vx, next_vy, self.radius)

    def get_full_state(self):
        return FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)

    def get_position(self):
        return self.px, self.py

    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

    def get_goal_position(self):
        return self.gx, self.gy

    def get_velocity(self):
        return self.vx, self.vy

    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]

    @abc.abstractmethod
    def act(self, ob):
        return

    def compute_position(self, action, delta_t):
        px = self.px + action.vx * delta_t
        py = self.py + action.vy * delta_t
        return px, py

    def step(self, action):
        pos = self.compute_position(action, self.time_step)
        self.px, self.py = pos
        self.vx = action.vx
        self.vy = action.vy

    def reached_destination(self):
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius

