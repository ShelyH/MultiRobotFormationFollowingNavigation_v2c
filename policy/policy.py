import numpy as np


class Policy(object):
    def __init__(self):
        self.trainable = False
        self.phase = 'train'
        self.model = None
        self.device = None
        self.last_state = None
        self.time_step = None
        self.env = None

    def set_phase(self, phase):
        self.phase = phase

    def set_device(self, device):
        self.device = device

    def set_env(self, env):
        self.env = env

    def get_model(self):
        return self.model
