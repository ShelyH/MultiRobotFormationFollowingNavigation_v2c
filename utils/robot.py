from utils.agent import Agent
from utils.state import JointState
# from action import ActionXY
from math import atan2


class Robot(Agent):
    def __init__(self):
        super(Robot, self).__init__()

    def configure(self,config):
        self.time_step = config.getfloat('env','time_step')

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        self.vx = action.vx
        self.vy = action.vy
        self.theta = atan2(self.vy, self.vx)
        return action
