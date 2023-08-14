from utils.agent import Agent
from utils.state import JointState
from policy.followDQN1 import Follower1
from policy.followDQN2 import Follower2
from math import atan2

class Follower(Agent):
    def __init__(self):
        super(Follower, self).__init__()

    def configure(self,config):
        self.time_step = config.getfloat('env','time_step')

    def act(self, ob, exclusiveState):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state, exclusiveState)
        self.vx = action.vx
        self.vy = action.vy
        self.theta = atan2(self.vy, self.vx)
        return action
