import numpy as np
import rvo2
from policy import Policy
from utils.action import ActionXY


class ORCA(Policy):
    def __init__(self):
        super(ORCA, self).__init__()
        self.name = 'ORCA'
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = 'holonomic'
        self.safety_space = 0.01
        self.neighbor_dist = 10.0
        self.max_neighbors = 10.0
        self.time_horizon = 5.0
        self.time_horizon_obst = 5.0
        self.radius = 0.3
        self.max_speed = 1.0
        self.sim = None
        self.time_step = 0.25

    def predict(self, state):
        self_state = state.self_state
        # params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
        if self.sim is not None and self.sim.getNumAgents() != len(state.human_states) + 1:
            del self.sim
            self.sim = None
        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(self.time_step, self.neighbor_dist, self.max_neighbors, self.time_horizon,
                                           self.time_horizon_obst, self.radius, self.max_speed)
            self.sim.addAgent(self_state.position, self.neighbor_dist, self.max_neighbors, self.time_horizon,
                              self.time_horizon_obst, 0.32, self_state.v_pref, self_state.velocity)
            for human_state in state.human_states:
                self.sim.addAgent(human_state.position, self.neighbor_dist, self.max_neighbors, self.time_horizon,
                                  self.time_horizon_obst, 0.32, self.max_speed, human_state.velocity)
        else:
            self.sim.setAgentPosition(0, self_state.position)
            self.sim.setAgentVelocity(0, self_state.velocity)
            for i, human_state in enumerate(state.human_states):
                self.sim.setAgentPosition(i + 1, human_state.position)
                self.sim.setAgentVelocity(i + 1, human_state.velocity)

        # Set the preferred velocity to be a vector of unit magnitude (speed) in the direction of the goal.
        velocity = np.array((self_state.gx - self_state.px, self_state.gy - self_state.py))
        speed = np.linalg.norm(velocity)
        pref_vel = velocity / speed if speed > 1 else velocity

        # Perturb a little to avoid deadlocks due to perfect symmetry.
        # perturb_angle = np.random.random() * 2 * np.pi
        # perturb_dist = np.random.random() * 0.01
        # perturb_vel = np.array((np.cos(perturb_angle), np.sin(perturb_angle))) * perturb_dist
        # pref_vel += perturb_vel

        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))
        for i, human_state in enumerate(state.human_states):
            # unknown goal position of other humans
            self.sim.setAgentPrefVelocity(i + 1, (0, 0))

        self.sim.doStep()
        action = ActionXY(*self.sim.getAgentVelocity(0))
        self.last_state = state

        return action
