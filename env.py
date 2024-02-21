import gymnasium 
from gymnasium import Env
from enum import IntEnum

class RandomWalkAction(IntEnum):
    LEFT = 0
    RIGHT = 1

# RandomWalk Env as described in Example 6.2 of Reinforcement Learning: An Introduction
class RandomWalk(Env):
    def __init__(self):
        self.state = 0
        self.action_space = gymnasium.spaces.Discrete(len(RandomWalkAction))
        self.observation_space = gymnasium.spaces.Discrete(7)

    def step(self, action: RandomWalkAction):
        assert self.action_space.contains(action)
        if action == 0:
            self.state = max(0, self.state - 1)
        else:
            self.state = min(6, self.state + 1)

        if self.state == 6:
            reward = 1
            done = True
        elif self.state == 0:
            reward = 0
            done = True
        else:
            reward = 0
            done = False
        return self.state, reward, done, False, {}

    def reset(self):
        self.state = 3
        return self.state, {}