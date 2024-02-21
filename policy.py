import numpy as np

def equiprobable_policy(env):
    def policy(observation):
        action = np.random.choice(env.action_space.n)
        return (action, 1 / env.action_space.n)
    return policy