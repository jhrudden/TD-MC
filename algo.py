from gymnasium import Env
from tqdm import trange, tqdm
from typing import Callable, Optional, List
import numpy as np
from collections import defaultdict

def generate_episode(env: Env, policy: Callable):
    episode = []
    state, _ = env.reset()
    done = False
    truncated = False
    while not done and not truncated:
        action, action_prob = policy(state)
        next_state, reward, done, truncated, _ = env.step(action)
        episode.append((state, action, action_prob, reward))
        state = next_state
    return episode

def get_time_steps_first_visit(episode, is_state_action_pair=False):
    """
    Returns the time steps of the first visit of each state in episode.

    Args:
        episode: The episode to get the time steps of the first visit for
        is_state_action_pair: Whether to define state-action or state as a visit. If True, the state-action pair is used as a visit.
    """  
    fv_time_steps = {}
    for i, (state, action, _, _) in enumerate(episode):
        if is_state_action_pair:
            visit = (state, action)
        else:
            visit = state
        if visit not in fv_time_steps:
            fv_time_steps[visit] = i
    return fv_time_steps

def mc_prediction(env: Env, policy: Callable, gamma: float, n_episodes: int, initial_V: np.ndarray = None, alpha: Optional[float] = None, verbose: bool = False):
    """
    Every visit Monte Carlo prediction algorithm for estimating the value function of a policy using n_episodes.

    Args:
        env: The environment to use
        policy: The policy to evaluate
        gamma: The discount factor
        n_episodes: The number of episodes to use
        initial_V: The initial value function estimates
        alpha: The step size to use. If None, alpha will be set to 1 / N(s), where N(s) is the number of times state s has been visited.
        verbose: Whether to print the progress of the algorithm
    
    Returns:
        The value function estimates and the value function estimates for each episode
    """
    if initial_V is not None:
        V = initial_V.copy()
    else:
        V = np.zeros(env.observation_space.n)

    C = np.zeros(env.observation_space.n)

    V_estimates = np.zeros((n_episodes, env.observation_space.n))
    
    # use range instead of trange to avoid tqdm overhead when verbose is False
    if verbose:
        range_ = trange(n_episodes)
    else:
        range_ = range(n_episodes)
    for i in range_:
        # Save the current value function estimates
        V_estimates[i] = V.copy()
        episode = generate_episode(env, policy)
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            state, action, _, reward = episode[t]
            G = gamma * G + reward
            C[state] += 1
            if alpha is None:
                alpha = 1 / C[episode[t][0]]
            V[state] += alpha * (G - V[state])
    return V, V_estimates


def td0(env: Env, policy: Callable, alpha: float, gamma: float, n_episodes: int, initial_V: np.ndarray = None, verbose: bool = False):
    """
    Temporal Difference Learning algorithm for estimating the value function of a policy using n_episodes.

    Args:
        env: The environment to use
        policy: The policy to evaluate
        alpha: The step size to use
        gamma: The discount factor
        n_episodes: The number of episodes to use
        initial_V: The initial value function estimates
        verbose: Whether to print the progress of the algorithm

    Returns:
        The value function estimates and the value function estimates for each episode
    """
    # TODO: Should these be initialized to 0 or random? Besides the terminal states
    if initial_V is not None:
        V = initial_V.copy()
    else:
        V = np.zeros(env.observation_space.n)
    V_estimates = np.zeros((n_episodes, env.observation_space.n))

    # use range instead of trange to avoid tqdm overhead when verbose is False
    if verbose:
        range_ = trange(n_episodes)
    else:
        range_ = range(n_episodes)
    for i in range_:
        # Save the current value function estimates
        V_estimates[i] = V.copy()
        
        state, _ = env.reset()
        done = False
        truncated = False
        while not done and not truncated:
            action, _ = policy(state)
            next_state, reward, done, truncated, _ = env.step(action)
            V[state] += alpha * (reward + gamma * V[next_state] - V[state])
            state = next_state
    return V, V_estimates

def on_policy_mc_control_fv(env: Env, policy_builder: Callable, gamma: float, n_episodes: int, verbose: bool = False):
    """
    On-policy first visit Monte Carlo control algorithm for estimating the optimal policy and the optimal value function using n_episodes.

    Args:
        env: The environment to use
        policy_builder: Callable which takes Q estimates and returns a policy
        gamma: The discount factor
        n_episodes: The number of episodes to use
        verbose: Whether to print the progress of the algorithm

    Returns:
        The optimal value function, the optimal policy, the action-value function estimates, and all episodes
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = policy_builder(Q)

    all_episodes = []

    
    # use range instead of trange to avoid tqdm overhead when verbose is False
    if verbose:
        range_ = trange(n_episodes)
    else:
        range_ = range(n_episodes)
    
    for i in range_:
        episode = generate_episode(env, policy)
        all_episodes.append(episode)
        fv_map = get_time_steps_first_visit(episode, is_state_action_pair=True)
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            state, action, _, reward = episode[t]
            G = gamma * G + reward
            if fv_map[(state, action)] == t:
                N[state][action] += 1
                Q[state][action] += (G - Q[state][action]) / N[state][action]
        
        if verbose:
            # add G to the trange description if verbose
            range_.set_description(f"Episode {i + 1} | Return: {G:.2f}")

    V = {state: np.max(action_values) for state, action_values in Q.items()}
    return V, policy, Q, all_episodes

def off_policy_mc_prediction(env: Env, gamma: float, pre_generated_eps: List, verbose: bool = False):
    """
    Off-policy Monte Carlo prediction algorithm for estimating the value function of a target policy using n_episodes.

    Args:
        env: The environment to use
        gamma: The discount factor
        pre_generated_eps: Pre-generated episodes to use (generated using a behavior policy)
        verbose: Whether to print the progress of the algorithm
    """
    C = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # use range instead of trange to avoid tqdm overhead when verbose is False
    if verbose:
        items_ = tqdm(pre_generated_eps, desc="Episodes")
    else:
        items_ = pre_generated_eps
    for episode in items_:
        G = 0
        W = 1
        for t in range(len(episode) - 1, -1, -1):
            if W == 0:
                break
            state, action, action_prob, reward = episode[t]
            
            G = gamma * G + reward
            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
            pi_action_prob = 0
            if action == np.argmax(Q[state]):
                pi_action_prob = 1
            W *= pi_action_prob / action_prob

    V = {state: np.max(action_values) for state, action_values in Q.items()}
    Q = {state: action_values for state, action_values in Q.items()}
    return Q, V


def on_policy_mc_Q_prediction(env: Env, policy: Callable, gamma: float, n_episodes: int, verbose: bool = False):
    """
    On-policy Monte Carlo prediction algorithm for estimating the action-value function of a policy using n_episodes.

    Args:
        env: The environment to use
        policy: The policy to evaluate
        gamma: The discount factor
        n_episodes: The number of episodes to use
        verbose: Whether to print the progress of the algorithm
    
    Returns:
        The action-value function estimates and the action-value function estimates for each episode
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))

    # use range instead of trange to avoid tqdm overhead when verbose is False
    if verbose:
        range_ = trange(n_episodes)
    else:
        range_ = range(n_episodes)
    for i in range_:
        episode = generate_episode(env, policy)
        fv_map = get_time_steps_first_visit(episode, is_state_action_pair=True)
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            state, action, action_prob, reward = episode[t]
            G = gamma * G + reward
            if fv_map[(state, action)] == t:
                N[state][action] += 1
                Q[state][action] += (G - Q[state][action]) / N[state][action]
    return Q