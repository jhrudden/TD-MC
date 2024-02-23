import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from typing import Dict, Optional

def plot_policy_on_grid_world(env, Q, title="Policy", save_path=None):
    """
    Plots the policy on a grid world environment.
    
    Args:
        env: The grid world environment to use
        Q: The action-value function estimates
    """
    fig, ax = plt.subplots()
    for i in range(env.unwrapped.rows):
        for j in range(env.unwrapped.cols):
            state = (j, i)  # Corrected state to match row, col format

            # Fill in walls
            if state in env.unwrapped.walls:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='grey'))  # Fill cell for walls
                continue  # Skip drawing arrows for walls

            actions_estimates = Q.get(state, np.zeros(env.action_space.n))
            if np.all(actions_estimates == 0) and state != env.unwrapped.goal_pos:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='yellow', alpha=0.3))  # Fill cell for terminal state
            action = np.argmax(actions_estimates)

            if action == 0: # left
                plt.arrow(j, i, -0.3, 0, head_width=0.1, head_length=0.1)
            elif action == 1: # doin 
                plt.arrow(j, i, 0, -0.3, head_width=0.1, head_length=0.1)
            elif action == 2: # right
                plt.arrow(j, i, 0.3, 0, head_width=0.1, head_length=0.1)
            elif action == 3: # up
                plt.arrow(j, i, 0, 0.3, head_width=0.1, head_length=0.1)

    ax.set_aspect('equal')
    ax.set_xlim(-0.5,env.unwrapped.cols-0.5)
    ax.set_ylim(-0.5,env.unwrapped.rows-0.5)

    # Set grid
    ax.set_xticks(np.arange(-0.5, env.unwrapped.cols, 1))
    ax.set_yticks(np.arange(-0.5, env.unwrapped.cols, 1))
    ax.grid(True, linestyle='-', color='black', linewidth=2)

    # Remove labels and tick marks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_grid_world_value_function(env, V: Dict, title: Optional[str] = None, save_path: Optional[str] = None):
    """
    plots the value function as a heatmap for a cardinal gridworld

    Args:
        env: The grid world environment to use
        V: The value function estimates
        title: The title of the plot
        save_path: The path to save the plot to
    """
    V_array = np.zeros((env.unwrapped.rows, env.unwrapped.cols))
    for i in range(env.unwrapped.rows):
        for j in range(env.unwrapped.cols):
            state = (j, i)
            V_array[i, j] = V.get(state, 0.0)

    V = np.flipud(V_array)

    fig, ax = plt.subplots(1,1, figsize=(10, 10))
    im = ax.imshow(V, cmap='coolwarm', norm=Normalize(vmin=V.min(), vmax=V.max()), alpha=0.8)
    

    # Loop over data dimensions and create text annotations.
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            text = ax.text(j, i, f'{V[i, j]:.3f}',
                           ha="center", va="center", color="black", fontsize=10)

    if title:
        # add some padding to the title
        ax.set_title(title, fontsize=15, pad=20)
    ax.axis('off')

    if save_path:
        plt.savefig(save_path, format='png')
    
    plt.show()
