import os

import argparse

import numpy as np
import pandas as pd
import gymnasium as gym

from tqdm import tqdm as tqdm

from Controllers.lqr import LQR
from Utils.linearize_dynamics import DynamicsHandler

def main():
    env_name = "Pendulum-v1"
    env = gym.make(env_name, render_mode="human")

    dh_ = DynamicsHandler(env)
    s_goal = np.array([1.0, 0.0, 0.0])
    u_goal = np.array([1.0])

    linear_model = dh_.linearization_method_(s_goal, u_goal)
    dynamics_matrix = linear_model[0]
    control_matrix = linear_model[1]

    controller = LQR(state_dim=2,
                     control_dim = env.action_space.shape[0],
                     A=dynamics_matrix,
                     B=control_matrix,
                     Q=np.eye(2),
                     R=np.eye(1),
                     P=np.eye(2),
                     N=5,
                     )

    progress_bar = tqdm(range(10000), unit='step')

    total_episodes = 0
    
    observation,_ = env.reset(options={"x_init":np.cos(18 * np.pi / 180),
                                       "y_init":np.sin(18 * np.pi / 180)},
                              )
    for t in progress_bar:
        state = np.array([[np.arctan2(observation[1], observation[0])],
                          [observation[2]]])
        action = controller.get_lqr_solution(state,
                                             -2.0,
                                             2.0)
        next_obs, reward, terminated, truncated, info = env.step(action.reshape(env.action_space.shape))
        if terminated or truncated:
            observation, _ = env.reset(options={"x_init":np.cos(18 * np.pi / 180), 
                                                "y_init":np.sin(18 * np.pi / 180)},
                                       )

            total_episodes += 1
        else:
            observation = next_obs
    print(f"drove for {total_episodes} number of episodes")

if __name__ == "__main__":
    main()
