import os

import argparse

import numpy as np
import pandas as pd
import gymnasium as gym

from tqdm import tqdm as tqdm

def main():

    env = gym.make("CarRacing-v3", render_mode="human", continuous=True)

    progress_bar = tqdm(range(10000), unit='step')

    total_episodes = 0
    
    observation,_ = env.reset()
    for t in progress_bar:
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, _ = env.reset()
            total_episodes += 1
        else:
            observation = next_obs
    print(f"drove for {total_episodes} number of episodes")

if __name__ == "__main__":
    main()
