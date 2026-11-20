import numpy as np
import gymnasium as gym

class DynamicsHandler:
    def __init__(self, env_name:str):
        if env_name in gym.registry.keys():
            self.name = env_name
        else:
            raise ValueError(f"{env_name} is not a valid gymnasium env")
