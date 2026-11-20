import numpy as np
import gymnasium as gym

class DynamicsHandler:
    def __init__(self, env_name:str):
        if env_name in gym.registry.keys():
            self.name = env_name
        else:
            raise ValueError(f"{env_name} is not a valid gymnasium env")

    def make_dynamics_dict(self)->None:
        # Envs share a method if and only if their dynamics are identical
        self.dynamics_dict = {
            "CartPole-v0": self.linearize_cartpole_v0_v1,
            "CartPole-v1": self.linearize_cartpole_v0_v1,
            "MountainCar-v0": self.linearize_mountaincar_v0,
            "MountainCarContinuous-v0": self.linearize_mountaincarcontinuous_v0,
            "Pendulum-v1": self.linearize_pendulum_v1,
            "phys2d/Pendulum-v0": self.linearize_pendulum_v1,
            "Acrobot-v1": self.linearize_acrobot_v1,
            "phys2d/CartPole-v0": self.linearize_phys2d_cartpole_v0_v1,
            "phys2d/CartPole-v1": self.linearize_phys2d_cartpole_v0_v1,
            "LunarLander-v3": self.linearize_lunarlander_v3,
            "LunarLanderContinuous-v3": self.linearize_lunarlandercontinuous_v3,
            "BipedalWalker-v3": self.linearize_bipedalwalker_v3,
            "BipedalWalkerHardcore-v3": self.linearize_bipedalwalkerhardcore_v3,
            "CarRacing-v3": self.linearize_carracing_v3,
            "Reacher-v2": self.linearize_reacher_v2,
            "Reacher-v4": self.linearize_reacher_v4_v5,
            "Reacher-v5": self.linearize_reacher_v4_v5,
            "Pusher-v2": self.linearize_pusher_v2,
            "Pusher-v4": self.linearize_pusher_v4,
            "Pusher-v5": self.linearize_pusher_v5,
            "InvertedPendulum-v2": self.linearize_invertedpendulum_v2,
            "InvertedPendulum-v4": self.linearize_invertedpendulum_v4_v5,
            "InvertedPendulum-v5": self.linearize_invertedpendulum_v4_v5,
            "InvertedDoublePendulum-v2": self.linearize_inverteddoublependulum_v2,
            "InvertedDoublePendulum-v4": self.linearize_inverteddoublependulum_v4_v5,
            "InvertedDoublePendulum-v5": self.linearize_inverteddoublependulum_v4_v5,
            "HalfCheetah-v2": self.linearize_halfcheetah_v2,
            "HalfCheetah-v3": self.linearize_halfcheetah_v3,
            "HalfCheetah-v4": self.linearize_halfcheetah_v4_v5,
            "HalfCheetah-v5": self.linearize_halfcheetah_v4_v5,
            "Hopper-v2": self.linearize_hopper_v2,
            "Hopper-v3": self.linearize_hopper_v3,
            "Hopper-v4": self.linearize_hopper_v4_v5,
            "Hopper-v5": self.linearize_hopper_v4_v5,
            "Swimmer-v2": self.linearize_swimmer_v2,
            "Swimmer-v3": self.linearize_swimmer_v3,
            "Swimmer-v4": self.linearize_swimmer_v4_v5,
            "Swimmer-v5": self.linearize_swimmer_v4_v5,
            "Walker2d-v2": self.linearize_walker2d_v2,
            "Walker2d-v3": self.linearize_walker2d_v3,
            "Walker2d-v4": self.linearize_walker2d_v4,
            "Walker2d-v5": self.linearize_walker2d_v5,
            "Ant-v2": self.linearize_ant_v2,
            "Ant-v3": self.linearize_ant_v3,
            "Ant-v4": self.linearize_ant_v4_v5,
            "Ant-v5": self.linearize_ant_v4_v5,
            "Humanoid-v2": self.linearize_humanoid_v2,
            "Humanoid-v3": self.linearize_humanoid_v3,
            "Humanoid-v4": self.linearize_humanoid_v4_v5,
            "Humanoid-v5": self.linearize_humanoid_v4_v5,
            "HumanoidStandup-v2": self.linearize_humanoidstandup_v2,
            "HumanoidStandup-v4": self.linearize_humanoidstandup_v4_v5,
            "HumanoidStandup-v5": self.linearize_humanoidstandup_v4_v5,
        }
