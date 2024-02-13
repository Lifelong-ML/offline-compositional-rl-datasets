import os
import json
import numpy as np

from composuite.env.gym_wrapper import GymWrapper


class CompoSuiteGymnasiumWrapper(GymWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.step_counter = 0

    def reset(self):
        obs = super().reset()
        self.step_counter = 0
        return obs, {}

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if self.step_counter == self.horizon - 1:
            truncated = True
            self.step_counter = 0
        else:
            truncated = False
            self.step_counter += 1

        return obs, reward, done, truncated, info
