# -*- coding: utf-8 -*-

from kernal import kernal


class SimInterface(object):
    def __init__(self, agent_num, time=180, render=True):
        self.game = kernal(car_num=agent_num, time=time, render=render)
        self.g_map = self.game.get_map()
        self.memory = []

    def reset(self):
        self.state = self.game.reset()
        # state, object
        self.obs = self.get_observation(self.state)
        return self.obs

    def step(self, actions):
        state = self.game.step(actions)
        obs = self.get_observation(state)
        rewards = self.get_reward(state)

        self.memory.append([self.obs, actions, rewards])
        self.state = state

        #return obs, rewards, state.done, None
        return state.done

    def get_observation(self, state):
        # personalize your observation here
        obs = state
        return obs

    def get_reward(self, state):
        # personalize your reward here
        rewards = None
        return rewards

    def play(self, cars_info, cars_guide, endFlag):
        self.game.play(cars_info, cars_guide, endFlag)
