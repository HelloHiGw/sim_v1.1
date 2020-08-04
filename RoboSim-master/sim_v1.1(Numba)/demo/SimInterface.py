# -*- coding: utf-8 -*-

from kernal import *
import pygame
import numpy as np

class SimInterface(object):
    def __init__(self, time=180, render=0, dev=0):
        # contant params
        map_length = 800
        map_width = 600
        barriers = np.array([[350.0, 450.0, 250.0, 275.0], # O1 -
                            [350.0, 450.0, 275.0, 300.0], # O1 -
                            [350.0, 450.0, 300.0, 325.0], # O1 -
                            [350.0, 450.0, 325.0, 350.0], # O1 -
                            [200.0, 300.0,  87.0, 113.0], # O2 -
                            [200.0, 300.0, 487.0, 513.0], # O3 -
                            [500.0, 600.0,  87.0, 113.0], # O4 -
                            [500.0, 600.0, 487.0, 513.0], # O5 -
                            [187.0, 213.0, 100.0, 200.0], # O2 |
                            [187.0, 213.0, 400.0, 500.0], # O3 |
                            [587.0, 613.0, 100.0, 200.0], # O4 |
                            [587.0, 613.0, 400.0, 500.0], # O5 | 
                        ], dtype=np.float32) # barriers
        areas = np.array([[[0.0, 100.0, 0.0, 100.0],
                        [0.0, 100.0, 125.0, 225.0],
                        [0.0, 100.0, 250.0, 350.0],
                        [0.0, 100.0, 375.0, 475.0],
                        [0.0, 100.0, 500.0, 600.0]],
                        [[700.0, 800.0, 0.0, 100.0],
                        [700.0, 800.0, 125.0, 225.0],
                        [700.0, 800.0, 250.0, 350.0],
                        [700.0, 800.0, 375.0, 475.0],
                        [700.0, 800.0, 500.0, 600.0]]], 
                        dtype=np.float32) # start areas

        # all the competition variables are bounded to member variables for easy access
        self.time, self.done, self.epoch, self.detect, self.vision, \
        self.agents, self.bullets, self.bullets_count, \
        self.orders, self.acts, \
        self.guides, self.arrivals, self.next_points, self.control_n \
        = init()

        self.render = render
        self.dev = dev
        if dev != 0:
            self.render = 1

        self.memory = [] # memory

        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((map_length, map_width))
            pygame.display.set_caption('RoboSim 5 V.S. 5')

            # load barriers img
            self.barriers_img = []
            self.barriers_rect = []
            for i in range(barriers.shape[0]):
                self.barriers_img.append(
                    pygame.image.load('./imgs/barrier_{}.png'.format('horizontal' if i < 8 else 'vertical')))
                self.barriers_rect.append(self.barriers_img[-1].get_rect())
                self.barriers_rect[-1].center = [barriers[i][0:2].mean(), barriers[i][2:4].mean()]

            # load areas and cars img
            self.areas_img = []
            self.areas_rect = []
            for oi, o in enumerate(['red', 'blue']):
                for ti in range(5): # start areas
                    self.areas_img.append(pygame.image.load('./imgs/area_start_{}.png'.format(o)))
                    self.areas_rect.append(self.areas_img[-1].get_rect())
                    self.areas_rect[-1].center = [areas[oi, ti][0:2].mean(), areas[oi, ti][2:4].mean()]

            self.chassis_img = pygame.image.load('./imgs/chassis_g.png') # chasis
            self.gimbal_img = pygame.image.load('./imgs/gimbal_g.png') # gimbal
            self.bullet_img = pygame.image.load('./imgs/bullet_s.png') # bullet
            self.bullet_rect = self.bullet_img.get_rect()
            self.info_bar_img = pygame.image.load('./imgs/info_bar.png') # info bar 
            self.info_bar_rect = self.info_bar_img.get_rect()
            self.info_bar_rect.center = [200, map_width / 2]

            pygame.font.init()
            self.font = pygame.font.SysFont('info', 20)
            self.clock = pygame.time.Clock()


    def reset(self):
        reset(self.time, self.done, self.epoch, self.detect, self.vision, 
            self.agents, self.bullets, self.bullets_count, \
            self.orders, self.acts, \
            self.guides, self.arrivals, self.next_points, self.control_n)


    def step(self, actions):
        set_orders(self.orders, actions)
        if self.render:
            step(self.time, self.done, self.epoch, 
                self.render, self.dev, 
                self.detect, self.vision, 
                self.agents, self.bullets, self.bullets_count, 
                self.orders, self.acts, 
                self.screen, self.clock, self.font, 
                self.barriers_rect, self.barriers_img, 
                self.areas_rect, self.areas_img, 
                self.bullet_rect, self.bullet_img, 
                self.chassis_img, self.gimbal_img, 
                self.info_bar_rect, self.info_bar_img)
        else:
            step(self.time, self.done, self.epoch, 
                self.render, self.dev, 
                self.detect, self.vision, 
                self.agents, self.bullets, self.bullets_count, 
                self.orders, self.acts)


    def get_observation(self):
        # personalize your observation here
        pass


    def get_reward(self):
        # personalize your reward here
        pass
        

    def play(self):
        # unsupported now
        pass
