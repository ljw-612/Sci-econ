import matplotlib.pyplot as plt
import numpy as np
import random
import math

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

class BlockWithholdingAttackAgent(Agent):

    def __init__(self, unique_id, model, sigma, a1, a2, gamma, R, C1, C2):
        super().__init__(unique_id, model)
        self.sigma = sigma; self.a1 = a1; self.a2 = a2
        self.lbd = a2 / a1; self.gamma = gamma
        self.R = R; self.C1 = C1; self.C2 = C2

        self.x = 0.3; self.y = 0.3

    def generate_x_change_rate(self):
        # print(self.x) 
        U11 = self.y*(self.a1*self.gamma*self.R-self.C1)+(1-self.y)*(self.a1*self.sigma*self.R+(1-self.sigma)*self.R*self.a1**2-self.C1)
        U12 = self.y*((1-self.sigma)*self.a1*self.a2*self.R-self.C2)+(1-self.y)*(-self.C2)
        U1averge = self.x * U11 + (1-self.x) * U12
        x_change_rate = U11 / U1averge
        return x_change_rate

    def generate_y_change_rate(self):
        # print(self.y)
        U21 = self.x*(self.a2*self.gamma*self.R-self.C1)+(1-self.x)*(self.a2*self.sigma*self.R+(1-self.sigma)*self.R*self.a2**2-self.C1)
        U22 = self.x*((1-self.sigma)*self.a1*self.a2*self.R-self.C2)+(1-self.x)*(-self.C2)
        U2averge = self.y * U21 + (1-self.y) * U22
        y_change_rate = U21 / U2averge
        return y_change_rate

    def generate_x(self):
        x = self.x * self.generate_x_change_rate()
        return x

    def generate_y(self):
        y = self.y * self.generate_y_change_rate()
        return y

    def step(self):

        x = self.generate_x(); y = self.generate_y()
        # update x and y
        self.x = x; self.y = y
        self.model.x = x; self.model.y = y
        result = (x, y)
        #print('x',a)
        #print('y',b)
        return result

class BlockWithholdingAttackModel(Model):

    def __init__(self, sigma, a1, a2, gamma, R, C1, C2):
        self.num_agent = 2
        self.schedule = RandomActivation(self)
        self.x = 0.3; self.y = 0.3

        for i in range(self.num_agent):
            a = BlockWithholdingAttackAgent(i, self, sigma, a1, a2, gamma, R, C1, C2)
            self.schedule.add(a)

    def step(self) -> None:
        self.schedule.step()

if __name__ == '__main__':
    model = BlockWithholdingAttackModel(sigma=0.2, a1=0.6, a2=0.4, gamma=2, R=10, C1=4, C2=1)
    x_list = []; y_list = []
    for i in range(20):
        model.step()
        x_list.append(model.x)
        y_list.append(model.y)

    print(x_list)
    print(y_list)

