import numpy as np
import matplotlib.pyplot as plt
from controller_config import *

def plot_obstacle(obs, c):
    for i in range(4):
        plt.plot([obs[i-1][0], obs[i][0]], [obs[i-1][1], obs[i][1]], color=c)

def construct_square_obstacle(center, l=1.5):
        d = l/2
        return [(center[0]+d, center[1]+d), (center[0]-d, center[1]+d), (center[0]-d, center[1]-d), (center[0]+d, center[1]-d)]
    
def construct_boundary(left, top, right, bottom, out=10.00):
    obs = [
        [(out, -out), (bottom, -out), (bottom, left), (out, left)], #bottom left
        [(bottom, -out), (top, -out), (top, left), (bottom, left)], #left
        [(top, -out), (-out, -out), (-out, left), (top, left)], #top left
        [(top, left), (-out, left), (-out, right), (top, right)], #top
        [(top, right), (-out, right), (-out, out), (top, out)], #top right
        [(bottom, right), (top, right), (top, out), (bottom, out)], #right
        [(out, right), (bottom, right), (bottom, out), (out, out)], #bottom right
        [(out, left), (bottom, left), (bottom, right), (out, right)]  #bottom
    ]
    return obs

if __name__ == '__main__':
    plt.figure(figsize=(12, 12))

    colors = ['red', 'green', 'blue', 'orange', 'yellow', 'black', 'purple', 'brown']
    static_obs = construct_boundary(ROBOT_BOUNDARY[0], ROBOT_BOUNDARY[1], ROBOT_BOUNDARY[2], ROBOT_BOUNDARY[3])

    for i, obs in enumerate(static_obs):
         plot_obstacle(obs, colors[i])

    workstation_polygons = []
    for o in WORKSTATION_OBS:
        plot_obstacle(construct_square_obstacle(o,l=1.25), c='pink')

    for g in GOALS:
         plt.scatter([g[0]], [g[1]], color='red')

    for g in AGENT_GOALS:
         plt.scatter([g[0]], [g[1]], color='blue')

    plt.xlim = (-12.5, 12.5)
    plt.ylim = (-12.5, 12.5)

    plt.show()

