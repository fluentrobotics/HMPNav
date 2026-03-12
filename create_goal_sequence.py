import numpy as np

if __name__ == '__main__':
    goals = []
    for i in range(40):
        if len(goals) == 0:
            gs = np.array([0, 1, 2, 3, 4, 5])
        elif goals[-1] == 0:
            gs = np.array([3, 4, 5])
        elif goals[-1] == 1:
            gs = np.array([3, 4, 5])
        elif goals[-1] == 2:
            gs = np.array([4, 5])
        elif goals[-1] == 3:
            gs = np.array([0, 1])
        elif goals[-1] == 4:
            gs = np.array([0, 1, 2])
        elif goals[-1] == 5:
            gs = np.array([0, 1, 2])
        g = np.random.choice(gs)
        goals.append(g.item())
    print(goals)