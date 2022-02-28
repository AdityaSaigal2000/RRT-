import matplotlib.pyplot as plt
import numpy as np
from l2_planning import *
def main():
    def plot_angle(c,r,theta):
        t = np.linspace(0,50)
        x = np.cos(-theta)*t + c
        y = np.sin(-theta)*t + r
        return x, y

    node_path_metric = np.load("shortest_path.npy")

    #Set map information
    # "willowgarageworld_05res.png"
    goal_cell = np.array([[1502],[1252]])

    # rows 350 1300
    # cols 400 1600
    region = np.array([[350, 400],
                       [1600, 1300]])
    map_filename ="willowgarageworld_05res.png"
    map_setings_filename = "willowgarageworld_05res.yaml"
    #robot information
    stopping_dist = 0.5 #m

    path_planner = PathPlanner(map_filename, map_setings_filename, goal_cell, stopping_dist)
    goal_point = path_planner.cell_to_point(goal_cell)

    #RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)

    r,c = path_planner.point_to_cell(node_path_metric[:2,:])
    plt.imshow(path_planner.occupancy_map)
    plt.plot(c, r)
    plt.scatter(c, r)
    thetas = node_path_metric[2, :]
    for cx, ry, th in zip(c, r, thetas):
        px, py = plot_angle(cx, ry, th)
        plt.plot(px,py,'k',linewidth=2)
    r,c = path_planner.point_to_cell(goal_point)
    plt.scatter(c, r)

    r,c = path_planner.point_to_cell(np.array([[0],[0]]))
    plt.scatter(c, r)

    plt.show()

if __name__ == "__main__":
    main()
