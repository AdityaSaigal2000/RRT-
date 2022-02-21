import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print(sys.path)
from l2_planning import *
import matplotlib.pyplot as plt

def main():

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

        nodes, samples = path_planner.rrt_planning(1000, region)
        node_path_metric = np.hstack(path_planner.recover_path())
        samples = np.hstack(samples)
        sr, sc = path_planner.point_to_cell(samples)
        plt.scatter(sc,sr)

        r,c = path_planner.point_to_cell(node_path_metric[:2,:])
        plt.imshow(path_planner.occupancy_map)
        plt.plot(c, r)

        r,c = path_planner.point_to_cell(node_path_metric[:2,:])
        plt.scatter(c, r)

        r,c = path_planner.point_to_cell(goal_point)
        plt.scatter(c, r)

        r,c = path_planner.point_to_cell(np.array([[0],[0]]))
        plt.scatter(c, r)

        plt.show()

        np.save("shortest_path.npy", node_path_metric)

if __name__ == '__main__':
    main()
