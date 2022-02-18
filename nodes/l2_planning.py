#!/usr/bin/env python
#Standard Libraries
import numpy as np
import yaml
import pygame
import time
try:
    # pygame_utils file doesn't have correct magic number i.e. wont run locally
    import pygame_utils
except:
    import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.draw import circle_perimeter
from scipy.linalg import block_diag
from scipy.spatial import KDTree
import copy


'''
Node Lifecycle variables
'''
NODE_ALIVE=0
NODE_DEAD=1


# y corresponds to rows
# x corresponds to cols
# map.shape[0] - is

#Map Handling Functions
def load_map(filename):
    im = mpimg.imread("../maps/" + filename)
    im_np = np.array(im)  #Whitespace is true, black is false
    #im_np = np.logical_not(im_np)
    return im_np

def load_map_yaml(filename):
    with open("../maps/" + filename, "r") as stream:
            map_settings_dict = yaml.safe_load(stream)
    return map_settings_dict

#Node for building a graph
class Node:
    def __init__(self, point, parent_id, cost):
        self.point = point # A 3 by 1 vector [x, y, theta]
        self.parent_id = parent_id # The parent node id that leads to this node (There should only every be one parent in RRT)
        self.cost = cost # The cost to come to this node
        self.children_ids = [] # The children node ids of this node
        self.lifecycle = NODE_ALIVE
        return

    def __hash__(self):
        '''
        Args:
            - none
        Returns:
            - hashed value based on points converted to tuple
        '''
        return hash(tuple(self.point[:,0]))

class NodeCollection:
    def __init__(self):
        self.list = []
        self.set = set()

        # needed for kdtree, need features along cols, and points along rows
        # n.b. we only keep the [x, y] points in val_arr
        self.val_arr = None
        self.kdtree = None

    def rebuild_kdtree(self):
        self.kdtree = KDTree(self.val_arr, leafsize=40)

    def query(self, point, k):
        '''
        Args:
            - point 1x2 vector that will be used to find the closest point [x, y]
            - k number of nearest neighbours to return
        Returns:
            - distance, index - index is wrt to node list
        '''
        point = point if point.shape[1] == 2 else point.T
        if self.kdtree is None or self.kdtree.size != len(self.list):
            self.rebuild_kdtree()
        return self.kdtree.query(point, k)

    def append(self, node):
        self.set.add(node)
        self.list.append(node)
        if self.val_arr is None:
            self.val_arr = node.point[:2,:] if node.point.shape[1]==3 else node.point.T[:,:2]
        else:
            new_point = node.point if node.point.shape[1]==3 else node.point.T
            self.val_arr = np.vstack([self.val_arr, new_point[:,:2]])

    def __contains__(self, point):
        # Needed for 'in' operator
        return point in self.set

    def __len__(self):
        return len(self.list)

    def __getitem__(self, i):
        return self.list[i]

#Path Planner
class PathPlanner:
    #A path planner capable of perfomring RRT and RRT*
    def __init__(self, map_filename, map_setings_filename, goal_point, stopping_dist):
        #Get map information
        self.occupancy_map = load_map(map_filename)
        self.map_shape = self.occupancy_map.shape
        self.map_settings_dict = load_map_yaml(map_setings_filename)

        #Get the metric bounds of the map
        self.bounds = np.zeros([2,2]) #m
        self.bounds[0, 0] = self.map_settings_dict["origin"][0]
        self.bounds[1, 0] = self.map_settings_dict["origin"][1]
        self.bounds[0, 1] = self.map_settings_dict["origin"][0] + self.map_shape[1] * self.map_settings_dict["resolution"]
        self.bounds[1, 1] = self.map_settings_dict["origin"][1] + self.map_shape[0] * self.map_settings_dict["resolution"]

        #Robot information
        self.robot_radius = 0.22 #m
        self.vel_max = 0.5 #m/s (Feel free to change!)
        self.rot_vel_max = 0.2 #rad/s (Feel free to change!)

        #Goal Parameters
        self.goal_point = goal_point #m
        self.stopping_dist = stopping_dist #m

        #Trajectory Simulation Parameters
        self.timestep = 1.0 #s
        self.num_substeps = 10

        #Planning storage
        self.nodes =  NodeCollection()
        self.nodes.append(Node(np.zeros((3,1)), -1, 0))

        #RRT* Specific Parameters
        self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] **2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * (self.lebesgue_free / self.zeta_d) ** (1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 2.5

        #Pygame window for visualization
        # self.window = pygame_utils.PygameWindow(
        #    "Path Planner", (1000, 1000), self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist)
        plt.imshow(self.occupancy_map)
        plt.show()
        return

    #Functions required for RRT
    def sample_map_space(self):
        '''
        Args:
            - None
        Returns:
            - gives random point within map space given some region, point is [x, y] numpy array
        TODO: add support for regions
        '''
        sample = np.random.rand(2,1)
        sample[0,:] *= self.map_shape[1]
        sample[1,:] *= self.map_shape[0]
        return sample.astype(int)

    def check_if_duplicate(self, point):
        '''
        Args:
            - point - Node object, check to see if this Node object exists within current list 3x1
        Returns:
            - bool True if in NodeCollection otherwise no
        '''
        return point in self.nodes

    def closest_node(self, point):
        '''
        Args:
            - point - the target point in 2x1 numpy array format
        Returns:
            - returns the closest point to target point as a index wrt to target list
        TODO: also return the distance
        '''
        dist, ind = self.nodes.query(point, 1)
        return ind[0]

    def simulate_trajectory(self, node_i, point_s):
        #Simulates the non-holonomic motion of the robot.
        #This function drives the robot from node_i towards point_s. This function does has many solutions!
        #node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{OI} in course notation
        #point_s is the sampled point vector [x; y] (2x1)

        # Aditya Saigal
        # expects point_s to have (x, y) map coordinates, not (row, col)
        # node_i is given in the world frame coordinates
        # Iteratively generate control commands and simulate a trajectory until point_s is reached on the map.

        error_thresh = 0.1

        curr_point = node_i
        # convert the map coordinates to world coordinates
        world_sample = np.array([[self.map_settings_dict["origin"][0] + point_s[0][0]*self.map_settings_dict["resolution"]], [self.map_settings_dict["origin"][1] + point_s[1][0]*self.map_settings_dict["resolution"]]]) # 2x1 vector

        #print(curr_point)
        #print(world_sample)
        #exit()
        complete_traj = None
        count = 0
        while(np.linalg.norm(curr_point[0:2] - world_sample) > error_thresh):
                #print(np.linalg.norm(curr_point[0:2] - world_sample))
                #print(curr_point[0:2], world_sample)
                v, w = self.robot_controller(curr_point , world_sample)

                robot_traj = self.trajectory_rollout(curr_point[2], v, w)

                #if(not w):
        #               print(robot_traj)
                robot_traj = np.vstack((curr_point[0] + robot_traj[0], curr_point[1] + robot_traj[1], curr_point[2] + robot_traj[2]))
#               if(not w):
#                       count += 1
        #               print("\n\n")
        #               print(robot_traj)
        #               print("\n")
#                       if(count == 110):
#                               exit()
                curr_point = robot_traj[:, self.num_substeps - 1].reshape(3,1)
                if  complete_traj is None:
                        complete_traj = robot_traj

                else:
                        complete_traj = np.hstack((complete_traj, robot_traj))



        return complete_traj

    def robot_controller(self, node_i, point_s):
        #This controller determines the velocities that will nominally move the robot from node i to node s
        #Max velocities should be enforced

        # Aditya Saigal
        # point_s is assumed to have (x, y) world coordinates, not (row, col)
        # node_i is a world point corresponding to the robot's current location

        # Implements a controller using proportional feedback to generate velocity and angular velocity commands
        # First fix heading and then drive in a straight line to reach desired point

        angular_gain = 0.2
        linear_gain = 0.5
        heading_thresh = 0.0001

        # Convert the map coordinates to world frame.
        #world_sample = np.array([[self.map_settings_dict["origin"][0] + point_s[0]*self.map_settings_dict["resolution"]], [self.map_settings_dict["origin"][1] + point_s[1]*self.map_settings_dict["resolution"]]]) # 2 x1 vector

        #curr_point = node_i.point

        heading = np.arctan2(point_s[1] - node_i[1], point_s[0] - node_i[0])

        if(abs(heading - node_i[2]) > heading_thresh):
                # Robot's current heading is offset from desired angle. First fix this using proportional feedback.
                w = angular_gain*(heading - node_i[2])
                v = 0
                #print("here")
        else:
                #print(node_i)
                # Robot has the correct orientation.Drive towards goal point without changing heading. Use proportional feedback for velocity.
                dist = np.linalg.norm(node_i[0:2] - point_s)
                w = 0
                v = linear_gain*dist
                if(v > self.vel_max):
                        # Enforce max velocity
                        v = self.vel_max

        return v, w

    def trajectory_rollout(self, init_theta, vel, rot_vel):
        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions

        # Aditya Saigal
        sample_times = np.linspace(self.timestep/self.num_substeps, self.timestep, self.num_substeps)

        if(rot_vel):
                # When a non-zero angular velocity is applied
                x = (vel/rot_vel)*np.sin(rot_vel*sample_times + init_theta)
                y = (vel/rot_vel)*(1 - np.cos(rot_vel*sample_times + init_theta))
        else:
                # Only a linear velocity is applied
                x = (vel)*np.cos(init_theta)*sample_times
                y = (vel)*np.sin(init_theta)*sample_times
        theta = rot_vel*sample_times

        return np.vstack((x, y, theta))

    def point_to_cell(self, point):
        #Convert a series of [x,y] points in the map to the indices for the corresponding cell in the occupancy map
        #point is a 2 by N matrix of points of interest
        #print("TO DO: Implement a method to get the map cell the robot is currently occupying")

        #Aditya Saigal
        # return pixel coordinates/indices in (row, col) format


        row = self.map_shape[0] - (point[1] - self.map_settings_dict["origin"][1])/self.map_settings_dict["resolution"] # Find y coordinate wrt lower left corner. Subtract from number of rows in map
        col = (point[0] - self.map_settings_dict["origin"][0])/self.map_settings_dict["resolution"] # Find x coordinate wrt lower left corner. This is the column

        return np.vstack((row, col)) # return indices for all points

    def points_to_robot_circle(self, points):
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function
        #print("TO DO: Implement a method to get the pixel locations of the robot path")

        #Aditya Saigal

        # Expects points in x, y format
        map_coords = self.point_to_cell(points) # convert points to map indices (row col)
        rows, cols = [], []
        print(map_coords)
        print(self.robot_radius/self.map_settings_dict["resolution"])
        for pt in range(map_coords.shape[1]):
                # Get occupancy footprint for each point and store the occupied rows and columns
                rr, cc = circle_perimeter(int(map_coords[0, pt]), int(map_coords[1, pt]), int(np.ceil(self.robot_radius/self.map_settings_dict["resolution"])))
                rows.append(rr)
                cols.append(cc)

        # Returns rows and cols occupied by circles centered at each point. Each array in the returned lists corresponds to a single point
        return rows, cols


    #Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function

    #RRT* specific functions
    def ball_radius(self):
        #Close neighbor distance
        card_V = len(self.nodes)
        return min(self.gamma_RRT * (np.log(card_V) / card_V ) ** (1.0/2.0), self.epsilon)

    def connect_node_to_point(self, node_i, point_f):
        #Given two nodes find the non-holonomic path that connects them
        #Settings
        #node is a 3 by 1 node
        #point is a 2 by 1 point
        print("TO DO: Implement a way to connect two already existing nodes (for rewiring).")
        return np.zeros((3, self.num_substeps))

    def cost_to_come(self, trajectory_o):
        #The cost to get to a node from lavalle
        print("TO DO: Implement a cost to come metric")
        return 0

    def update_children(self, node_id):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        print("TO DO: Update the costs of connected nodes after rewiring.")
        return

    #Planner Functions
    def rrt_planning(self):
        '''
        Args:
            - None
        Returns:
            - NodeCollection object of built RRT structure
        '''
        def collision_free(trajectory):
            # note that 0 means we cannot occupy that cell
            # and 1 means that we are able to occupy it i.e.
            # if all 1, then trajectory is valid
            #TODO: Got index out of bounds error
            # Need to ask aditya if there is an off by
            # one error for the points_to_robot_circle function
            oc_rows, oc_cols = self.points_to_robot_circle(trajectory[:2,:])
            oc_cells = self.occupancy_map[oc_rows, oc_cols]
            return (oc_cells == 0).any()

        #This function performs RRT on the given map and robot
        for i in range(10):
            #Sample map space
            sample = self.sample_map_space()

            #Get the closest point
            closest_node_id = self.closest_node(sample)

            #Simulate driving the robot towards the closest point
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, sample)

            #Check for collisions
            if collision_free(trajectory_o):
                '''
                for i, point in enumerate(trajectory_o.T):
                    parent_id = len(self.nodes) if i > 0 else closest_node_id
                    parent = self.nodes[parent_id]
                    cost = np.sqrt(np.sum(np.square(parent.point[:2,:] - point[:2,:])))
                    self.nodes.append(Node(point.T, parent_id, cost))

                if np.sqrt(np.sum(np.square(sample[:2] - point[:2]))) < 0.1:
                    print("Reached")
                '''

                point = trajectory_o.T[None,-1, :]
                parent_id = closest_node_id
                parent = self.nodes[parent_id]
                cost = np.sqrt(np.sum(np.square(parent.point[:2,:] - point[:2,:]))) + parent.cost
                self.nodes.append(Node(point.T, parent_id, cost))

                if np.sqrt(np.sum(np.square(sample[:2] - point[:2]))) < 0.1:
                    print("Reached")



            #Check if goal has been reached

        return self.nodes

    def rrt_star_planning(self):
        #This function performs RRT* for the given map and robot
        for i in range(1): #Most likely need more iterations than this to complete the map!
            #Sample
            point = self.sample_map_space()

            #Closest Node
            closest_node_id = self.closest_node(point)

            #Simulate trajectory
            print(self.nodes[closest_node_id])
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            #Check for Collision
            print("TO DO: Check for collision.")

            #Last node rewire
            print("TO DO: Last node rewiring")

            #Close node rewire
            print("TO DO: Near point rewiring")

            #Check for early end
            print("TO DO: Check for early end")
        return self.nodes

    def recover_path(self, node_id = -1):
        path = [self.nodes[node_id].point]
        current_node_id = self.nodes[node_id].parent_id
        while current_node_id > -1:
            path.append(self.nodes[current_node_id].point)
            current_node_id = self.nodes[current_node_id].parent_id
        path.reverse()
        return path

def main():
        #Set map information
        map_filename = "willowgarageworld_05res.png"
        map_setings_filename = "willowgarageworld_05res.yaml"
        #robot information
        goal_point = np.array([[10], [10]]) #m
        stopping_dist = 0.5 #m

        #RRT precursor
        path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)

        init_node = Node(np.array([[0], [0], [0]]), -1, 0)
        point_s = [-100, 100]
        #print(path_planner.bounds)
        #print(path_planner.map_shape)
        #print(path_planner.point_to_cell(np.array([[59], [30.75]])))
        print(path_planner.points_to_robot_circle(np.array([[59], [30.75]])))
        #path_planner.simulate_trajectory(init_node.point, point_s)

        nodes = path_planner.rrt_planning()
        node_path_metric = np.hstack(path_planner.recover_path())

        print(node_path_metric)
        #Leftover test functions
        np.save("shortest_path.npy", node_path_metric)


if __name__ == '__main__':
    main()
