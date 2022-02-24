#!/usr/bin/env python
#Standard Libraries
import heapq as hq
from enum import Enum
from tqdm import tqdm
import numpy as np
import yaml
import pygame
import matplotlib.image as mpimg
import time
from scipy.linalg import block_diag
from scipy.spatial import KDTree

class Extend(Enum):
    FAILED = 2
    SUCCESS = 0
    HITGOAL = 1

try:
    # pygame_utils file doesn't have correct magic number i.e. wont run locally
    import pygame_utils
except:
    import matplotlib.pyplot as plt

try:
    from skimage.draw import circle as circle_jerk
    def circle(x, y, r):
        return circle_jerk(int(x), int(y), int(r))
except:
    from skimage.draw import disk
    def circle(x, y, r):
        return disk((x,y), r)

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
    def __init__(self, point, parent_id, cost, heuristic=0):
        self.point = point # A 3 by 1 vector [x, y, theta]
        self.parent_id = parent_id # The parent node id that leads to this node (There should only every be one parent in RRT)
        self.cost = cost # The cost to come to this node
        self.full_cost = cost + 5*heuristic # A_start like total cost function used for sampling priority, weight exploration more
        self.children_ids = [] # The children node ids of this node
        self.num_fails = 0 #number of times we failed to sample a good point from this node
        return

    def __repr__(self):
        return f"{self.full_cost}"
    def __lt__(self, other_inst):
        return self.full_cost < other_inst.full_cost

    def __hash__(self):
        '''
        Args:
            - none
        Returns:
            - hashed value based on points converted to tuple
        '''
        return hash(tuple(self.point[:2,0]))

class NodeCollection:
    def __init__(self):
        self.list = []

        # This is for local sampling
        # maintain frontier so that we only search alive nodes
        # made with priorty queue of the alive nodes
        # by default this is a min heap
        self.frontier = []

        # needed for kdtree, need features along cols, and points along rows
        # n.b. we only keep the [x, y] points in val_arr
        self.val_arr = None
        self.kdtree = None

    def get_sample_center(self):
        # used for sampling, get
        # the node we should sample near based on full_cost (ctc + heuristic)
        if len(self.frontier) == 0:
            return None
        node = self.frontier[0]
        # Keep no matter what
        if self.frontier[0].num_fails == 10:
            hq.heappop(self.frontier)

        return node

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
        self.list.append(node)
        hq.heappush(self.frontier, node)

        if self.val_arr is None:
            self.val_arr = node.point[:2,:] if node.point.shape[1]==3 else node.point.T[:,:2]
        else:
            new_point = node.point if node.point.shape[1]==3 else node.point.T
            self.val_arr = np.vstack([self.val_arr, new_point[:,:2]])

        # update children ids of parent
        if node.parent_id != -1:
            self[node.parent_id].children_ids.append(len(self)-1)

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
        self.nodes.append(Node(np.zeros((3,1)), -1, 0, np.sqrt(np.sum(np.square(self.goal_point)))))

        #RRT* Specific Parameters
        self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] **2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * (self.lebesgue_free / self.zeta_d) ** (1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 2.5

        # Error threshold for Trajectory and Check
        self.error_thresh = 1e-2

        #Pygame window for visualization
        # self.window = pygame_utils.PygameWindow(
        #    "Path Planner", (1000, 1000), self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist)
        return

    #Functions required for RRT
    def sample_map_space(self, region=np.array([[0,0], [1600, 1600]]), rep="", center=None):
        '''
        Args:
            - region is an array of two points [[r,c],[r,c]] in map/cell frame
            region[0,0] - top left row
            region[0,1] - top left col
            region[1,0] - bot right row
            region[1,1,] - bot right col
            - rep:
            "polar" - sample in between a donout around the live nodes
            default - cartesian sample within provided box
            - center array of 2x1 [x,y] used in polar, use this to pass in the location of node frontier
        Returns:
            - gives random point within map space given some region, point is [x, y] numpy array
        TODO: add support for regions
        '''

        sample = np.random.rand(2,1)
        if rep == "polar" and center is not None:
            rmin, rmax = region[0, :]
            radius = rmin + sample[0,0]*(rmax - rmin)
            theta = 2*np.pi*sample[1,0]

            sample = np.array([
                [radius*np.cos(theta) + center[0]],
                [radius*np.sin(theta) + center[1]]
            ])

        else:
            if center is None and rep =='polar':
                # Fall back
                region = np.array([[350, 400],
                                   [1600, 1300]])

            sample[0,0] = region[0,0] + sample[0,0]*(region[1,0] - region[0,0])
            sample[1,0] = region[0,1] + sample[1,0]*(region[1,1] - region[0,1])
            sample = self.cell_to_point(sample)

        return sample

    def check_if_duplicate(self, point):
        '''
        Args:
            - point - Node object, check to see if this Node object exists within current list 3x1
        Returns:
            - bool True if in NodeCollection otherwise no
        '''
        dist, ind = self.nodes.query(point, 1)

        return dist[0] < 0.2

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

    def simulate_trajectory(self, node_i, world_sample):
        #Simulates the non-holonomic motion of the robot.
        #This function drives the robot from node_i towards point_s. This function does has many solutions!
        #node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{OI} in course notation
        #point_s is the sampled point vector [x; y] (2x1)

        # Aditya Saigal
        # expects point_s to have (x, y) map coordinates, not (row, col)
        # node_i is given in the world frame coordinates
        # Iteratively generate control commands and simulate a trajectory until point_s is reached on the map.

        curr_point = node_i
        # convert the map coordinates to world coordinates

        #print(curr_point)
        #print(world_sample)
        #exit()
        complete_traj = None
        count = 0
        while(np.linalg.norm(curr_point[0:2] - world_sample) > self.error_thresh):
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
        """
        tr, tc = self.point_to_cell(complete_traj)
        plt.imshow(self.occupancy_map)
        plt.plot(tc, tr)
        plt.show()
        """
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

    def cell_to_point(self, cell):
        # Convert a series of [x,y] cells in the cell occupancy indicies to the map frame
        # point is a 2 by N matrix of points of interest
        # return pixel coordinates/indices in (row, col) format

        py = -(cell[0,:] - self.map_shape[0])*self.map_settings_dict["resolution"] + self.map_settings_dict["origin"][1]
        px = cell[1,:]*self.map_settings_dict["resolution"] + self.map_settings_dict["origin"][0]

        return np.vstack((px, py)) # return indices for all points

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
        #print(map_coords)
        #print(self.robot_radius/self.map_settings_dict["resolution"])
        for pt in range(map_coords.shape[1]):
            # Get occupancy footprint for each point and store the occupied rows and columns
            rr, cc = circle(map_coords[0, pt], map_coords[1, pt], np.ceil(self.robot_radius/self.map_settings_dict["resolution"]))

            rr = np.clip(rr, 0, self.map_shape[0] - 1)
            cc = np.clip(cc, 0, self.map_shape[1] - 1)

            # Remove duplicates after clipping
            rr, cc = zip(*set(zip(rr, cc)))

            rows.append(np.array(rr))
            cols.append(np.array(cc))

        # Returns rows and cols occupied by circles centered at each point. Each array in the returned lists corresponds to a single point
        return rows, cols


    #Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function

    #RRT* specific functions
    def ball_radius(self):
        #Close neighbor distance
        card_V = len(self.nodes)
        return min(self.gamma_RRT * (np.log(card_V) / card_V ) ** (1.0/2.0), self.epsilon)

    def connect_node_to_point(self, node_i, world_sample):
        #Given two nodes find the non-holonomic path that connects them
        #Settings
        #node is a 3 by 1 node (with coordinates in the world space)
        #world_sample is a 2 by 1 point (with coordinates [x, y] in world sapce)

        # Aditya Saigal
        # Function used to find a connection between newly sampled point and an existing node in the nearest neighbor list.
        # Find the straight linear trajectory between the 2 points, as expressed in the world frame. If a line exists, then the heading can be adjusted to go from one node to the other.
        # Use this for collision detection.

        # Generate points between the 2 landmarks
        xs = np.linspace(node_i[0], world_sample[0], self.num_substeps + 2)
        ys = np.linspace(node_i[1], world_sample[1], self.num_substeps + 2)
        thetas = np.array([np.arctan2(point_f[1] - node_i[1], point_f[0] - node_i[0])]*(self.num_substeps + 2)).reshape((12,))

        # Return sampled points on the trajectory in the world frame. Use previous collision detection functions to see if a collision free path exists
        return np.vstack((xs, ys, thetas))

    def cost_to_come(self, trajectory_o):
        #The cost to get to a node from lavalle
        dist_traveled=np.linalg.norm(trajectory_o[:2,0] - trajectory_o[:2,-1])
        return dist_traveled

    def node_cost_to_come(self, node_id_1,  node_id_2):
        # Calculate the l2-norm distance between two node objects
        dist_traveled = np.linalg.norm(self.nodes[node_id_1].point[:2, 0] -self.nodes[node_id_2].point[:2,0])
        return dist_traveled

    def update_children(self, node_id, old_cost):
        '''
        Args:
            node_id - int the id of the node that was changed, i.e some parent that
            was rewired
            old_cost - float, the old cost to come to node_id, before rewiring
        Returns:
            Nothing - update happens silently
        '''
        #Given a node_id with a changed cost, update all connected nodes with the new cost

        ids_update_required = self.nodes[node_id].children_ids[:]
        new_cost = self.nodes[node_id].cost
        while len(ids_update_required) > 0:
            problem_child = ids_update_required.pop(0)
            self.nodes[problem_child].cost += new_cost - old_cost
            ids_update_required += self.nodes[problem_child].children_ids
        return True

    def collision_detected(self, trajectory):
        # note that 0 means we cannot occupy that cell
        # and 1 means that we are able to occupy it i.e.
        # if all 1, then trajectory is valid
        #TODO: Got index out of bounds error
        # returns true if there is an obstacle in the way

        #returns True if trajectory DNE
        if trajectory is None:
            return True
        oc_rows, oc_cols = self.points_to_robot_circle(trajectory[:2,:])
        rows = np.hstack(oc_rows)
        cols = np.hstack(oc_cols)
        oc_cells = self.occupancy_map[rows, cols]
        return (oc_cells == 0).any()

    #Planner Functions
    def rrt_planning(self, num_samples=1000, region=None, rep=""):
        '''
        Args:
            - None
        Returns:
            - NodeCollection object of built RRT structure
        '''
        def extend(sample, _remove=[]):
            if self.check_if_duplicate(sample):
                return Extend.FAILED

            #Get the closest point
            closest_node_id = self.closest_node(sample)

            #Simulate driving the robot towards the closest point
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, sample)

            #Check for collisions
            if not self.collision_detected(trajectory_o):
                point = trajectory_o.T[-1, :,None]
                parent_id = closest_node_id
                parent = self.nodes[parent_id]
                cost = np.linalg.norm(parent.point[:2, 0] - point[:2, 0]) + parent.cost
                self.nodes.append(Node(point, parent_id, cost, np.linalg.norm(point[:2, 0] - self.goal_point[:2, 0])))

                _remove.append(sample)
                if np.sqrt(np.sum(np.square(point[:2, 0] - self.goal_point[:2, 0]))) < self.stopping_dist:
                    return Extend.HITGOAL
                return Extend.SUCCESS
            else:
                return Extend.FAILED

        _remove = []
        #This function performs RRT on the given map and robot
        for _ in tqdm(range(num_samples)):
            #Sample map space
            sample_anchor = self.nodes.get_sample_center()
            anchor = sample_anchor if sample_anchor is None else sample_anchor.point[:2, 0]
            sample = self.sample_map_space(region, "polar", sample_anchor.point[:2,0] ) if rep == "polar" else self.sample_map_space(region)
            ret_val = extend(sample, _remove)
            if ret_val == Extend.FAILED:
                sample_anchor.num_fails += 1
            if ret_val == Extend.HITGOAL:
                break

        # Slightly different, do extend with goal points
        # If need to use RRT later, remove the below code
        extend(self.goal_point, _remove)
        goal_cell = self.point_to_cell(self.goal_point)
        return self.nodes, _remove

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

        pt_s = np.array([[0], [0], [0]])
        pt_f = np.array([[10], [15]])


        #print(path_planner.bounds)
        #print(path_planner.map_shape)
        #print(path_planner.point_to_cell(np.array([[59, 21], [30.75, 21]])))


        #path_planner.simulate_trajectory(init_node.point, point_s)

        nodes = path_planner.rrt_star_planning()
        node_path_metric = np.hstack(path_planner.recover_path())

        #Leftover test functions
        np.save("shortest_path.npy", node_path_metric)

if __name__ == '__main__':
    main()
