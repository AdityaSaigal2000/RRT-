#!/usr/bin/env python
#Standard Libraries
import numpy as np
import yaml
import pygame
import time
import pygame_utils
import matplotlib.image as mpimg

RUNNING_ON_LAB = True

if RUNNING_ON_LAB:
	from skimage.draw import circle
else:
	from skimage.draw import disk

from scipy.linalg import block_diag
import time

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
        return

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
        self.nodes = [Node(np.zeros((3,1)), -1, 0)]

        #RRT* Specific Parameters
        self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] **2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * (self.lebesgue_free / self.zeta_d) ** (1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 2.5
        
        #Pygame window for visualization
        self.window = pygame_utils.PygameWindow(
            "Path Planner", (1000, 1000), self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist)
        return

    #Functions required for RRT
    def sample_map_space(self):
        #Return an [x,y] coordinate to drive the robot towards
        print("TO DO: Sample point to drive towards")
        return np.zeros((2, 1))
    
    def check_if_duplicate(self, point):
        #Check if point is a duplicate of an already existing node
        print("TO DO: Check that nodes are not duplicates")
        return False
    
    def closest_node(self, point):
        #Returns the index of the closest node
        print("TO DO: Implement a method to get the closest node to a sapled point")
        return 0
    
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
	#		print(robot_traj)
		robot_traj = np.vstack((curr_point[0] + robot_traj[0], curr_point[1] + robot_traj[1], curr_point[2] + robot_traj[2]))
#		if(not w):
#			count += 1
	#		print("\n\n")
	#		print(robot_traj)
	#		print("\n")
#			if(count == 110):
#				exit()
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
	#print(map_coords)
	#print(self.robot_radius/self.map_settings_dict["resolution"])
	for pt in range(map_coords.shape[1]):
		# Get occupancy footprint for each point and store the occupied rows and columns
		if(RUNNING_ON_LAB):
			rr, cc = circle(int(map_coords[0, pt]), int(map_coords[1, pt]), int(np.ceil(self.robot_radius/self.map_settings_dict["resolution"])))
		else:
			rr, cc = disk(map_coords[:, pt], self.robot_radius/self.map_settings_dict["resolution"])
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
    
    def connect_node_to_point(self, node_i, point_f):
        #Given two nodes find the non-holonomic path that connects them
        #Settings
        #node is a 3 by 1 node (with coordinates in the world space)
        #point is a 2 by 1 point (with coordinates [x, y] in the map frame)
        
	# Aditya Saigal
	# Function used to find a connection between newly sampled point and an existing node in the nearest neighbor list.
	# Find the straight linear trajectory between the 2 points, as expressed in the world frame. If a line exists, then the heading can be adjusted to go from one node to the other.
	# Use this for collision detection.

	# Convert sample point to world frame

	world_sample = np.array([[self.map_settings_dict["origin"][0] + point_f[0][0]*self.map_settings_dict["resolution"]], [self.map_settings_dict["origin"][1] + point_f[1][0]*self.map_settings_dict["resolution"]]]) # 2x1 vector

	# Generate points between the 2 landmarks
	xs = np.linspace(node_i[0], world_sample[0], self.num_substeps + 2)
	ys = np.linspace(node_i[1], world_sample[1], self.num_substeps + 2)
	thetas = np.array([np.arctan2(point_f[1] - node_i[1], point_f[0] - node_i[0])]*(self.num_substeps + 2)).reshape((12,)) 
	
	# Return sampled points on the trajectory in the world frame. Use previous collision detection functions to see if a collision free path exists
	return np.vstack((xs, ys, thetas))
    
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
        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
        for i in range(1): #Most likely need more iterations than this to complete the map!
            #Sample map space
            point = self.sample_map_space()

            #Get the closest point
            closest_node_id = self.closest_node(point)

            #Simulate driving the robot towards the closest point
            print(self.nodes[closest_node_id])
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            #Check for collisions
            print("TO DO: Check for collisions and add safe points to list of nodes.")
            
            #Check if goal has been reached
            print("TO DO: Check if at goal point.")
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
