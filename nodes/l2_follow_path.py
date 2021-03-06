#!/usr/bin/env python
from __future__ import division, print_function
import os

import numpy as np
from scipy.linalg import block_diag
from scipy.spatial.distance import cityblock
import rospy
import tf2_ros

# msgs
from geometry_msgs.msg import TransformStamped, Twist, PoseStamped
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from visualization_msgs.msg import Marker
import matplotlib.pyplot as plt

# ros and se2 conversion utils
import utils

try:
    from skimage.draw import circle as circle_jerk
    def circle(x, y, r):
        return circle_jerk(int(x), int(y), int(r))
except:
    from skimage.draw import disk
    def circle(x, y, r):
        return disk((x,y), r)

TRANS_GOAL_TOL = .1  # m, tolerance to consider a goal complete
ROT_GOAL_TOL = .3  # rad, tolerance to consider a goal complete
TRANS_VEL_OPTS = [0.0, 0.025, 0.13, 0.26]  # m/s, max of real robot is .26
ROT_VEL_OPTS = np.linspace(-1.82, 1.82, 11)  # rad/s, max of real robot is 1.82
CONTROL_RATE = 5  # Hz, how frequently control signals are sent
CONTROL_HORIZON = 0.75  # seconds. if this is set too high and INTEGRATION_DT is too low, code will take a long time to run!
INTEGRATION_DT = .025  # s, delta t to propagate trajectories forward by
COLLISION_RADIUS = 0.225  # m, radius from base_link to use for collisions, min of 0.2077 based on dimensions of .281 x .306
ROT_DIST_MULT = 15  # multiplier to change effect of rotational distance in choosing correct control
OBS_DIST_MULT = .1  # multiplier to change the effect of low distance to obstacles on a path
MIN_TRANS_DIST_TO_USE_ROT = .1 # m, robot has to be within this distance to use rot distance in cost
PATH_NAME = 'Final_path_rrt_star_20000.npy'  # saved path from l2_planning.py, should be in the same directory as this file

# here are some hardcoded paths to use if you want to develop l2_planning and this file in parallel
TEMP_HARDCODE_PATH = [[2, -0.5, 0], [2.75, -1, -np.pi/2], [2.75, -4, -np.pi/2], [2, -4.4, np.pi]]  # almost collision-free
# TEMP_HARDCODE_PATH = [[2, -.5, 0], [2.4, -1, -np.pi/2], [2.45, -3.5, -np.pi/2], [1.5, -4.4, np.pi]]  # some possible collisions


class PathFollower():
    def __init__(self):
        # time full path
        self.path_follow_start_time = rospy.Time.now()

        # use tf2 buffer to access transforms between existing frames in tf tree
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(1.0)  # time to get buffer running

        # constant transforms
        self.map_odom_tf = self.tf_buffer.lookup_transform('map', 'odom', rospy.Time(0), rospy.Duration(2.0)).transform

        # subscribers and publishers
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.global_path_pub = rospy.Publisher('~global_path', Path, queue_size=1, latch=True)
        self.local_path_pub = rospy.Publisher('~local_path', Path, queue_size=1)
        self.collision_marker_pub = rospy.Publisher('~collision_marker', Marker, queue_size=1)

        # map
        map_message = rospy.wait_for_message('/map', OccupancyGrid)
        self.map_np = np.array(map_message.data).reshape(map_message.info.height, map_message.info.width)
        self.map_np = self.map_np[::-1,:]
        self.map_resolution = round(map_message.info.resolution, 5)
        self.map_origin = -utils.se2_pose_from_pose(map_message.info.origin)  # negative because of weird way origin is stored
        self.map_nonzero_idxes = np.argwhere(self.map_np)

        # collisions
        self.collision_radius_pix = COLLISION_RADIUS / self.map_resolution
        self.collision_marker = Marker()
        self.collision_marker.header.frame_id = '/map'
        self.collision_marker.ns = '/collision_radius'
        self.collision_marker.id = 0
        self.collision_marker.type = Marker.CYLINDER
        self.collision_marker.action = Marker.ADD
        self.collision_marker.scale.x = COLLISION_RADIUS * 2
        self.collision_marker.scale.y = COLLISION_RADIUS * 2
        self.collision_marker.scale.z = 1.0
        self.collision_marker.color.g = 1.0
        self.collision_marker.color.a = 0.5

        # transforms
        self.map_baselink_tf = self.tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0), rospy.Duration(2.0))
        self.pose_in_map_np = np.zeros(3)
        self.pos_in_map_pix = np.zeros(2)
        self.update_pose()

        # path variables
        cur_dir = os.path.dirname(os.path.realpath(__file__))

        # to use the temp hardcoded paths above, switch the comment on the following two lines
        self.path_tuples = np.load(os.path.join(cur_dir, PATH_NAME)).T
        # self.path_tuples = np.array(TEMP_HARDCODE_PATH)

        self.path = utils.se2_pose_list_to_path(self.path_tuples, 'map')
        self.global_path_pub.publish(self.path)

        # goal
        self.cur_goal = np.array(self.path_tuples[0])
        self.cur_path_index = 0

        # trajectory rollout tools
        # self.all_opts is a Nx2 array with all N possible combinations of the t and v vels, scaled by integration dt
        self.all_opts = np.array(np.meshgrid(TRANS_VEL_OPTS, ROT_VEL_OPTS)).T.reshape(-1, 2)

        # if there is a [0, 0] option, remove it
        all_zeros_index = (np.abs(self.all_opts) < [0.001, 0.001]).all(axis=1).nonzero()[0]
        if all_zeros_index.size > 0:
            self.all_opts = np.delete(self.all_opts, all_zeros_index, axis=0)
        self.all_opts_scaled = self.all_opts * INTEGRATION_DT

        self.num_opts = self.all_opts_scaled.shape[0]
        self.horizon_timesteps = int(np.ceil(CONTROL_HORIZON / INTEGRATION_DT))

        self.rate = rospy.Rate(CONTROL_RATE)

        rospy.on_shutdown(self.stop_robot_on_shutdown)
        self.follow_path()

    def trajectory_rollout(self, init_x, init_y, init_theta, vel, rot_vel):
        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions

        # Aditya Saigal
        sample_times = np.linspace(INTEGRATION_DT, CONTROL_HORIZON + INTEGRATION_DT, self.horizon_timesteps)

        if(np.abs(rot_vel) > 1e-4):
                # When a non-zero angular velocity is applied
                # The below applies in the robot ego frame
                xego = (vel/rot_vel)*np.sin(rot_vel*sample_times)
                yego = (vel/rot_vel)*(1 - np.cos(rot_vel*sample_times))

                # Rotate and Translate to world frame
                x = xego*np.cos(init_theta) - yego*np.sin(init_theta) + init_x
                y = xego*np.sin(init_theta) + yego*np.cos(init_theta) + init_y
        else:
                # Only a linear velocity is applied
                # Do this in world frame directly
                x = (vel)*np.cos(init_theta)*sample_times + init_x
                y = (vel)*np.sin(init_theta)*sample_times + init_y
        theta = (rot_vel*sample_times + init_theta) % np.pi*2
        theta = theta + (theta > np.pi) * (-2*np.pi) + (theta < -np.pi)*(2*np.pi)

        return np.vstack((x, y, theta))

    def point_to_cell(self, point):
        #Convert a series of [x,y] points in the map to the indices for the corresponding cell in the occupancy map
        #point is a 2 by N matrix of points of interest

        #Aditya Saigal
        # return pixel coordinates/indices in (row, col) format

        row = self.map_np.shape[0] - (point[1] + self.map_origin[1])/self.map_resolution # Find y coordinate wrt lower left corner. Subtract from number of rows in map
        col = (point[0] + self.map_origin[0])/self.map_resolution # Find x coordinate wrt lower left corner. This is the column

        return np.vstack((row, col)) # return indices for all points

    def points_to_robot_circle(self, points):
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function
        #Aditya Saigal

        # Expects points in x, y format
        map_coords = self.point_to_cell(points) # convert points to map indices (row col)
        rows, cols = [], []
        for pt in range(map_coords.shape[1]):
            # Get occupancy footprint for each point and store the occupied rows and columns
            rr, cc = circle(map_coords[0, pt], map_coords[1, pt], np.ceil(COLLISION_RADIUS/self.map_resolution))

            rr = np.clip(rr, 0, self.map_np.shape[0] - 1)
            cc = np.clip(cc, 0, self.map_np.shape[1] - 1)

            # Remove duplicates after clipping
            rr, cc = zip(*set(zip(rr, cc)))

            rows.append(np.array(rr))
            cols.append(np.array(cc))

        # Returns rows and cols occupied by circles centered at each point. Each array in the returned lists corresponds to a single point
        return rows, cols

    def collision_detected(self, trajectory):

        oc_rows, oc_cols = self.points_to_robot_circle(trajectory[:2,:])
        rows = np.hstack(oc_rows)
        cols = np.hstack(oc_cols)
        oc_cells = self.map_np[rows, cols]
        return (oc_cells == 100).any()

    def follow_path(self):
        while not rospy.is_shutdown():
            # timing for debugging...loop time should be less than 1/CONTROL_RATE
            tic = rospy.Time.now()

            self.update_pose()
            self.check_and_update_goal()

            # start trajectory rollout algorithm
            local_paths = np.zeros([self.horizon_timesteps + 1, self.num_opts, 3])
            local_paths[0] = np.atleast_2d(self.pose_in_map_np).repeat(self.num_opts, axis=0)

            # print("TO DO: Propogate the trajectory forward, storing the resulting points in local_paths!")
            for i in range(self.num_opts):
                curr_opt = self.all_opts[i, :]
                local_paths[1:, i, :] = self.trajectory_rollout(self.pose_in_map_np[0], self.pose_in_map_np[1], self.pose_in_map_np[2], curr_opt[0], curr_opt[1]).T


            # check all trajectory points for collisions
            # first find the closest collision point in the map to each local path point
            local_paths_pixels = (self.map_origin[:2] + local_paths[:, :, :2]) / self.map_resolution
            valid_opts = []
            local_paths_lowest_collision_dist = np.ones(self.num_opts) * 50

            for opt in range(self.num_opts):
                trajectory = local_paths[:, opt, :].T
                if (not self.collision_detected(trajectory)):
                    # Keep this option if there is no collision
                    valid_opts += [opt]

            # remove trajectories that were deemed to have collisions
            local_paths = local_paths[:, valid_opts, :]

            # calculate final cost and choose best option
            Euclidean_dist = np.linalg.norm(local_paths[-1, :, :2] - self.cur_goal.reshape(1, 3)[:, :2], axis=1)

            final_cost = Euclidean_dist
            curr_dist_from_goal = np.linalg.norm(self.pose_in_map_np[:2] - self.cur_goal[:2])
            if (curr_dist_from_goal < MIN_TRANS_DIST_TO_USE_ROT):
                # abs_angle_diff = np.abs(local_paths[-1, :, 2] - self.cur_goal.reshape(1, 3)[:, 2]) % 2*np.pi
                # rot_dist_from_goal = np.minimum(2*np.pi - abs_angle_diff, abs_angle_diff)
                abs_angle_diff_1 = np.abs(local_paths[-1, :, 2] - self.cur_goal.reshape(1, 3)[:, 2])
                abs_angle_diff_2 = np.abs(self.cur_goal.reshape(1, 3)[:, 2] - local_paths[-1, :, 2])
                # abs_angle_diff = np.minimum(abs_angle_diff_1, abs_angle_diff_2) % 2*np.pi
                rot_dist_from_goal = np.minimum(abs_angle_diff_1, abs_angle_diff_2) # np.minimum(2*np.pi - abs_angle_diff, abs_angle_diff)
                final_cost += ROT_DIST_MULT*rot_dist_from_goal

            '''
            ind = 0
            for valid, cost in zip(local_paths[-1, :, :2], final_cost):

                print(valid, end="/")
                print(self.all_opts[valid_opts[ind]], end="/")
                print(cost, end='/')
                print(ind)
                ind+=1
            '''


            if final_cost.size == 0:  # hardcoded recovery if all options have collision
                control = [-.1, 0]
            else:
                final_cost_min_index = final_cost.argmin()
                best_opt = valid_opts[final_cost_min_index]
                control = self.all_opts[best_opt]
                self.local_path_pub.publish(utils.se2_pose_list_to_path(local_paths[:, final_cost_min_index], 'map'))


            # send command to robot
            self.cmd_pub.publish(utils.unicyle_vel_to_twist(control))

            # uncomment out for debugging if necessary
            print("Selected control: {control}, Loop time: {time}, Max time: {max_time}".format(
                 control=control, time=(rospy.Time.now() - tic).to_sec(), max_time=1/CONTROL_RATE))

            self.rate.sleep()

    def update_pose(self):
        # Update numpy poses with current pose using the tf_buffer
        self.map_baselink_tf = self.tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0)).transform
        self.pose_in_map_np[:] = [self.map_baselink_tf.translation.x, self.map_baselink_tf.translation.y,
                                  utils.euler_from_ros_quat(self.map_baselink_tf.rotation)[2]]
        self.pos_in_map_pix = (self.map_origin[:2] + self.pose_in_map_np[:2]) / self.map_resolution
        self.collision_marker.header.stamp = rospy.Time.now()
        self.collision_marker.pose = utils.pose_from_se2_pose(self.pose_in_map_np)
        self.collision_marker_pub.publish(self.collision_marker)

    def check_and_update_goal(self):
        # iterate the goal if necessary
        dist_from_goal = np.linalg.norm(self.pose_in_map_np[:2] - self.cur_goal[:2])
        abs_angle_diff = np.abs(self.pose_in_map_np[2] - self.cur_goal[2])
        rot_dist_from_goal = min(np.pi * 2 - abs_angle_diff, abs_angle_diff)
        if dist_from_goal < TRANS_GOAL_TOL and rot_dist_from_goal < ROT_GOAL_TOL:
            rospy.loginfo("Goal {goal} at {pose} complete.".format(
                    goal=self.cur_path_index, pose=self.cur_goal))
            if self.cur_path_index == len(self.path_tuples) - 1:
                rospy.loginfo("Full path complete in {time}s! Path Follower node shutting down.".format(
                    time=(rospy.Time.now() - self.path_follow_start_time).to_sec()))
                rospy.signal_shutdown("Full path complete! Path Follower node shutting down.")
            else:
                self.cur_path_index += 1
                self.cur_goal = np.array(self.path_tuples[self.cur_path_index])
        else:
            rospy.logdebug("Goal {goal} at {pose}, trans error: {t_err}, rot error: {r_err}.".format(
                goal=self.cur_path_index, pose=self.cur_goal, t_err=dist_from_goal, r_err=rot_dist_from_goal
            ))

    def stop_robot_on_shutdown(self):
        self.cmd_pub.publish(Twist())
        rospy.loginfo("Published zero vel on shutdown.")


if __name__ == '__main__':
    try:
        rospy.init_node('path_follower', log_level=rospy.DEBUG)
        pf = PathFollower()
    except rospy.ROSInterruptException:
        pass
