import math

import torch
from pytorch_mppi import MPPI

import rclpy
import rclpy.time
from geometry_msgs.msg import TwistStamped, PoseStamped, Point, Quaternion
from rclpy.node import Node

from tf2_msgs.msg import TFMessage

from stretch_mppi.tf2_wrapper import TF2Wrapper
from stretch_mppi.vis_utils import VisualizationUtils, DataProcessor

from stretch_mppi.controller_config import *

from geometry_msgs.msg import TwistStamped, Twist
from nav_msgs.msg import Odometry
from skeleton_interfaces.msg import Predictions
#from skeleton_interfaces.srv import GetTrajectory
from cohan_msgs.srv import GetTrajectory
from rclpy_message_converter import message_converter

from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.vectorized import contains

from cohan_msgs.srv import GetTrajectory
from cohan_msgs.msg import AgentPredictionArray, AgentPrediction
from rclpy_message_converter import message_converter

import tf_transformations

import numpy as np

from stretch_mppi.utils import dynamics, normalize_angle, save_data
from stretch_mppi.cv import CV

import time
import os

import roslibpy
import pickle

from pathlib import Path

import json

class MPPI_node(Node):
    def __init__(self) -> None:
        super().__init__("stretch_mppi")
        self.device = DEVICE

        if not os.path.exists(os.path.join(DATA_DIR, DATA_TIME)):
            os.mkdir(os.path.join(DATA_DIR, DATA_TIME))

        dir_path = os.path.join(DATA_DIR, DATA_TIME + '/pkl/')
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        if JSON:
            DATA_PATH = os.path.join(dir_path, METHOD + '.jsonl')
        else:
            DATA_PATH = os.path.join(dir_path, METHOD + '.pkl')

        cov = torch.eye(2, dtype=torch.float32)
        cov[0, 0] = 2
        cov[1, 1] = 2

        self.predictions_json = open('/home/socnav/frb_study_data/12_03_2026_00_00/json/cohan.json', 'r+')
        self.hst_pkl = open('/home/socnav/frb_study_data/12_03_2026_00_00/json/cohan_hst_predictions.pkl')
        OFFLINE = True
        self.offline_agent_states, self.offline_robot_states, self.offline_predictions, self.offline_robot_predictions, self.offline_logits, self.offline_robot_goals, self.offline_hst_agent_predictions, self.offline_hst_robot_predictions = self.load_offline(self.predictions_json, self.hst_pkl)
        self.offline_out = '/home/socnav/frb_study_data/12_03_2026_00_00/json/max_rollouts.pkl'
        predictions = np.full((1, 1, WINDOW_LENGTH, ACTIVE_AGENTS, 2), 9999.9)
        logits = np.full((MODES), 1/MODES)
        self.predictions = predictions[:,:,HISTORY_LENGTH+1:,:,:]
        self.robot_prediction = self.predictions[:,:,:,0,None,:]
        self.predictions = torch.Tensor(self.predictions).to(self.device)
        self.logits = torch.Tensor(logits).to(self.device)
        self.robot_prediction = torch.Tensor(self.robot_prediction).to(self.device)
        self.num_agents = MAX_AGENT_NUM
        self.mppi = MPPI(
            self.dynamics,
            self.cost,
            3,
            cov,
            num_samples=NUM_SAMPLES,
            device=self.device,
            terminal_state_cost=self.terminal_cost,
            u_min=torch.tensor([0.0, -1], dtype=torch.float32),
            u_max=torch.tensor([1.0, 1], dtype=torch.float32),
            step_dependent_dynamics=True,
            horizon=PREDICTION_LENGTH-1,
        )

        self.tf2_wrapper = TF2Wrapper(self)
        self.vis_utils = VisualizationUtils(self)

        self.cmd_vel_pub = self.create_publisher(TwistStamped, "/stretch/cmd_vel", 1)
        self.vel = 0.0
        self.model_predictor = CV()
        self.model_predictor.set_params()

        self.goals = GOALS.to(self.device)
        self.goal_indices = GOAL_INDICES.to(self.device)
        self.index = 0
        self.goal_index = self.goal_indices[GOAL_SEQUENCE][self.index]
        self.initial_orientation = None
        self.turning_to_center = True

        self.counter = 0
        self.avg_linear_accel = 0.0
        self.avg_angular_accel = 0.0
        self.states = list()
        self.dt = HZ

        self.data_processor = DataProcessor(DATA_PATH, True)

        self.rollouts = torch.zeros((7, NUM_SAMPLES, 2))
        self.s2_ego = torch.zeros((NUM_SAMPLES, 3)).to(self.device)

        self.fake_predictions = torch.full((1, MAX_AGENT_NUM, WINDOW_LENGTH, 1, 2), 15).to(self.device)
        self.fake_logits = torch.tensor([1]).to(self.device)

        static_obs = self.model_predictor.construct_boundary(ROBOT_BOUNDARY[0], ROBOT_BOUNDARY[1], ROBOT_BOUNDARY[2], ROBOT_BOUNDARY[3])

        self.polygons = [Polygon(obs) for obs in static_obs]
        self.multi_polygon = MultiPolygon(self.polygons)
        self.bounds = self.multi_polygon.bounds

        if not COHAN:
            self._subscriber = self.create_subscription(
                Predictions, PREDICTION_TOPIC, self._prediction_callback, 5
            )
        if OFFLINE:
            self.timer = self.create_timer(HZ, self.timer_callback)
        else:
            self.timer = self.create_timer(HZ, self.timer_callback_offline)

        if NEED_ODOM:
            self._odom_subscriber = self.create_subscription(
                Odometry, '/odom', self._odometry_callback, 5
            )

        if NEED_LASER:
            self.laser_timer = self.create_timer(0.01, self.laser_callback)
            self.laser_pub = self.create_publisher(TFMessage, "/tf", 1)
            self.odom_pose = torch.tensor([0, 0, 0], dtype=torch.float32) 
        self.pose = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

        self.time = time.time()
        if USE_COHAN:
            self.client = roslibpy.Ros(host='localhost', port=9090)
            self.client.run()
            self.service = roslibpy.Service(self.client, '/get_trajectories', 'cohan_msgs/srv/GetTrajectory')
            self.cohan_timer = self.create_timer(0.01, self.cohan_timer)
        self.plan = None
        self.future = None
        self.cohan_velocity = Twist()
        self.robot_trajectory = None

        self.prev_agent_states = list()
        self.cv_predictions = None
        self.cv_robot_prediction = None
        self.cv_logits = None
        self.cohan_time = time.time()
        self.cohan_velocity = Twist()

        self.prediction_time = 0.0
        self.robot_predictions = []
        self.raw_predictions = None
        self.time_since_last_goal_sent = time.time()

        if COHAN:
            self.goal_pub = self.create_publisher(PoseStamped, "/stretch/nav_goal", 1)
            self._cohan_human_predictions = self.create_subscription(
                    AgentPredictionArray, '/cohan_human_predictions', self._cohan_prediction_callback, 5
                )
            self._cohan_robot_predictions = self.create_subscription(
                    AgentPrediction, '/cohan_robot_predictions', self._robot_prediction_callback, 5
                )
            
    def load_offline(self, f, hst_f=None):
        robot_states = []
        agent_states = []
        robot_goals = []
        logits = []
        predictions = []
        robot_predictions = []

        hst_agent_predictions = []
        hst_robot_predictions = []

        dict_to_read = json.load(f)

        for i in range(len(dict_to_read)):
            robot_states.append(dict_to_read[i]['robot_state'])
            agent_states.append(dict_to_read[i]['agent_states'])
            robot_goals.append(dict_to_read[i]['robot_goal'])
            predictions.append(dict_to_read[i]['predictions'])
            robot_predictions.append(dict_to_read[i]['robot_prediction'])
            if hst_f is not None:
                hst_dict = pickle.load(hst_f)
                hst_agent_predictions.append(hst_dict['human'])
                hst_robot_predictions.append(hst_dict['robot'])

        robot_states = np.array(robot_states)
        agent_states = np.array(agent_states)
        robot_goals = np.array(robot_goals)
        predictions = np.array(predictions)
        logits = np.array(logits)
        robot_predictions = np.array(robot_predictions)

        hst_agent_predictions = np.array(hst_agent_predictions)
        hst_robot_predictions = np.array(hst_robot_predictions)

        return agent_states, robot_states, predictions, robot_predictions, logits, robot_goals, hst_agent_predictions, hst_robot_predictions


    def _robot_prediction_callback(self, msg):
        self.robot_predictions = []
        for pose in msg.predictions.poses:
            pr = np.array([pose.position.x, pose.position.y])
            self.robot_predictions.append(pr) 

    def interpolate_poses(self, pose1, pose2, dt, t):
        t_ratio = t / dt #dt is difference in time between poses, t is time in [0, dt] we want the interpolated pose at
        pose = pose1 + (pose2 - pose1) * t_ratio
        return pose

    def _cohan_prediction_callback(self, msg):

        raw_predictions_all_agents = []
        poses_all_agents = []
        times_all_agents = []
        if len(msg.agent_predictions) == 0:
            return

        for a in range(len(msg.agent_predictions)):
            preds = msg.agent_predictions[a].predictions.poses
            raw_predictions_all_agents.append(preds)
            poses = []
            times = []
            time_sum = 0.0
            prev_pose = None
            
            for i in range(len(preds)):
                if len(poses) == 0:
                    pr = np.array([preds[i].position.x, preds[i].position.y])
                    poses.append(pr)
                    prev_pose = pr
                    time_sum += preds[i].position.z
                else:
                    if len(poses) >= PREDICTION_LENGTH:
                        break

                    if time_sum + preds[i].position.z > len(poses) * DT:
                        t = len(poses) * DT - time_sum
                        pose = self.interpolate_poses(prev_pose, np.array([preds[i].position.x, preds[i].position.y]), preds[i].position.z, t)
                        poses.append(pose)

                    time_sum += preds[i].position.z

                prev_pose = np.array([preds[i].position.x, preds[i].position.y])

            
            num_extra_poses = PREDICTION_LENGTH - len(poses)
            for p in range(num_extra_poses):
                poses.append(poses[-1])

            predictions = torch.tensor(poses)
            predictions = predictions.unsqueeze(1) # (1, 1, PREDICTION_LENGTH, ACTIVE_AGENTS, 2)
            self.predictions[0,0,:,msg.agent_predictions[a].id-1,None,:] = predictions
            poses_all_agents.append(poses)
            
        self.logits = torch.tensor([1]).to(self.device)
        self.cv_logits = self.logits
        self.predictions = torch.tensor(self.boundary_aware_predictions(self.predictions.cpu().numpy())).to(self.device)
        self.cv_predictions = self.predictions
        self.raw_predictions = raw_predictions_all_agents

    def find_start_point(self):
        for i in range(self.tracking_trajectory.shape[0]):
            dist = torch.norm(self.tracking_trajectory[i] - self.last_state[:2])

    def laser_callback(self):
        T_map_laser = self.tf2_wrapper.get_latest_transform("map", "laser")
        if T_map_laser is not None:
            self.laser_pub.publish(TFMessage(transforms=[T_map_laser]))

    def terminal_cost(self, s, a=None):
        if not USE_TERMINAL_COST: 
            return 0

        gc = self.model_predictor.goal_cost_terminal(s.squeeze(), self.goals[self.goal_index])
        cc = self.model_predictor.collision_avoidance_cost_terminal(s.squeeze())
        if BLIND:
            oc = 0
        elif CV_PREDICTIONS:
            oc = self.model_predictor.obstacle_cost_terminal(s.squeeze(), self.cv_predictions, self.cv_logits).squeeze()
        else:
            oc = self.model_predictor.obstacle_cost_terminal(s.squeeze(), self.predictions, self.logits).squeeze()

        if COHAN_MPPI:
            return gc
        else:
            if BLIND:
                return gc + 1000 * cc
            else:
                return gc + oc + 1000 * cc

    def cost(self, s, a, t):
        if USE_TERMINAL_COST:
            return 0

        gc = self.model_predictor.goal_cost(s, a, self.goals[self.goal_index])
        oc = self.model_predictor.obstacle_cost(s, a, self.predictions, self.logits, t).squeeze()
        sc = self.model_predictor.static_cost(s, a)

        if COHAN_MPPI:
            return gc #+ tc
        else:
            return gc + oc + sc
        
    def boundary_aware_predictions(self, predictions):
       # Extract the (x, y) coordinates from the states
        #xy_coords = state[..., :2].cpu().numpy()  # Shape: (N, T', 2)
        # Flatten the coordinates for bulk processing
        #print("predictions shape ", predictions.shape)
        flattened_coords = predictions[0].reshape(-1, 2)

        # Efficient vectorized check for points within bounds
        x_min, y_min, x_max, y_max = self.bounds
        within_bounds = (
            (flattened_coords[:, 0] >= x_min) &
            (flattened_coords[:, 0] <= x_max) &
            (flattened_coords[:, 1] >= y_min) &
            (flattened_coords[:, 1] <= y_max)
        )

        # Use Shapely's vectorized `contains` for points within bounds
        collision_flags = contains(self.multi_polygon, flattened_coords[:, 0], flattened_coords[:, 1])
        collision_flags[~within_bounds] = False  # Points outside bounds are not collisions

        collision_flags = collision_flags.reshape((predictions.shape[2], predictions.shape[3]))
        for i in range(collision_flags.shape[0]):
            out_inds = np.argwhere(collision_flags[i,:])
            if i > 0:
                predictions[0,0,i,out_inds,:] = predictions[0,0,i-1,out_inds,:]

        return predictions

    def _prediction_callback(self, msg):
        predictions = np.array(msg.predictions.data)
        logits = np.array(msg.logits.data)
        num_agents = msg.num_agents.data
        predictions = np.expand_dims(np.reshape(predictions, (MAX_AGENT_NUM, WINDOW_LENGTH, MODES, 2)), axis=0)  
        robot_prediction = predictions[:,num_agents,:,:,:]
        predictions = predictions[:,:num_agents,:,:,:]
        predictions = np.transpose(predictions, (0, 3, 2, 1, 4)) #batch modes window_length max_agent_num xy
        robot_prediction = np.transpose(robot_prediction[:,None,:,:,:], (0, 3, 2, 1, 4))

        self.robot_prediction = robot_prediction[:,:,HISTORY_LENGTH+1:,:,:]
        self.predictions = predictions[:,:,HISTORY_LENGTH+1:,:,:]
        self.predictions = self.predictions - 5
        self.robot_prediction = self.robot_prediction - 5

        if HIGHEST_PROB_ONLY:
            best_ind = np.argmax(logits)
            self.predictions = self.predictions[:,best_ind,None,:,:,:]
            self.robot_prediction = self.robot_prediction[:,best_ind,None,:,:,:]
            logits = [1.]

        if USE_PREDICTIONS:
            self.predictions = self.boundary_aware_predictions(self.predictions)
            self.robot_prediction = self.boundary_aware_predictions(self.robot_prediction)
            self.predictions = torch.Tensor(self.predictions).to(self.device)
            self.logits = torch.Tensor(logits).to(self.device)
            self.robot_prediction = torch.Tensor(self.robot_prediction).to(self.device)

        self.num_agents = num_agents

        current_time = time.time()
        self.prediction_time = current_time

    def _odometry_callback(self, msg):
        roll = pitch = yaw = 0.0
        q = msg.pose.pose.orientation
        yaw = np.arctan2(2.0 * (q.z * q.w + q.x * q.y), - 1.0 + 2.0 * (q.w * q.w + q.x * q.x))
        
        self.odom_pose = torch.tensor([msg.pose.pose.position.x, msg.pose.pose.position.y, yaw], dtype=torch.float32)
        self.odom_orientation = msg.pose.pose.orientation
        if self.initial_orientation is None:
            self.initial_orientation = msg.pose.pose.orientation

    def timer_callback_offline(self) -> None:
        # Get the latest global pose
        self.counter += 1

        timer_callback_start_time = time.time()

        agent_states = torch.tensor(self.offline_agent_states[self.counter])
        prev_agent_states = torch.tensor(self.offline_agent_states[self.counter-1])
        agent_vel = (agent_states - prev_agent_states) / 0.02
        for a in range(2):
            agent_states[a,2] = torch.atan2(agent_vel[a,1], agent_vel[a,0])

        state = self.offline_robot_states[self.counter]
        state_prev = self.offline_robot_states[self.counter-1]
        vel = (state - state_prev) / dt
        state[2] = torch.atan2(vel[1] - vel[0])

        agent_states_tensor = agent_states
        state[2] = normalize_angle(state[2])
        self.states.append(state)

        if CV_PREDICTIONS:
            self.prev_agent_states.append(agent_states_tensor)
            if self.counter < 1 + int(DT/HZ):
                return
            else:
                dt = 1
                self.cv_predictions, self.cv_logits = self.model_predictor.construct_cv_predictions(self.prev_agent_states[-1], self.prev_agent_states[-1-dt], static=STATIC)
                self.cv_logits = self.cv_logits.to(self.device)
                self.cv_predictions = torch.tensor(self.boundary_aware_predictions(self.cv_predictions.cpu().numpy())).to(self.device)

                self.cv_robot_prediction, _ = self.model_predictor.construct_cv_predictions(torch.tensor(self.states[-1]).cpu()[None,:], torch.tensor(self.states[-1-dt]).cpu()[None,:], static=STATIC)
                self.cv_robot_prediction = torch.tensor(self.boundary_aware_predictions(self.cv_robot_prediction.cpu().numpy())).to(self.device)

        data_to_save = {
            'blind': None,
            'static': None,
            'cv': None,
            'hst': None,
            'cohan': None
        }

        for key in data_to_save:
            if key == 'cohan':
                self.predictions = self.offline_predictions[self.counter]
                self.robot_prediction = self.offline_robot_predictions[self.counter]
            elif key == 'hst':
                self.predictions = self.offline_hst_agent_predictions[self.counter]
                self.robot_prediction = self.offline_hst_robot_predictions[self.counter]
            elif key == 'cv':
                self.predictions = self.cv_predictions
                self.robot_prediction = self.cv_robot_prediction
            elif key == 'static':
                self.predictions, _ = self.model_predictor.construct_cv_predictions(self.prev_agent_states[-1], self.prev_agent_states[-1-dt], static=True)
                self.robot_prediction, _ = self.model_predictor.construct_cv_predictions(torch.tensor(self.states[-1]).cpu()[None,:], torch.tensor(self.states[-1-dt]).cpu()[None,:], static=True)
            else:
                predictions = np.full((1, 1, WINDOW_LENGTH, ACTIVE_AGENTS, 2), 9999.9)
                self.predictions = predictions[:,:,HISTORY_LENGTH+1:,:,:]
                self.robot_prediction = self.predictions[:,:,:,0,None,:]
            self.logits = torch.tensor([1.])
            # Get best action from MPPI
            self.rollouts = torch.zeros_like(self.rollouts)
            action = self.mppi.command(state)

            rollouts = self.mppi.states
            costs = self.mppi.cost_total

            rollouts = rollouts.squeeze(0)
            costs = costs.squeeze(0)
            max_cost = torch.argmax(costs)
            min_cost = torch.argmin(costs)
            max_rollout = rollouts[max_cost]
            min_rollout = rollouts[min_cost]

            data_to_save[key] = max_rollout

        if self.counter == 0:
            write_mod = 'wb'
        else:
            write_mod = 'ab'

        with open(self.offline_filepath, write_mod) as pickle_hd:
            pickle.dump(data_to_save, pickle_hd)


        current_time = time.time()
        self.time = current_time

    def timer_callback(self) -> None:
        # Get the latest global pose
        self.counter += 1

        timer_callback_start_time = time.time()

        if not NEED_ODOM:
            T_map_baselink = self.tf2_wrapper.get_latest_pose("map", "base_link")
            if T_map_baselink is None:
                print("Can't find robot pose")
                return
            # Convert pose to MPPI state representation
            yaw = 2.0 * math.atan2(
                T_map_baselink.rotation.z, T_map_baselink.rotation.w
            )  # NOTE: assuming roll and pitch are negligible
            state = torch.tensor(
                [
                    T_map_baselink.translation.x,
                    T_map_baselink.translation.y,
                    yaw,
                ],
                dtype=torch.float32,
            ).to(self.device)
        else: 
            state = self.odom_pose

        agent_states = []
        if SAVE_DATA:
            for i in range(ACTIVE_AGENTS):
                tt = time.time()
                T_map_agent = self.tf2_wrapper.get_latest_pose("map", HUMAN_FRAME + "_" + str(i+1))
                if T_map_agent is None:
                    print("Can't find human pose")
                    agent_state = torch.tensor([10.0, 10.0, 0.0], dtype=torch.float32)
                else:
                    yaw = 2.0 * math.atan2(
                        T_map_agent.rotation.z, T_map_agent.rotation.w
                    )  # NOTE: assuming roll and pitch are negligible
                    agent_state = torch.tensor([T_map_agent.translation.x, T_map_agent.translation.y, yaw], dtype=torch.float32)
                    agent_state[2] = normalize_angle(agent_state[2])
                agent_states.append(agent_state)
        agent_states_tensor = torch.cat(agent_states).reshape((ACTIVE_AGENTS, 3))
        state[2] = normalize_angle(state[2])
        self.states.append(state)

        if CV_PREDICTIONS:
            self.prev_agent_states.append(agent_states_tensor)
            if self.counter < 1 + int(DT/HZ):
                return
            else:
                dt = int(DT / HZ)
                self.cv_predictions, self.cv_logits = self.model_predictor.construct_cv_predictions(self.prev_agent_states[-1], self.prev_agent_states[-1-dt], static=STATIC)
                self.cv_logits = self.cv_logits.to(self.device)
                self.cv_predictions = torch.tensor(self.boundary_aware_predictions(self.cv_predictions.cpu().numpy())).to(self.device)

                self.cv_robot_prediction, _ = self.model_predictor.construct_cv_predictions(torch.tensor(self.states[-1]).cpu()[None,:], torch.tensor(self.states[-1-dt]).cpu()[None,:], static=STATIC)
                self.cv_robot_prediction = torch.tensor(self.boundary_aware_predictions(self.cv_robot_prediction.cpu().numpy())).to(self.device)

        if torch.norm(state[0:2] - self.goals[self.goal_index]) < 0.5:
            self.index = self.index + 1
            self.goal_index = self.goal_indices[GOAL_SEQUENCE][self.index]
            self.turning_to_center = True

        if self.turning_to_center:
            vx_curr = torch.sin(state[2])
            vy_curr = torch.cos(state[2])
            dx = self.goals[self.goal_index][0] - state[0]
            dy = self.goals[self.goal_index][1] - state[1]
            dist = torch.sqrt(dx**2 + dy**2)
            dx = dx / dist
            dy = dy / dist

            dtheta = torch.atan2(dy, dx)

            delta_angle = dtheta - state[2]
            delta_angle = (delta_angle + torch.pi) % (2 * torch.pi) - torch.pi

            if abs(delta_angle) < torch.pi / 64:
                self.turning_to_center = False
            else:
                if abs(delta_angle) < torch.pi/6:
                    v_theta = 0.2
                else:
                    v_theta = 1.5
                if delta_angle < 0.0:
                    v_theta = -1. * v_theta

                command = TwistStamped()
                command.header.frame_id = "odom_combined"
                command.header.stamp = self.get_clock().now().to_msg()
                x = 0.0
                command.twist.linear.x = min(x, 0.3)
                command.twist.angular.z = v_theta
                self.cmd_vel_pub.publish(command)

        else:
            if COHAN and (time.time() - self.time_since_last_goal_sent > 0.5):
                g = self.goals[self.goal_index].cpu().numpy()
                nav_goal = PoseStamped()
                nav_goal.header.frame_id = "map"
                nav_goal.header.stamp = self.get_clock().now().to_msg()
                nav_goal.pose.position.x = float(g[0])
                nav_goal.pose.position.y = float(g[1])
                nav_goal.pose.position.z = 0.0

                q = tf_transformations.quaternion_from_euler(0.0, 0.0, state[2].item())
                nav_goal.pose.orientation.x = q[0]
                nav_goal.pose.orientation.y = q[1]
                nav_goal.pose.orientation.z = q[2]
                nav_goal.pose.orientation.w = q[3]

                self.goal_pub.publish(nav_goal)
                self.time_since_last_goal_sent = time.time()
            if COHAN_ONLY:
                command = TwistStamped()
                command.header.frame_id = "odom_combined"
                command.header.stamp = self.get_clock().now().to_msg()
                command.twist = self.cohan_velocity
                self.vel = (self.cohan_velocity.linear.x, self.cohan_velocity.angular.z)
                return
            else:
                # Get best action from MPPI
                self.rollouts = torch.zeros_like(self.rollouts)
                action = self.mppi.command(state)
                self.mppi.u_init = action

                rollouts = self.mppi.states
                costs = self.mppi.cost_total

            # Publish action
            command = TwistStamped()
            command.header.frame_id = "odom_combined"
            command.header.stamp = self.get_clock().now().to_msg()
            x = action[0].item()
            command.twist.linear.x = min(x, 0.3)
            command.twist.angular.z = action[1].item()
            self.cmd_vel_pub.publish(command)
            self.vel = (action[0].item(), action[1].item())

            rollouts = rollouts.squeeze(0)
            costs = costs.squeeze(0)
            max_cost = torch.argmax(costs)
            min_cost = torch.argmin(costs)
            max_rollout = rollouts[max_cost]
            min_rollout = rollouts[min_cost]

        data_save_start = time.time()
        if COHAN:
            save_robot_preds = self.robot_predictions
            save_raw_preds = self.raw_predictions
        else:
            save_robot_preds = []
            save_raw_preds = []
        if JSON == False:
            if CV_PREDICTIONS:
                save_preds = self.cv_predictions.cpu().numpy()
                save_logits = self.cv_logits.cpu().numpy()
                save_robot_pred = self.cv_robot_prediction.cpu().numpy()
            else:
                save_preds = self.predictions.cpu().numpy()
                save_logits = self.logits.cpu().numpy()
                save_robot_pred = self.robot_prediction.cpu().numpy()
            data_to_save = {
                'robot_state': state.cpu().numpy(),
                'agent_states': agent_states,
                'predictions': save_preds,
                'robot_prediction': save_robot_pred,
                'logits': save_logits,
                'robot_goal': self.goals[self.goal_index].cpu().numpy(),
                'obstacles': OBSTACLE_LIST,
                'turning': self.turning_to_center,
                'time': time.time(),
                'raw_cohan_robot_prediction': save_robot_preds,
                'raw_cohan_predictions': save_raw_preds
            }
        else:
            if CV_PREDICTIONS:
                save_preds = self.cv_predictions.cpu().tolist()
                save_logits = self.cv_logits.cpu().tolist()
                save_robot_pred = self.cv_robot_prediction.cpu().tolist()
            else:
                save_preds = self.predictions.cpu().tolist()
                save_logits = self.logits.cpu().tolist()
                save_robot_pred = self.robot_prediction.cpu().tolist()
            data_to_save = {
                'robot_state': state.cpu().tolist(),
                'agent_states': agent_states_tensor.cpu().tolist(),
                'predictions': save_preds,
                'robot_prediction': save_robot_pred,
                'logits': save_logits,
                'robot_goal': self.goals[self.goal_index].cpu().tolist(),
                'obstacles': OBSTACLE_LIST,
                'turning': self.turning_to_center,
                'time': time.time(),
                'raw_cohan_robot_prediction': save_robot_preds,
                'raw_cohan_predictions': save_raw_preds
            }
        self.data_processor.write_data(data_to_save)
        current_time = time.time()
        self.time = current_time

    def dynamics(self, s: torch.Tensor, a: torch.Tensor, t=None) -> torch.Tensor:
        """
        Input:
        s: robot global state  (shape: BS x 3)
        a: robot action   (shape: BS x 2)


        Output:
        next robot global state after executing action (shape: BS x 3)
        """
        assert s.ndim == 2 and s.shape[-1] == 3
        assert a.ndim == 2 and a.shape[-1] == 2

        dt = DT

        self.s2_ego.zero_()
        s2_ego = torch.zeros_like(s).to(self.device)
        s2_ego = self.s2_ego
        d_theta = a[:, 1] * dt
        turning_radius = a[:, 0] / a[:, 1]

        s2_ego[:, 0] = torch.where(
            a[:, 1] == 0, a[:, 0] * dt, turning_radius * torch.sin(d_theta)
        )
        s2_ego[:, 1] = torch.where(
            a[:, 1] == 0, 0.0, turning_radius * (1.0 - torch.cos(d_theta))
        )
        s2_ego[:, 2] = torch.where(a[:, 1] == 0, 0.0, d_theta)

        s2_global = torch.zeros_like(s)
        s2_global[:, 0] = (
            s[:, 0] + s2_ego[:, 0] * torch.cos(s[:, 2]) - s2_ego[:, 1] * torch.sin(s[:, 2])
        )
        s2_global[:, 1] = (
            s[:, 1] + s2_ego[:, 0] * torch.sin(s[:, 2]) + s2_ego[:, 1] * torch.cos(s[:, 2])
        )
        s2_global[:, 2] = normalize_angle(s[:, 2] + s2_ego[:, 2])

        return s2_global

if __name__ == "__main__":
    rclpy.init()

    node = MPPI_node()

    rclpy.spin(node)
    rclpy.shutdown()