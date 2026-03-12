import math

import torch
import numpy as np

from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from rclpy.duration import Duration
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import time

from stretch_mppi.controller_config import *
import pickle

import os
import cv2

class VisualizationUtils:
    def __init__(self, node: Node) -> None:
        self._node = node

        self._rollouts_pub = self._node.create_publisher(
            MarkerArray, f"/{self._node.get_name()}/vis/rollouts", 1
        )

        self._predictions_pub = self._node.create_publisher(
            MarkerArray, f"/{self._node.get_name()}/vis/predictions", 1
        )

        self._path_pub = self._node.create_publisher(
            Path, f"/{self._node.get_name()}/vis/path", 1
        )

    def visualize_predictions(self, predictions, probs):
        marker_array = MarkerArray()
        max_prob = np.argmax(probs)
        print(time.time())
        for p in range(3):
            for i in range(6):
                marker = Marker()
                marker.header.frame_id = "map"
                marker.id = (i + 1) * (p + 1)
                marker.type = Marker.LINE_STRIP
                marker.scale.x = 0.01
                marker.lifetime = Duration(seconds=1.0).to_msg()
                marker.color.r = marker.color.g = marker.color.b = marker.color.a = 1.0

                for t in range(12):
                    pos = predictions[0,i,t,p,:]
                    marker.points.append(Point(x=pos[0].item(), y=pos[1].item()))

                marker_array.markers.append(marker)
        self._predictions_pub.publish(marker_array)
            


    def visualize_rollouts(self, rollouts: torch.Tensor, costs: torch.Tensor) -> None:
        """
        Input:
        rollouts: (shape: 1 x NUM_SAMPLES x HORIZON x 3)
        costs: (shape: NUM_SAMPLES)
        """
        assert rollouts.ndim == 4 and rollouts.shape[0] == 1 and rollouts.shape[-1] == 3

        min_cost = torch.min(costs).item()
        max_cost = torch.max(costs).item()

        marker_array = MarkerArray()
        for sample_idx in range(rollouts.shape[1]):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.id = sample_idx
            marker.type = Marker.LINE_STRIP
            marker.scale.x = 0.01
            marker.lifetime = Duration(seconds=1.0).to_msg()  # type: ignore

            cost = costs[sample_idx].item()
            if cost == min_cost:
                marker.color.r = marker.color.g = marker.color.b = marker.color.a = 1.0
            else:
                cost_prop = (cost - min_cost) / (max_cost - min_cost)
                # Smooth transition from red -> yellow -> green
                # prop: 1.0 -> 0.5 -> 0.0
                # r   : 1.0 -> 1.0 -> 0.0
                # g   : 0.0 -> 1.0 -> 1.0
                marker.color.r = min(1.0, 2 * cost_prop)
                marker.color.g = max(0.0, 1 - 2 * cost_prop)
                marker.color.a = 0.5

            for state_idx in range(rollouts.shape[2]):
                state = rollouts[0, sample_idx, state_idx]
                marker.points.append(Point(x=state[0].item(), y=state[1].item()))

            marker_array.markers.append(marker)
        self._rollouts_pub.publish(marker_array)

    def visualize_path(self, path: list[tuple[float, float, float]]) -> None:
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self._node.get_clock().now().to_msg()

        for state in path:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = self._node.get_clock().now().to_msg()

            pose.pose.position.x = state[0]
            pose.pose.position.y = state[1]
            pose.pose.orientation.w = math.cos(state[2] / 2)
            pose.pose.orientation.z = math.sin(state[2] / 2)

            path_msg.poses.append(pose)

        self._path_pub.publish(path_msg)

class DataProcessor:
    def __init__(self, filepath, use_predictions=True):
        self.filepath = filepath
        self.use_predictions = use_predictions

        self.robot_states = None
        self.agent_states = None
        self.predictions = None
        self.robot_goals = None
        self.obstacles = None
        self.logits = None
        self.max_rollouts = None
        self.min_rollouts = None

    def write_data(self, data_to_save):
        with open(self.filepath, "ab") as pickle_data:
            pickle.dump(data_to_save, pickle_data)

    def read_data(self):
        counter = 0
        robot_states = []
        agent_states = []
        predictions = []
        robot_goals = []
        obstacles = []
        logits = []
        max_rollouts = []
        min_rollouts = []
        robot_goals = []
        with open(self.filepath, "rb") as pickle_data:
            while True:
                try:
                    dict_to_read = pickle.load(pickle_data)
                    prediction_arr = np.array(dict_to_read['predictions'])
                    if self.use_predictions and prediction_arr.shape[3] == ACTIVE_AGENTS:
                        predictions.append(dict_to_read['predictions'])
                    elif not self.use_predictions and prediction_arr.shape[3] == 1:
                        predictions.append(dict_to_read['predictions'])
                    else:
                        continue

                    counter += 1

                    robot_states.append(dict_to_read['robot_state'])
                    agent_states.append(dict_to_read['agent_states'])
                    logits.append(dict_to_read['logits'])
                    robot_goals.append(dict_to_read['robot_goal'])
                    max_rollouts.append(dict_to_read['max_rollout'])
                    min_rollouts.append(dict_to_read['min_rollout'])

                    if counter == 1:
                        self.obstacles = np.array(dict_to_read['obstacles'])

                except EOFError:
                    break

        self.robot_states = np.array(robot_states)
        self.agent_states = np.array(agent_states)
        self.predictions = np.array(predictions)
        self.logits = np.array(logits)
        self.robot_goals = np.array(robot_goals)
        self.max_rollouts = np.array(max_rollouts)
        self.min_rollouts = np.array(min_rollouts)

        print("Read Data", self.robot_states.shape, self.agent_states.shape, self.predictions.shape, self.robot_goals.shape, self.obstacles, self.max_rollouts.shape, self.min_rollouts.shape)

    def calculate_metrics(self):
        ADE = 0.0
        FDE = 0.0

        min_dist = 1e5
        avg_min_dist = 0.0
        distance = 0.0
        rotation = 0.0
        acceleration = 0.0

        goal_step = 0
        time = 0.0
        success = False

        min_dist_to_travel = 0.0
        first_goal = True

        agent_distance = np.zeros(ACTIVE_AGENTS)
        agent_rotation = np.zeros(ACTIVE_AGENTS)
        agent_acceleration = np.zeros(ACTIVE_AGENTS)
        agent_min_dist_to_travel = np.zeros(ACTIVE_AGENTS)

        done_step = DONE_STEP
        if done_step == 0:
            done_step = self.robot_states.shape[0]

        agent_goals = []
        for a in range(self.agent_states.shape[1]):
            goal_steps, goals, mdt = self.find_agent_goals(a, done_step)
            agent_goals.append(goals)
            agent_min_dist_to_travel[a] = mdt
        self.agent_goals = np.array(agent_goals)

        last_min_distance = 0.0

        for step in range(2, done_step):

            if self.use_predictions:
                ADE_s, FDE_s = self.collect_prediction_metrics(step)
                ADE += ADE_s
                FDE += FDE_s

            min_dist_s, distance_s, extra_rotation_s, acceleration_s = self.collect_navigation_metrics(step)
            if min_dist_s < min_dist:
                min_dist = min_dist_s
            avg_min_dist += min_dist_s
            distance += distance_s
            rotation += extra_rotation_s
            acceleration += acceleration_s

            for a in range(ACTIVE_AGENTS):
                distance_a, extra_rotation_a, acceleration_a = self.collect_agent_metrics(a, step)
                agent_distance[a] = agent_distance[a] + distance_a
                agent_rotation[a] = agent_rotation[a] + extra_rotation_a
                agent_acceleration[a] = agent_acceleration[a] + acceleration_a

            if not np.allclose(self.robot_goals[step], self.robot_goals[step-1]):
                if first_goal:
                    min_dist_to_travel += (np.linalg.norm(self.robot_goals[step-1] - self.robot_states[0,:2]) - 1.0)
                min_dist_to_travel += (np.linalg.norm(self.robot_goals[step] - self.robot_goals[step-1]) - 1.0)
                last_min_distance = np.linalg.norm(self.robot_goals[step] - self.robot_goals[step-1]) - 1.0
        
        goal_step = done_step

        ADE = ADE / goal_step
        FDE = FDE / goal_step

        min_dist_to_travel = min_dist_to_travel - last_min_distance

        path_efficiency = distance - min_dist_to_travel
        path_irregularity = rotation / distance
        average_acceleration = acceleration / goal_step
        average_min_dist = avg_min_dist / goal_step
        time = HZ * goal_step

        api = agent_rotation / agent_distance
        ape = agent_distance - agent_min_dist_to_travel
        aaa = agent_acceleration / goal_step

        return np.array([ADE, FDE, min_dist, average_min_dist, path_efficiency, min_dist_to_travel, path_irregularity, average_acceleration]), np.array([ape, aaa, api])

    def collect_prediction_metrics(self, step):
        if self.robot_states.shape[0] - step < PREDICTION_LENGTH:
            return 0.0, 0.0

        best_mode_idx = np.argmax(self.logits[step])
        #steps batch modes prediction_length max_agent_num xy

        ADE = 0.0
        FDE = 0.0

        for s in range(PREDICTION_LENGTH):
            pred = self.predictions[step,0,best_mode_idx,s,:,:]
            for a in range(ACTIVE_AGENTS):
                dist = np.linalg.norm(self.agent_states[step+(s * 8),a,:2] - pred[a])
                ADE += dist
                if s == PREDICTION_LENGTH - 1:
                    FDE += dist

        ADE = ADE / ACTIVE_AGENTS / PREDICTION_LENGTH
        FDE = FDE / ACTIVE_AGENTS

        return ADE, FDE

    def collect_agent_metrics(self, agent, step):
        if step < 2:
            return None, None, None

        np.seterr(divide='raise', invalid='raise')

        approx_v = (self.agent_states[step,agent,:2] - self.agent_states[step-1,agent,:2]) / HZ
        approx_vprev = (self.agent_states[step-1,agent,:2] - self.agent_states[step-2,agent,:2]) / HZ
        try:
            optimal_v = (self.agent_goals[agent,step] - self.agent_states[step-1,agent,:2]) / np.linalg.norm(self.agent_goals[agent,step] - self.agent_states[step-1,agent,:2]) * VMAX
        except:
            return 0.0, 0.0, 0.0

        tan1 = np.arctan2(approx_v[1], approx_v[0])
        tan2 = np.arctan2(optimal_v[1], optimal_v[0])
        extra_rotation = np.abs(tan1 - tan2)
        acceleration = np.linalg.norm(approx_v - approx_vprev)
        distance = np.linalg.norm(self.agent_states[step,agent,:2] - self.agent_states[step-1,agent,:2])

        return distance, extra_rotation, acceleration

    def collect_navigation_metrics(self, step):
        if step < 2:
            return None, None, None, None

        dists = np.linalg.norm(self.robot_states[step,:2] - self.agent_states[step,:,:2], axis=1)
        min_dist = np.min(dists)

        approx_v = (self.robot_states[step,:2] - self.robot_states[step-1,:2]) / HZ
        approx_vprev = (self.robot_states[step-1,:2] - self.robot_states[step-2,:2]) / HZ
        optimal_v = (self.robot_goals[step] - self.robot_states[step-1,:2]) / np.linalg.norm(self.robot_goals[step] - self.robot_states[step-1,:2]) * VMAX

        tan1 = np.arctan2(approx_v[1], approx_v[0])
        tan2 = np.arctan2(optimal_v[1], optimal_v[0])
        extra_rotation = np.abs(tan1 - tan2)
        acceleration = np.linalg.norm(approx_v - approx_vprev)
        distance = np.linalg.norm(self.robot_states[step,:2] - self.robot_states[step-1,:2])

        return min_dist, distance, extra_rotation, acceleration

    def find_agent_goals(self, agent, done_step, threshold=18, dist=0.5):
        min_dist_travel = 0.0
        goal_steps = []
        goals = []
        prev_goal_step = 0
        for step in range(threshold, done_step):
            if step == threshold:
                goal_steps.append(step)
            for g in AGENT_GOALS:
                if np.linalg.norm(self.agent_states[step,agent,:2] - g) < dist:
                    if step - goal_steps[-1] > 1 and np.linalg.norm(self.agent_states[step,agent,:2] - self.agent_states[goal_steps[-1],agent,:2]) > 1.25:
                        min_dist_travel += np.linalg.norm(self.agent_states[step,agent,:2] - self.agent_states[goal_steps[-1],agent,:2])
                        for i in range(step-prev_goal_step):
                            goals.append(self.agent_states[step,agent,:2])
                        prev_goal_step = step
                    goal_steps.append(step)
            if step == done_step-1:
                np.linalg.norm(self.agent_states[step,agent,:2] - self.agent_states[goal_steps[-1],agent,:2])
                for i in range(step-prev_goal_step+1):
                        goals.append(self.agent_states[step,agent,:2])
        return np.array(goal_steps), np.array(goals), min_dist_travel

    def video(self, video_name, video_dir, robot_states=True, agent_states=False, histories=False, goals=False, predictions=False, rollouts=False):
        ra = self.robot_states.shape[0]-(8*PREDICTION_LENGTH)
        if True:
            for step in range(ra):
                plt.figure(figsize=(5, 5))
                print("logits shape ", self.logits.shape)
                best_mode_idx = np.argmax(self.logits[step])

                if robot_states:
                    plt.scatter(self.robot_states[step,0], self.robot_states[step,1], color='blue')

                if agent_states:
                    plt.scatter(self.agent_states[step,:,0], self.agent_states[step,:,1], color='red')
                    plt.scatter(self.agent_states[step:step+(8*PREDICTION_LENGTH):8,:,0], self.agent_states[step:step+(8*PREDICTION_LENGTH):8,:,1], s=8, c='r')

                if histories:
                    steps = min(step, HISTORY_LENGTH * 4)
                    plt.scatter(self.robot_states[step-steps:step, 0], self.robot_states[step-steps:step, 1], s=8, color='black')
                    for a in range(self.predictions.shape[4]):
                        plt.scatter(self.agent_states[step-steps:step,a,0], self.agent_states[step-steps:step,a,1], s=8, color='black')

                if goals:
                    plt.scatter(self.robot_goals[step,0], self.robot_goals[step,1], color='blue', marker='*')

                if predictions:
                    for a in range(self.predictions.shape[4]):
                        if CV_PREDICTIONS:
                            plt.scatter(self.predictions[step, 0, best_mode_idx,:,a,0], self.predictions[step, 0, best_mode_idx,:,a,1], s=8, color='purple')
                        else:
                            top_logs = np.argpartition(self.logits[step], -5)[-5:]
                            for m in top_logs:
                                if m == best_mode_idx:
                                    continue
                                plt.scatter(self.predictions[step, 0, m, :, a, 0], self.predictions[step, 0, m, :, a, 1], s=8, color=(1.0, 0.65, 0.0, 0.4))
                            plt.scatter(self.predictions[step, 0, best_mode_idx,:,a,0], self.predictions[step, 0, best_mode_idx,:,a,1], s=8, color='purple')

                if rollouts:
                    plt.scatter(self.min_rollouts[step,:,0], self.min_rollouts[step,:,1], s=8, color='green')

                plt.xlim(-3.5, 3.5)
                plt.ylim(-3.5, 3.5)
                plt.xlabel('x(m)', fontsize=16)
                plt.ylabel('y(m)', fontsize=16)
                plt.title('frame ' + str(step) + " " + str(self.agent_states[step,0,:2]))

                fpi = self.create_path(video_dir, 'frame_' + str(step))
                plt.savefig(fpi)
                plt.close()
                print("saved step ", step)

        current_directory_path = os.getcwd()
        fp = os.path.join(current_directory_path, video_dir)
        images = [img for img in os.listdir(fp)
                if img.endswith(".jpg") or
                img.endswith(".jpeg") or
                img.endswith("png")]#I'll use my own function for that, just easier to read

        frame = cv2.imread(os.path.join(fp, images[0]))   
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 20, (500, 500))#0.25 so one image is 4 seconds

        for step in range(ra):
            im = cv2.imread(self.create_path(video_dir, 'frame_' + str(step) + '.png'))
            print(im.shape)
            video.write(im)

        cv2.destroyAllWindows()
        video.release()

    def create_path(self, dir_name, file_name):
        current_directory_path = os.getcwd()
        file_path = os.path.join(current_directory_path, file_name)
        subfolder_path = os.path.join(current_directory_path, dir_name)

        if not os.path.exists(subfolder_path):
            os.umask(0)
            os.makedirs(subfolder_path, mode=0o777)

        file_path = os.path.join(subfolder_path, file_name)
        return file_path

    def visualize_trajectories(self):
        plt.scatter(self.robot_states[:,0], self.robot_states[:,1], cmap='plasma', s=8)
        for i in range(ACTIVE_AGENT_NUM):
            plt.scatter(self.agent_states[:,i,0], self.agent_states[:,i,1], cmap='viridis', s=5)
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.show()

def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def get_color_gradient(c1, c2, n):
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors] 
