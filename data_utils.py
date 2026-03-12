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

from controller_config import *
import pickle

import os
import cv2
import json

from scipy.stats import shapiro
from numpy.random import randn
import glob
from collections import defaultdict
import pandas as pd
import pingouin as pg

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from scipy.interpolate import griddata

from rectangle import OBB, obb_intersect, closest_distance_two_particles_with_stops

TLX = {
    'hst': [[0, 0, 0, 21, 2, 0], [5, 3, 5, 18, 5, 12], [1, 1, 2, 18, 2, 2], [13, 3, 3, 20, 10, 4]],
    'cvp': [[0, 0, 2, 21, 2, 0], [3, 3, 3, 18, 4, 8], [1, 1, 1, 17, 6, 3], [12, 3, 6, 20, 8, 6]],
    'static': [[0, 0, 0, 21, 0, 0], [2, 3, 3, 18, 5, 9], [2, 2, 5, 15, 6, 4], [14, 4, 4, 20, 6, 4]],
    'blind': [[0, 0, 2, 21, 2, 4], [3, 3, 5, 18, 6, 6], [1, 1, 3, 19, 6, 2], [16, 3, 2, 20, 6, 8]]
}

ROSAS = {
    'hst': [[4.83, 5.16, 2.67], [5.0, 4.5, 3.83], [1.5, 6.16, 1.83], [1.0, 2.83, 3.5]],
    'cvp': [[5.0, 6.83, 2.0], [4.33, 5.33, 4.16], [1.67, 6.67, 2.16], [1.0, 2.0, 3.0]],
    'static': [[5.0, 6.33, 2.0], [4.83, 6.16, 4.0], [1.5, 6.0, 2.83], [1.0, 3.66, 3.33]],
    'blind': [[4.0, 5.0, 4.67], [4.67, 4.5, 4.33], [3.66, 5.83, 2.33], [1.0, 2.0, 4.83]]
}

class DataProcessor:
    def __init__(self, filepath, use_predictions=True, count_turning=False, done_step=0):
        self.filepath = filepath
        self.reaction_time = 0
        self.avg_reaction = []
        self.robot_states = None
        self.agent_states = None
        self.robot_goals = None
        self.max_rollouts = None
        self.min_rollouts = None
        self.turning = None
        self.goal_step = 0
        self.use_predictions = use_predictions
        self.done_step = done_step
        self.time = None
        self.ADE_AVERAGE = 0.0
        self.ADE_SKIP_AVERAGE = 0.0
        self.ADE_TOTAL_AVERAGE = 0.0

    def write_data(self, data_to_save):
        if JSON:
            with open(self.filepath, "w+") as json_data:
                json.dump(data_to_save, json_data)
                json_data.write('\n')
        else:
            with open(self.filepath, "ab") as pickle_data:
                pickle.dump(data_to_save, pickle_data)

    def read_pkl(self, cohan=False):
        counter = 0
        robot_states = []
        agent_states = []
        robot_goals = []
        max_rollouts = []
        min_rollouts = []
        robot_goals = []
        logits = []
        predictions = []
        robot_predictions = []
        turning = []
        time_list = [0.0]
        last_size = None
        counter = 1
        raw_cohan_predictions = []

        pickle_data = open(self.filepath, "rb")

        while True:
            try:
                dict_to_read = pickle.load(pickle_data)
                counter = counter + 1
                if self.filepath == '/home/socnav/experiment_data/cohan_test.pkl' and counter < 2960:
                    continue
                
                agent_states_arr = np.array(dict_to_read['agent_states'])
                if agent_states_arr.shape[0] != ACTIVE_AGENTS:
                    continue
                prediction_arr = np.array(dict_to_read['predictions'])
                if counter == 0:
                    last_size = prediction_arr.shape
                else:
                    if prediction_arr.shape != last_size:
                        last_size = prediction_arr.shape
                if self.use_predictions and prediction_arr.shape[3] == ACTIVE_AGENTS and np.array(dict_to_read['logits']).shape[0] == 1:
                    predictions.append(dict_to_read['predictions'])
                    if cohan:
                        if len(dict_to_read['robot_prediction']) > 0:
                            robot_predictions.append(dict_to_read['robot_prediction'])
                        else:
                            robot_predictions.append([[0.0, 0.0]])
                    else:
                        robot_predictions.append(dict_to_read['robot_prediction'])
                elif not self.use_predictions and prediction_arr.shape[3] == 1:
                    predictions.append(dict_to_read['predictions'])
                else:
                    continue
                if 'time' in dict_to_read:
                    time_list.append(dict_to_read['time'])
                else:
                    time_list.append(time_list[-1] + 0.02)

                robot_states.append(dict_to_read['robot_state'])
                logits.append(dict_to_read['logits'])
                agent_states.append(dict_to_read['agent_states'])
                robot_goals.append(dict_to_read['robot_goal'])
                #max_rollouts.append(dict_to_read['max_rollout'])
                #min_rollouts.append(dict_to_read['min_rollout'])
                if 'turning' in dict_to_read:
                    turning.append(dict_to_read['turning'])
                else:
                    turning.append(False)
                if cohan:
                    if dict_to_read['raw_cohan_predictions'] is not None:
                        rp = dict_to_read['raw_cohan_predictions']
                        rpl = []
                        for p in range(len(rp)):
                            pl = []
                            for s in range(len(rp[p])):
                                pl.append([rp[p][s].position.x, rp[p][s].position.y])
                            rpl.append(pl)
                        raw_cohan_predictions.append(rpl)
                    else:
                        raw_cohan_predictions.append([[[0.0, 0.0]], [[0.0, 0.0]]])

            except EOFError:
                break

        self.robot_states = np.array(robot_states)
        self.agent_states = np.array(agent_states)
        self.robot_goals = np.array(robot_goals)
        self.max_rollouts = np.array(max_rollouts)
        self.min_rollouts = np.array(min_rollouts)
        self.predictions = np.array(predictions)
        self.logits = np.array(logits)
        self.turning = np.array(turning)
        if cohan:
            self.robot_prediction = robot_predictions
        else:
            self.robot_prediction = np.array(robot_predictions)
        self.time = np.array(time_list)
        self.raw_cohan_predictions = raw_cohan_predictions

    def read_json(self, cohan):
        json_data = open(self.filepath, "rb")
        dict_to_read = json.load(json_data)

        robot_states = []
        agent_states = []
        robot_goals = []
        logits = []
        predictions = []
        robot_predictions = []
        turning = []
        time_list = []
        last_size = None
        counter = 1
        raw_cohan_predictions = []

        for i in range(len(dict_to_read)):
            robot_states.append(dict_to_read[i]['robot_state'])
            agent_states.append(dict_to_read[i]['agent_states'])
            robot_goals.append(dict_to_read[i]['robot_goal'])
            logits.append(dict_to_read[i]['logits'])
            predictions.append(dict_to_read[i]['predictions'])
            robot_predictions.append(dict_to_read[i]['robot_prediction'])
            turning.append(dict_to_read[i]['turning'])
            time_list.append(dict_to_read[i]['time'])
            if cohan:
                raw_cohan_predictions.append(dict_to_read[i]['raw_cohan_predictions'])

        self.robot_states = np.array(robot_states)
        self.agent_states = np.array(agent_states)
        self.robot_goals = np.array(robot_goals)
        self.predictions = np.array(predictions)
        self.logits = np.array(logits)
        self.turning = np.array(turning)
        if cohan:
            self.robot_prediction = robot_predictions
        else:
            self.robot_prediction = np.array(robot_predictions)
        self.time = np.array(time_list)
        self.raw_cohan_predictions = raw_cohan_predictions

    def moving_average_numpy(self, data, window_size):
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        if window_size > len(data):
            raise ValueError("Window size cannot be greater than the data length.")

        weights = np.ones(window_size) / window_size
        return np.convolve(data, weights, mode='same')

    def save_to_json(self, name, cohan=False):
        data = []

        for i in range(self.robot_states.shape[0]):
            step_data = {}
            step_data['robot_state'] = self.robot_states[i].tolist()
            step_data['agent_states'] = self.agent_states[i].tolist()
            step_data['predictions'] = self.predictions[i].tolist()
            step_data['robot_prediction'] = self.robot_prediction[i].tolist()
            step_data['robot_goal'] = self.robot_goals[i].tolist()
            step_data['turning'] = self.turning[i].tolist()
            step_data['time'] = self.time[i].tolist()
            step_data['logits'] = self.logits[i].tolist()
            if 'cohan' in name:
                step_data['raw_cohan_predictions'] = self.raw_cohan_predictions[i]
            data.append(step_data)

        for i in range(len(data)):
            for key in data[i]:
                if isinstance(data[i][key], np.ndarray):
                    print(i, key)

        fp = open(name, 'w+')
        json.dump(data, fp)

    def calculate_metrics(self, agent_labels=0, cohan=False, fr=False):
        ADE = 0.0
        FDE = 0.0
        ADE_skips = 0.0
        min_dist = 1e5
        avg_min_dist = 0.0
        distance = 0.0
        rotation = 0.0
        acceleration = 0.0
        velocity = 0.0
        resp0 = 0.0
        resp1 = 0.0

        goal_step = 0
        time = 0.0
        agent_times = []
        success = False

        min_dist_to_travel = 0.0
        first_goal = True
        robot_goals_visited = 0

        agent_distance = np.zeros(ACTIVE_AGENTS)
        agent_rotation = np.zeros(ACTIVE_AGENTS)
        agent_acceleration = np.zeros(ACTIVE_AGENTS)
        agent_velocity = np.zeros(ACTIVE_AGENTS)
        agent_min_dist_to_travel = np.zeros(ACTIVE_AGENTS)
        agent_agent_intersections = np.zeros(ACTIVE_AGENTS)
        agent_robot_intersections = np.zeros(ACTIVE_AGENTS)
        responsibility_robot = np.zeros(ACTIVE_AGENTS)
        responsibility_other = np.zeros(ACTIVE_AGENTS)
        rsteps_robot = np.zeros(ACTIVE_AGENTS)
        rsteps_other = np.zeros(ACTIVE_AGENTS)
        rsteps_r0 = 0
        rsteps_r1 = 0
        ADE_a = np.zeros(ACTIVE_AGENTS)
        goal_step_agent = np.zeros(ACTIVE_AGENTS)

        done_step = self.robot_states.shape[0]

        agent_goals = []
        agent_goal_indices = []
        agent_goals_visited = []
        dist_thresh = [0.8, 1.0, 1.2, 1.5]
        heading_thresh = [0.25, 0.175, 0.1, 0.05, 0.01]
        norm_thresh = [0.5, 0.65, 0.80, 1.0, 1.2]
        for a in range(self.agent_states.shape[1]):
            bl = False
            for d in dist_thresh:
                for h in heading_thresh:
                    for n in norm_thresh:
                        goals, mdt, new_goals, goal_indices = self.find_agent_goals_overfit(a, done_step, agent_labels, fr=fr, dist=d, head=h, nm=n)
                        if new_goals >= 40:
                            bl = True
                            break
                    if bl:
                        break
                if bl:
                    break

            agent_goals.append(goals)
            agent_goal_indices.append(goal_indices)
            agent_min_dist_to_travel[a] = mdt
            agent_goals_visited.append(new_goals)
        self.agent_goals = np.array(agent_goals)

        last_min_distance = 0.0

        robot_accels = []
        robot_vels = []
        human_accels = [[], []]
        human_vels = [[], []]
        steps_plot = []
        steps_ade = []
        ades = []
        human_pos = [[], []]
        human_pis = [[], []]
        human_ades = [[], []]

        skipped_steps = 0
        non_skipped = 0

        goals_visited = {}
        agents_done = {}
        for a in range(ACTIVE_AGENTS):
            goals_visited[a] = 0
            agents_done[a] = False

        num_resp_timesteps = 0

        for step in range(2, done_step):
            if not np.all(self.robot_goals[step] == self.robot_goals[step-1]):
                robot_goals_visited += 1

            dt = self.time[step] - self.time[step-1]
            dt_prev = self.time[step-1] - self.time[step-2]
            time = time + dt

            for a in range(ACTIVE_AGENTS):
                if not np.all(agent_goals[a][step] == agent_goals[a][step-1]):
                    goals_visited[a] = goals_visited[a] + 1
                if goals_visited[a] >= agent_goals_visited[a]:
                    if agents_done[a] == False:
                        agents_done[a] = True
                        agent_times.append(time)
                    continue

            if self.turning[step]:
                continue

            # human_edge_filter = self.workspace_edge_filter(step, dist=0.5, fr=fr)
            # human_goal_filter = self.goal_crossing_filter(step, [agent_goal_indices[0][step], agent_goal_indices[1][step]], self.robot_goals[step], fr=fr)
            # human_half_filter = self.first_half_filter(step, [agent_goal_indices[0][step], agent_goal_indices[1][step]], fr=fr)
            human_step_filter = [True, True]
            if FILTER_BOTH_AGENTS and ((not human_step_filter[0]) or (not human_step_filter[1])):
                continue

            goal_step += 1

            if RESPONSIBILITY_ONLY_TIMESTEPS:
                skipped_steps += 1
                resp = [False, False]
                for a in range(ACTIVE_AGENTS):
                    distance_a, extra_rotation_a, acceleration_a, velocity_a, intersect_robot, intersect_other, resp_robot, resp_other = self.collect_agent_metrics(a, step, dt, dt_prev, fr=fr)
                    if resp_other != 0:
                        resp[a] = True
                
                if (not resp[0]) and (not resp[1]):
                    continue
                else:
                    num_resp_timesteps += 1

            for a in range(ACTIVE_AGENTS):
                if not human_step_filter[a]:
                    continue
                goal_step_agent[a] = goal_step_agent[a] + 1
                distance_a, extra_rotation_a, acceleration_a, velocity_a, intersect_robot, intersect_other, resp_robot, resp_other = self.collect_agent_metrics(a, step, dt, dt_prev, fr=fr)
                if step - self.goal_step > 5:
                    self.calculate_reaction_time(a, step)
                agent_distance[a] = agent_distance[a] + distance_a
                agent_rotation[a] = agent_rotation[a] + extra_rotation_a
                agent_acceleration[a] = agent_acceleration[a] + acceleration_a
                agent_velocity[a] = agent_velocity[a] + velocity_a

                if intersect_other:
                    agent_agent_intersections[a] += 1
                if intersect_robot:
                    agent_robot_intersections[a] += 1

                responsibility_robot[a] += resp_robot
                responsibility_other[a] += resp_other

                if not resp_robot == 0.0:
                    rsteps_robot[a] += 1
                if not resp_other == 0.0:
                    rsteps_other[a] += 1

                human_accels[a].append(acceleration_a)
                human_vels[a].append(velocity_a)
                human_pis[a].append(extra_rotation_a)

            if self.use_predictions:
                ADE_s, FDE_s, ADE_as, pid, vd, skip = self.collect_prediction_metrics(step, dt, cohan)
                if skip:
                    skipped_steps += 1
                    ADE_skips += ADE_s
                else:
                    non_skipped += 1
                    ADE += ADE_s
                    FDE += FDE_s
                    ades.append(ADE_s)
                    steps_ade.append(step)
                    for a in range(ACTIVE_AGENTS):
                        human_pos[a].append(self.agent_states[step,a,:2])
                        human_ades[a].append(ADE_as[a])
                        if human_step_filter[a]:
                            ADE_a[a] = ADE_a[a] + ADE_as[a]

            min_dist_s, distance_s, extra_rotation_s, acceleration_s, velocity_s, rr0, rr1 = self.collect_navigation_metrics(step, dt, dt_prev)
            if min_dist_s < min_dist:
                min_dist = min_dist_s
            avg_min_dist += min_dist_s
            distance += distance_s
            rotation += extra_rotation_s
            acceleration += acceleration_s
            velocity += velocity_s

            robot_accels.append(acceleration_s)
            robot_vels.append(velocity_s)
            steps_plot.append(step)

            if not rr0 == 0.0:
                rsteps_r0 += 1
            if not rr1 == 0.0:
                rsteps_r1 += 1

            resp0 += rr0
            resp1 += rr1

            if not np.allclose(self.robot_goals[step], self.robot_goals[step-1]):
                if first_goal:
                    min_dist_to_travel += (np.linalg.norm(self.robot_goals[step-1] - self.robot_states[0,:2]) - 1.0)
                min_dist_to_travel += (np.linalg.norm(self.robot_goals[step] - self.robot_goals[step-1]) - 1.0)
                last_min_distance = np.linalg.norm(self.robot_goals[step] - self.robot_goals[step-1]) - 1.0

        ADE = ADE / (non_skipped)
        ADE_skips = ADE_skips / skipped_steps
        FDE = FDE / (non_skipped)
        ADE_a = ADE_a / (goal_step_agent - skipped_steps)

        min_dist_to_travel = min_dist_to_travel - last_min_distance

        distance = distance + 1e-8
        path_efficiency = distance - min_dist_to_travel
        path_irregularity = rotation / distance
        average_acceleration = acceleration / goal_step
        average_velocity = velocity / goal_step
        average_min_dist = avg_min_dist / goal_step

        agent_distance = agent_distance + 1e-8
        api = agent_rotation / agent_distance
        ape = agent_distance
        aaa = agent_acceleration / goal_step_agent
        aav = agent_velocity / goal_step_agent

        if rsteps_robot[0] > 0:
            responsibility_robot[0] = responsibility_robot[0] / rsteps_robot[0]
        else:
            responsibility_robot[0] = 0.0
        if rsteps_robot[1] > 0:
            responsibility_robot[1] = responsibility_robot[1] / rsteps_robot[1]
        else:
            responsibility_robot[1] = 0.0 
        if rsteps_other[0] > 0:
            responsibility_other[0] = responsibility_other[0] / rsteps_other[0]
        else:
            responsibility_other[0] = 0.0
        if rsteps_other[1] > 0:
            responsibility_other[1] = responsibility_other[1] / rsteps_other[1]
        else:
            responsibility_other[1] = 0.0

        if rsteps_r0 > 0:
            resp0 = resp0 / rsteps_r0
        else:
            resp0 = 0.0
        if rsteps_r1 > 0:
            resp1 = resp1 / rsteps_r1
        else:
            resp1 = 0.0
        resp = (resp0 + resp1) / 2

        agent_gps = []
        for a in range(ACTIVE_AGENTS):
            agent_gps.append(40 / agent_times[a])
        robot_gps = robot_goals_visited / time

        rob_stats = np.array([ADE, FDE, min_dist, average_min_dist, path_efficiency, min_dist_to_travel, path_irregularity, average_acceleration, average_velocity, time, robot_goals_visited, robot_gps, resp])
        human_stats = np.array([ape, aaa, aav, api, agent_gps, agent_gps, ADE_a, agent_robot_intersections, agent_agent_intersections, responsibility_robot, responsibility_other])
        per_human_stats = [ades, human_ades, robot_accels, robot_vels, human_accels, human_vels, human_pis, human_pos, steps_plot, steps_ade]

        return rob_stats, human_stats, per_human_stats
    

    
    def workspace_edge_filter(self, step, dist=0.5, fr=False):
        if fr:
            workspace_edges = ROBOT_BOUNDARY_FR
        else:
            workspace_edges = ROBOT_BOUNDARY

        include = [True, True]

        for a in range(ACTIVE_AGENTS):
            pos = self.agent_states[step,a,:2]
            if (pos[0] - dist < workspace_edges[0]) or (pos[0] + dist > workspace_edges[2]) or (pos[1] - dist < workspace_edges[1]) or (pos[1] + dist > workspace_edges[3]):
                include[a] = False

        return include
    
    def goal_crossing_filter(self, step, agent_goal_inds, robot_goal, fr=False):
        if fr:
            goals = AGENT_GOALS_FR
        else:
            goals = AGENT_GOALS

        include = [False, False]

        robot_pos = self.robot_states[step,:2]
        robot_goal_ind = 0
        for g in range(GOALS.shape[0]):
            if torch.all(GOALS[g] == robot_goal):
                robot_goal_ind = g

        for a in range(ACTIVE_AGENTS):
            alt = 0
            if a == 0:
                alt = 1
            if agent_goal_inds[a] == 6 or agent_goal_inds[alt] == 6:
                include[a] = False
            elif robot_goal_ind == agent_goal_inds[a]:
                include[a] = True
            else:
                pos = self.agent_states[step,a,:2]
                intersect_rob = segments_intersect(robot_pos, pos, robot_goal, goals[agent_goal_inds[a]])
                intersect_agent = segments_intersect(self.agent_states[step,alt, :2], pos, goals[agent_goal_inds[alt]], goals[agent_goal_inds[a]])
                if intersect_rob or intersect_agent:
                    include[a] = True

        return include
    
    def first_half_filter(self, step, agent_goal_inds, fr=False):
        if fr:
            goals = AGENT_GOALS_FR
        else:
            goals = AGENT_GOALS

        include = [False, False]
        for a in range(ACTIVE_AGENTS):
            if agent_goal_inds[a] == 6:
                include[a] = False
            else:
                pos = self.agent_states[step,a,:2]
                dist = np.linalg.norm(pos - goals[agent_goal_inds[a]])
                if dist > 1.75:
                    include[a] = True

        return include

    
    def collect_prediction_metrics(self, step, dt, cohan=False):
        if self.robot_states.shape[0] - step < PREDICTION_LENGTH * (DT / HZ):
            return 0.0, 0.0, 0.0, 0.0, 0.0, True

        best_mode_idx = np.argmax(self.logits[step])
        #steps batch modes prediction_length max_agent_num xy

        ADE = 0.0
        FDE = 0.0

        ADE_a = np.zeros(ACTIVE_AGENTS)

        for s in range(PREDICTION_LENGTH):
            pred = self.predictions[step,0,best_mode_idx,s,:,:]
            for a in range(ACTIVE_AGENTS):
                dist = np.linalg.norm(self.agent_states[step+(s * int(DT / HZ)),a,:2] - pred[a])
                ADE += dist
                ADE_a[a] = ADE_a[a] + dist
                if s == PREDICTION_LENGTH - 1:
                    FDE += dist

        ADE = ADE / ACTIVE_AGENTS / PREDICTION_LENGTH
        ADE_a = ADE_a / PREDICTION_LENGTH
        FDE = FDE / ACTIVE_AGENTS

        self.ADE_TOTAL_AVERAGE += ADE

        timesteps = int(DT / HZ)
        future_end_timestep = s + int(timesteps * 12)
        pred = self.predictions[s,0,best_mode_idx,:,:,:]
        gt = self.agent_states[s:future_end_timestep:timesteps,:,:2]

        vp = np.zeros(ACTIVE_AGENTS)
        vgt = np.zeros(ACTIVE_AGENTS)

        dp = np.zeros(ACTIVE_AGENTS)
        dgt = np.zeros(ACTIVE_AGENTS)

        rp = np.zeros(ACTIVE_AGENTS)
        rgt = np.zeros(ACTIVE_AGENTS)

        for a in range(ACTIVE_AGENTS):
            for t in range(1, pred.shape[0]):
                v_pred = (pred[t,a,:2] - pred[t-1,a,:2]) / 0.4
                v_gt = (gt[t,a,:2] - gt[t-1,a,:2]) / 0.4

                vp[a] += np.linalg.norm(v_pred)
                vgt[a] += np.linalg.norm(v_gt)

                if np.linalg.norm((pred[-1,a,:2] - pred[t,a,:2])) == 0.0:
                    vpg = np.zeros(ACTIVE_AGENTS)
                else:
                    vpg = (pred[-1,a,:2] - pred[t,a,:2]) / np.linalg.norm((pred[-1,a,:2] - pred[t,a,:2]))
                if np.linalg.norm((gt[-1,a,:2] - gt[t,a,:2])) == 0.0:
                    gtpg = np.zeros(ACTIVE_AGENTS)
                else:
                    gtpg = (gt[-1,a,:2] - gt[t,a,:2]) / np.linalg.norm((gt[-1,a,:2] - gt[t,a,:2]))

                v_pred_r = np.arctan2(v_pred[1], v_pred[0])
                v_gt_r = np.arctan2(v_gt[1], v_gt[0])

                vpg_r = np.arctan2(vpg[1], vpg[0])
                gtpg_r = np.arctan2(gtpg[1], gtpg[0])

                rp += np.abs(vpg_r - v_pred_r)
                rgt += np.abs(gtpg_r - v_gt_r)

                dp += np.linalg.norm(pred[t,a,:2] - pred[t-1,a,:2])
                dgt += np.linalg.norm(gt[t,a,:2] - gt[t-1,a,:2])

        if dp[0] == 0.0 or dp[1] == 0.0:
            ppi = np.zeros(ACTIVE_AGENTS)
        else:
            ppi = rp / dp
        vp = vp / pred.shape[0]

        if dgt[0] == 0.0 or dgt[1] == 0.0:
            gtpi = np.zeros(ACTIVE_AGENTS)
        gtpi = rgt / dgt
        vgt = vgt / pred.shape[0]

        return ADE, FDE, ADE_a, gtpi - ppi, vgt - vp, False
    
    def calculate_social_zone(self, pos, vel, r=0.3, t=0.77, d=0.5):
        l = r / 2 + d + t * np.linalg.norm(vel)
        theta = np.arctan2(vel[1], vel[0])
        if np.linalg.norm(vel) < 1e6:
            center = pos
        else:
            center = pos + (vel / np.linalg.norm(vel)) * (l / 2)

        social_zone = OBB(cx=center[0], cy=center[1], hx=r / 2, hy=l / 2, theta=theta)

        return social_zone
    
    def calculate_cp(self, agent_pos, agent_vel, other_pos, other_vel, agent_r=0.3, other_r=0.3, heading=None, other_heading=None):
        if USE_MOCAP_HEADING:
            agent_vel_h = np.array([np.sin(heading), np.cos(heading)]) * np.linalg.norm(agent_vel)
            other_vel_h = np.array([np.sin(other_heading), np.cos(other_heading)]) * np.linalg.norm(other_vel)
            d_min, t_min, pa, po = closest_distance_two_particles_with_stops(agent_pos, agent_vel_h, other_pos, other_vel_h, -1.75, 1.75, -1.75, 1.75)
        else:
            d_min, t_min, pa, po = closest_distance_two_particles_with_stops(agent_pos, agent_vel, other_pos, other_vel, -1.75, 1.75, -1.75, 1.75)
        return max(0, 1 - (d_min / (agent_r + other_r)))
    
    def collect_agent_metrics(self, agent, step, dt, dt_prev, fr=False):
        if step < 2:
            return None, None, None

        np.seterr(divide='raise', invalid='raise')

        approx_v = (self.agent_states[step,agent,:2] - self.agent_states[step-1,agent,:2]) / dt
        approx_vprev = (self.agent_states[step-1,agent,:2] - self.agent_states[step-2,agent,:2]) / dt_prev
        try:
            optimal_v = (self.agent_goals[agent,step] - self.agent_states[step-1,agent,:2]) / np.linalg.norm(self.agent_goals[agent,step] - self.agent_states[step-1,agent,:2]) * VMAX
        except:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        if USE_MOCAP_HEADING:
            tan1 = self.agent_states[step,agent,2]
        else:
            tan1 = np.arctan2(approx_v[1], approx_v[0])
        tan2 = np.arctan2(optimal_v[1], optimal_v[0])
        if np.linalg.norm(self.agent_goals[agent,step] - self.agent_states[step-1,agent,:2]) > 0.2:
            extra_rotation = np.abs(tan1 - tan2)
        else:
            extra_rotation = 0.0
        acceleration = np.linalg.norm(approx_v - approx_vprev)
        distance = np.linalg.norm(self.agent_states[step,agent,:2] - self.agent_states[step-1,agent,:2])

        #Social zone violation
        other = 0
        if agent == 0:
            other = 1
        other_v = (self.agent_states[step,other,:2] - self.agent_states[step-1,other,:2]) / dt
        robot_v = (self.robot_states[step,:2] - self.robot_states[step-1,:2]) / dt

        agent_sz = self.calculate_social_zone(self.agent_states[step,agent,:2], approx_v)
        other_sz = self.calculate_social_zone(self.agent_states[step,other,:2], other_v)
        robot_sz = self.calculate_social_zone(self.robot_states[step,:2], robot_v)

        #pDCE calculation
        if fr:
            rr = R_PR2
        else:
            rr = R_STRETCH
        cp_ar = self.calculate_cp(self.agent_states[step,agent,:2], approx_v, self.robot_states[step,:2], robot_v, agent_r=R_HUM, other_r=rr, heading=self.agent_states[step,agent,2], other_heading=self.robot_states[step,2])
        cp_ar_prev = self.calculate_cp(self.agent_states[step,agent,:2], approx_vprev, self.robot_states[step,:2], robot_v, agent_r=R_HUM, other_r=rr, heading=self.agent_states[step,agent,2], other_heading=self.robot_states[step,2])
        cp_ao = self.calculate_cp(self.agent_states[step,agent,:2], approx_v, self.agent_states[step,other,:2], other_v, agent_r=R_HUM, other_r=R_HUM, heading=self.agent_states[step,agent,2], other_heading=self.agent_states[step,other,2])
        cp_ao_prev = self.calculate_cp(self.agent_states[step,agent,:2], approx_vprev, self.agent_states[step,other,:2], other_v, agent_r=R_HUM, other_r=R_HUM, heading=self.agent_states[step,agent,2], other_heading=self.agent_states[step,other,2])

        return distance, extra_rotation, acceleration, np.linalg.norm(approx_v), obb_intersect(agent_sz, robot_sz), obb_intersect(agent_sz, other_sz), cp_ar - cp_ar_prev, cp_ao - cp_ao_prev

    def collect_navigation_metrics(self, step, dt, dt_prev, fr=False):
        if step < 2:
            return None, None, None, None

        dists = np.linalg.norm(self.robot_states[step,:2] - self.agent_states[step,:,:2], axis=1)
        min_dist = np.min(dists)

        approx_v = (self.robot_states[step,:2] - self.robot_states[step-1,:2]) / dt
        approx_vprev = (self.robot_states[step-1,:2] - self.robot_states[step-2,:2]) / dt_prev
        optimal_v = (self.robot_goals[step] - self.robot_states[step-1,:2]) / np.linalg.norm(self.robot_goals[step] - self.robot_states[step-1,:2]) * VMAX

        tan1 = np.arctan2(approx_v[1], approx_v[0])
        tan2 = np.arctan2(optimal_v[1], optimal_v[0])
        extra_rotation = np.abs(tan1 - tan2)
        acceleration = np.linalg.norm(approx_v - approx_vprev)
        distance = np.linalg.norm(self.robot_states[step,:2] - self.robot_states[step-1,:2])

        if fr:
            rr = R_PR2
        else:
            rr = R_STRETCH
        approx_v0 = (self.agent_states[step,0,:2] - self.agent_states[step-1,0,:2]) / dt
        approx_v1 = (self.agent_states[step,1,:2] - self.agent_states[step-1,1,:2]) / dt
        robot_v = (self.robot_states[step,:2] - self.robot_states[step-1,:2]) / dt
        cp_ar0 = self.calculate_cp(self.robot_states[step,:2], robot_v, self.agent_states[step,0,:2], approx_v0, agent_r=rr, other_r=R_HUM, heading=self.robot_states[step,2], other_heading=self.agent_states[step,0,2])
        cp_ar0_prev = self.calculate_cp(self.robot_states[step,:2], approx_vprev, self.agent_states[step,0,:2], approx_v0, agent_r=rr, other_r=R_HUM, heading=self.robot_states[step,2], other_heading=self.agent_states[step,0,2])
        cp_ar1 = self.calculate_cp(self.robot_states[step,:2], robot_v, self.agent_states[step,1,:2], approx_v1, agent_r=rr, other_r=R_HUM, heading=self.robot_states[step,2], other_heading=self.agent_states[step,1,2])
        cp_ar1_prev = self.calculate_cp(self.robot_states[step,:2], approx_vprev, self.agent_states[step,1,:2], approx_v1, agent_r=rr, other_r=R_HUM, heading=self.robot_states[step,2], other_heading=self.agent_states[step,1,2])

        return min_dist, distance, extra_rotation, acceleration, np.linalg.norm(approx_v), cp_ar0 - cp_ar0_prev, cp_ar1 - cp_ar1_prev
    
    def find_agent_goals(self, agent, done_step, threshold=18, dist=0.5):
        min_dist_travel = 0.0
        goal_steps = []
        goals = []
        prev_goal_step = 0
        for step in range(threshold, done_step-8):
            if step == threshold:
                goal_steps.append(step)
                dists = np.linalg.norm(AGENT_GOALS - self.agent_states[step,agent,:2])
                mind = np.argmin(dists)
                for i in range(threshold+1):
                    goals.append(AGENT_GOALS[mind])
                continue
            for g in AGENT_GOALS:
                if np.linalg.norm(self.agent_states[step,agent,:2] - g) < dist and np.linalg.norm(g - goals[goal_steps[-1]]) > 2.0:
                    if abs(self.agent_states[step+8,agent,2] - self.agent_states[step-8,agent,2]) > 1.0:
                        min_dist_travel += np.linalg.norm(self.agent_states[step,agent,:2] - self.agent_states[goal_steps[-1],agent,:2])

                        for i in range(step-prev_goal_step):
                            goals.append(g)
                        prev_goal_step = step
                        goal_steps.append(step)
                        break
            if step == done_step-9:
                np.linalg.norm(self.agent_states[step,agent,:2] - self.agent_states[goal_steps[-1],agent,:2])
                for i in range(step-prev_goal_step+9):
                        goals.append(self.agent_states[step,agent,:2])
        return np.array(goals), min_dist_travel
    
    def find_agent_goals_overfit(self, agent, done_step, agent_labels, dist=0.8, head=0.25, nm=0.5, fr=False):
        if fr:
            goal_positions = AGENT_GOALS_FR
        else:
            goal_positions = AGENT_GOALS
        new_goals = 0
        min_dist_travel = 0.0
        goal_steps = []
        goals = []
        goal_indices = []
        prev_goal_step = 0
        if agent == AGENTS[agent_labels][0]:
            prev_goal = 5
        else:
            prev_goal = 0
        for step in range(done_step):
            if step == 0:
                goal_steps.append(step)
                if agent == AGENTS[agent_labels][1]:
                    goals.append(goal_positions[0])
                    goal_indices.append(0)
                else:
                    goals.append(goal_positions[5])
                    goal_indices.append(5)

            if agent == AGENTS[agent_labels][1]:
                if prev_goal == 0 and new_goals == 20:
                    goal_set = [2]
                elif prev_goal == 0:
                        goal_set = [4]
                elif prev_goal == 2:
                    goal_set = [5]
                elif prev_goal == 4:
                    goal_set = [0]
                elif prev_goal == 5:
                    goal_set = [2]
            else:
                if prev_goal == 5 and new_goals == 20:
                    goal_set = [3]
                elif prev_goal == 0:
                    goal_set = [3]
                elif prev_goal == 1:
                    goal_set = [5]
                elif prev_goal == 3:
                    goal_set = [0]
                elif prev_goal == 5:
                    goal_set = [1]
            for g in goal_set:
                if np.linalg.norm(self.agent_states[step,agent,:2] - goal_positions[g]) < dist and np.linalg.norm(self.agent_states[step, agent,:2] - self.agent_states[step-1, agent, :2]) < nm:
                    if step + 8 >= done_step:
                        continue
                    if abs(self.agent_states[step+8,agent,2] - self.agent_states[step-8,agent,2]) > head and (step - prev_goal_step > 30):
                        min_dist_travel += np.linalg.norm(self.agent_states[step,agent,:2] - self.agent_states[goal_steps[-1],agent,:2])
                        new_goals += 1
                        prev_goal = g

                        for i in range(step-prev_goal_step):
                            goals.append(goal_positions[g])
                            goal_indices.append(g)
                        prev_goal_step = step
                        goal_steps.append(step)
                        break
            if step == done_step-9:
                np.linalg.norm(self.agent_states[step,agent,:2] - self.agent_states[goal_steps[-1],agent,:2])
                for i in range(step-prev_goal_step+9):
                        goals.append(self.agent_states[step,agent,:2])
                        goal_indices.append(6)
        return np.array(goals), min_dist_travel, new_goals, np.array(goal_indices)

    
    def find_agent_goals_true(self, agent, done_step):
        # Extract agent positions and yaw
        min_dist_travel = 0
        goals = []
        for step in range(0, done_step):
            agent_position = self.agent_states[step,agent,:2] 
            agent_yaw = self.agent_states[step,agent,2] 

            direction_to_goals = AGENT_GOALS - agent_position  
            distance_to_goals = np.linalg.norm(direction_to_goals,axis=1)  
            direction_to_goals_normalized = direction_to_goals / (distance_to_goals[:, None] + 1e-6)  
            agent_direction = np.array([np.cos(agent_yaw), np.sin(agent_yaw)])  
            alignment_scores = direction_to_goals_normalized @ agent_direction  

            scores = -distance_to_goals + 10*alignment_scores  # Higher is better
            best_goal_index = np.argmax(scores).item()
            goals.append(AGENT_GOALS[best_goal_index])
            if min_dist_travel < np.linalg.norm(AGENT_GOALS[best_goal_index] - agent_position):
                min_dist_travel = np.linalg.norm(AGENT_GOALS[best_goal_index] - agent_position)
        return np.array(goals),min_dist_travel
    
    def filter_agent_goals(self, agent, agent_goals):
        steps_with_goal = 0
        last_goal = agent_goals[agent,0]
        for i in range(1, agent_goals.shape[1]):
            if np.allclose(agent_goals[agent, i], last_goal):
                steps_with_goal += 1
            else:
                last_goal = agent_goals[agent,i]
    

    def create_video(self, output_file='trial.mp4', use_predictions=False, cv_predictions=False, cohan=False, blind=False, static=False, adem=None, adestd=None, data_path=None, data=None):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("Min Rollout Trajectories (5 Time Horizons)")
        ax.grid(True)

        high_ade_x = []
        high_ade_y = []

        # Function to update the plot for each frame
        def update(step):
            if adem is not None:
                ade, _, _, _ = self.collect_prediction_metrics(step, 0.02)
                if ade < adem + adestd:
                    return
                else:
                    high_ade_x.append(step)
                    high_ade_y.append(ade)

            agent_colors = ['red', 'red']
            ax.clear()
            best_mode_idx = np.argmax(self.logits[step])

            dt = int(DT / HZ)
            s = 100
            norm = Normalize(vmin=0, vmax=1)

            if not blind:
                norm = Normalize(vmin=0, vmax=1)
                colors = np.linspace(1, 0, self.predictions.shape[3])
                cl = np.linspace(1, 0, self.predictions.shape[3]-1)

                ax.scatter(self.robot_states[step,0], self.robot_states[step,1], s=s, color=plt.cm.Blues(norm(1)))
                ax.scatter(self.robot_states[step-(dt*HISTORY_LENGTH):step:dt,0], self.robot_states[step-(dt*HISTORY_LENGTH):step:dt,1], s=s, facecolors='none', edgecolors=plt.cm.Blues(norm(1)))
                ax.scatter(self.robot_states[step+dt:step+(dt*PREDICTION_LENGTH):dt,0], self.robot_states[step+dt:step+(dt*PREDICTION_LENGTH):dt,1], s=s, c=cl, cmap=plt.cm.Purples)

                for i in range(ACTIVE_AGENTS):
                    ax.scatter(self.agent_states[step,i,0], self.agent_states[step,i,1], s=s, color=plt.cm.Reds(norm(1)))
                    ax.scatter(self.agent_states[step-(dt*HISTORY_LENGTH):step:dt,i,0], self.agent_states[step-(dt*HISTORY_LENGTH):step:dt,i,1], s=s, facecolors='none', edgecolors=plt.cm.Reds(norm(1)))
                    ax.scatter(self.agent_states[step+dt:step+(dt*PREDICTION_LENGTH):dt,i,0], self.agent_states[step+dt:step+(dt*PREDICTION_LENGTH):dt,i,1], s=s, c=cl, cmap=plt.cm.Purples)

            steps = min(step, HISTORY_LENGTH * 4)

            if use_predictions and not (static or blind):
                colors = np.linspace(1, 0, self.predictions.shape[3])

                for a in range(self.predictions.shape[4]):
                    cb = plt.cm.Blues(norm(a))  # 'Blues' goes from white (low) to blue (high)
                    cr = plt.cm.Reds(norm(a))
                    if cv_predictions:
                        ax.scatter(self.predictions[step, 0, best_mode_idx,:,a,0], self.predictions[step, 0, best_mode_idx,:,a,1], s=s, c=colors, cmap=plt.cm.Reds)
                        ax.scatter(self.robot_prediction[step, 0, best_mode_idx,:,0,0], self.robot_prediction[step,0,best_mode_idx,:,0,1], s=s, c=colors, cmap=plt.cm.Blues)
                    else:
                        if not HIGHEST_PROB_ONLY:
                            top_logs = np.argpartition(self.logits[step], -5)[-5:]
                            for m in top_logs:
                                if m == best_mode_idx:
                                    continue
                                ax.scatter(self.predictions[step, 0, m, :, a, 0], self.predictions[step, 0, m, :, a, 1], s=s, color=(1.0, 0.65, 0.0, 0.4))
                        if cohan:
                            rpa = np.array(self.robot_prediction[step])
                            ax.scatter(rpa[0,0,:,0,0], rpa[0,0,:,0,1], c=colors, s=s, cmap=plt.cm.Blues)
                        else:
                            ax.scatter(self.robot_prediction[step, 0, best_mode_idx,:,0,0], self.robot_prediction[step,0,best_mode_idx,:,0,1], s=s, c=colors, cmap=plt.cm.Blues)
                        ax.scatter(self.predictions[step, 0, best_mode_idx,:,a,0], self.predictions[step, 0, best_mode_idx,:,a,1], s=s, c=colors, cmap=plt.cm.Reds)

            ax.set_xlim(-2.25, 2.25)
            ax.set_ylim(-2.25, 2.25)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)
            ax.set_title('frame ' + str(step) + " " + str(self.agent_states[step,0,:2]))

        if adem is not None:
            plt.clf()
            plt.close()
            plt.figure(figsize=(24,12))
            plt.scatter(data[-1], data[0], s=2, c='blue')
            plt.savefig(data_path + 'ade' + '.png')
            plt.clf()
            plt.close()

    def calculate_reaction_time(self, agent, step, deviation_threshold=0.3):

        human_position = self.agent_states[step,agent,:2]
        goal = self.agent_goals[agent,step]
        start = self.agent_states[0,agent,:2]

        if np.linalg.norm(human_position - goal) < 0.2:
            start = goal
            self.reaction_time = 0
            self.goal_step = step
        else:
            direction_vector = goal - start
            direction_vector /= np.linalg.norm(direction_vector)
            projection = start + np.dot(human_position - start, direction_vector) * direction_vector
                # Calculate the deviation as the perpendicular distance
            deviation = np.linalg.norm(human_position - projection)
            if deviation > deviation_threshold and self.reaction_time == 0:
                 self.reaction_time = step - self.goal_step
                 self.avg_reaction.append(self.reaction_time)

    def visualize_trajectories(self):
        plt.scatter(self.robot_states[:,0], self.robot_states[:,1], cmap='plasma', s=8)
        for i in range(ACTIVE_AGENTS):
            plt.scatter(self.agent_states[:,:,0], self.agent_states[:,:,1], cmap='viridis', s=5)
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.show()

def heatmap_plot(x, y, value):
    # Example data: (x, y, value)
    #x = np.random.uniform(0, 10, 100)
    #y = np.random.uniform(0, 10, 100)
    #value = np.sin(x) * np.cos(y)  # some function of x and y

    # Create a regular grid to interpolate the data
    grid_x, grid_y = np.meshgrid(
        np.linspace(min(x), max(x), 100),
        np.linspace(min(y), max(y), 100)
    )

    # Interpolate the values onto the grid
    grid_z = griddata((x, y), value, (grid_x, grid_y), method='linear')

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(grid_z, extent=(min(x), max(x), min(y), max(y)),
            origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Value')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Heatmap of ADE over (X, Y)')
    plt.show()

def bar_plot(bins, values, labels, name, ylim):
    #plt.xlim = xlim
    plt.ylim = ylim

    xvals = np.arange(len(bins))
    xvals2 = [x + 0.15 for x in xvals]
    xvals3 = [x + 0.15 for x in xvals2]
    xvals4 = [x + 0.15 for x in xvals3]
    xvals_l = [xvals, xvals2, xvals3, xvals4]

    for l in range(len(values)):
        plt.bar(xvals_l[l], values[l], width=0.15, label=labels[l])

    plt.xticks([r + 0.15 for r in range(len(bins))], bins)

    plt.xlabel(name)

    plt.legend()
    plt.show()

def plot_bars():
    #Plot TLX
    tlx = []
    rosas = []
    methods = ['blind', 'cvp', 'hst', 'static']
    for l in methods:
        tlx.append(TLX[l])
        rosas.append(ROSAS[l])

    tlx = np.array(tlx)
    rosas = np.array(rosas)

    tlx = np.mean(tlx, axis=1)
    rosas = np.mean(rosas, axis=1)
    bar_plot(bins=['Mental', 'Physical', 'Rushed', 'Success', 'Hard', 'Annoyed'], values=tlx, labels=methods, name='TLX', ylim=21)
    bar_plot(bins=['Warmth', 'Confidence', 'Discomfort'], values=rosas, labels=methods, name='RoSAS', ylim=9)

def plot_quantitative(metrics, metrics_std):
    names = ['ADE', 'FDE', 'Time', 'PI', 'AA', 'AMD', 'MDT']
    colors = ['red', 'blue', 'purple', 'green', 'orange']
    methods = ['HST', 'COHAN', 'CVP', 'BLIND', 'STATIC']

    for i in range(metrics.shape[0]):
        plt.scatter(names, metrics[i], c=colors[i], label=methods[i])

    plt.legend()
    plt.show()


def list_metrics(list_name, use_predictions, done_step, count_turning=False, video=False, cohan=False, fr=False, data_path=None):
    robt = np.zeros((len(list_name),13))
    agentst = np.zeros((len(list_name),11,ACTIVE_AGENTS))

    for i, f in enumerate(list_name):
        data_processor = DataProcessor(f, use_predictions, count_turning, done_step[i])
        data_processor.read_json(cohan)
        #ADE, FDE, min_dist, average_min_dist, path_efficiency, min_dist_to_travel, path_irregularity, average_acceleration = data_processor.calculate_metrics()
        rob, agents, data = data_processor.calculate_metrics(agent_labels=i, cohan=cohan, fr=fr)
        robt[i] = rob
        agentst[i] = agents
        adestd = np.std(np.array(data[0]))
        #print(agents.shape)

        if data_path is None:
            data_path = os.path.join(DATA_DIR, DATA_TIME)
        if not os.path.exists(data_path):
            os.mkdir(data_path)

        if 'hst' in list_name[0] and video:
            data_processor.create_video(os.path.join(data_path, 'hst_' + str(i) + '.mp4'), use_predictions=True, cv_predictions=False, blind=False, static=False, adem=rob[0], adestd=adestd, data_path=os.path.join(data_path, 'hst_'), data=data)
        if 'cohan' in list_name[0] and video:
            data_processor.create_video(os.path.join(data_path, 'cohan_' + str(i) + '.mp4'), use_predictions=True, cv_predictions=False, cohan=True, blind=False, static=False, adem=rob[0], adestd=adestd, data_path=os.path.join(data_path, 'cohan_'), data=data)
        elif 'blind' in list_name[0] and video:
            data_processor.create_video(os.path.join(data_path, 'blind_' + str(i) + '.mp4'), use_predictions=True, cv_predictions=False, blind=True, static=False, adem=rob[0], adestd=adestd, data_path=os.path.join(data_path, 'blind_'), data=data)
        elif 'cv' in list_name[0] and video:
            data_processor.create_video(os.path.join(data_path, 'cvp_' + str(i) + '.mp4'), use_predictions=True, cv_predictions=True, blind=False, static=False, adem=rob[0], adestd=adestd, data_path=os.path.join(data_path, 'cv_'), data=data)
        elif 'static' in list_name[0] and video:
            data_processor.create_video(os.path.join(data_path, 'static_' + str(i) + '.mp4'), use_predictions=True, cv_predictions=True, blind=False, static=True, adem=rob[0], adestd=adestd, data_path=os.path.join(data_path, 'static_'), data=data)

    agents_separate = np.squeeze(agentst)
    agentst = np.mean(agentst,axis=2)
    am = np.mean(agentst, axis=0)
    astd = np.std(agentst, axis=0)

    rm = np.mean(robt, axis=0)
    rstd = np.std(robt, axis=0)

    agents_separate[5,:] = (agents_separate[5,:] + rm[-2]) / 2
    metrics_list = [rm[0], rm[1], rm[8], am[2], am[1], rm[3], rm[4]]
    metrics_std = [rstd[0], rstd[1], rstd[8], astd[2], astd[1], rstd[3], rstd[4]]

    if MAKE_CSV:
        return agents_separate, [rm[0], rm[1], rm[-2], rm[3], rm[6], rm[7], rm[8], rm[-1]], data
    else:
        return metrics_list, metrics_std, data

def all_metrics(hst_data_list, cohan_data_list, cvp_data_list, static_data_list, blind_data_list, videos):
    metrics = []
    metrics_std = []
    print("HST")
    if len(hst_data_list) != 0:
        m_hst, mstd_hst = list_metrics(hst_data_list, True, count_turning=False, done_step=DONE_STEPS[0], video=videos[0])
        metrics.append(m_hst)
        metrics_std.append(mstd_hst)
    print("COHAN")
    if len(cohan_data_list) != 0:
        m_cohan, mstd_cohan = list_metrics(cohan_data_list, True, count_turning=False, done_step=DONE_STEPS[1], video=videos[1], cohan=True)
        m_cohan[0] = 0.0
        m_cohan[1] = 0.0
        metrics.append(m_cohan)
        metrics_std.append(mstd_cohan)
    print("CVP")
    if len(cvp_data_list) != 0:
        m_cvp, mstd_cvp = list_metrics(cvp_data_list, True, count_turning=False, done_step=DONE_STEPS[2], video=videos[2])
        metrics.append(m_cvp)
        metrics_std.append(mstd_cvp)
    print("STATIC")
    if len(static_data_list) != 0:
        m_static, mstd_static = list_metrics(static_data_list, True, count_turning=False, done_step=DONE_STEPS[3], video=videos[3])
        metrics.append(m_static)
        metrics_std.append(mstd_static)
    print("BLIND")
    if len(blind_data_list) != 0:
        m_blind, mstd_blind = list_metrics(blind_data_list, True, count_turning=False, done_step=DONE_STEPS[4], video=videos[4])
        metrics.append(m_blind)
        metrics_std.append(mstd_blind)

    metrics = np.array(metrics)
    mn = np.linalg.norm(metrics, axis=0)
    mm = np.mean(metrics, axis=0)
    metrics = metrics - mm
    metrics = metrics / mn

    metrics_std = np.array(metrics_std)
    plot_quantitative(metrics, metrics_std)
    plot_bars()

def save_to_jsons(hst_data_list, cohan_data_list, cvp_data_list, static_data_list, blind_data_list, jsons_path):
    print("HST")
    if len(hst_data_list) != 0:
        for i, f in enumerate(hst_data_list):
            p = os.path.join(jsons_path, 'hst.json')
            if os.path.exists(p):
                os.remove(p)
                print('deleted file ', p)
            data_processor = DataProcessor(f, True, False, DONE_STEPS[0])
            data_processor.read_pkl(cohan=False)
            data_processor.save_to_json(os.path.join(jsons_path, 'hst.json'))
            print('saved file ', p)
    print("COHAN")
    if len(cohan_data_list) != 0:
        for i, f in enumerate(cohan_data_list):
            p = os.path.join(jsons_path, 'cohan.json')
            if os.path.exists(p):
                os.remove(p)
            data_processor = DataProcessor(f, True, False, DONE_STEPS[1])
            data_processor.read_pkl(cohan=True)
            data_processor.save_to_json(os.path.join(jsons_path, 'cohan.json'), cohan=True)
    print("CVP")
    if len(cvp_data_list) != 0:
        for i, f in enumerate(cvp_data_list):
            p = os.path.join(jsons_path, 'cv.json')
            if os.path.exists(p):
                os.remove(p)
            data_processor = DataProcessor(f, True, False, DONE_STEPS[2])
            data_processor.read_pkl(cohan=False)
            data_processor.save_to_json(os.path.join(jsons_path, 'cv.json'))
    print("STATIC")
    if len(static_data_list) != 0:
        for i, f in enumerate(static_data_list):
            p = os.path.join(jsons_path, 'static.json')
            if os.path.exists(p):
                os.remove(p)
            data_processor = DataProcessor(f, True, False, DONE_STEPS[3])
            data_processor.read_pkl(cohan=False)
            data_processor.save_to_json(os.path.join(jsons_path, 'static.json'))
    print("BLIND")
    if len(blind_data_list) != 0:
        for i, f in enumerate(blind_data_list):
            p = os.path.join(jsons_path, 'blind.json')
            if os.path.exists(p):
                os.remove(p)
            data_processor = DataProcessor(f, True, False, DONE_STEPS[4])
            data_processor.read_pkl(cohan=False)
            data_processor.save_to_json(os.path.join(jsons_path, 'blind.json'))

def add_to_heatmap_lists(x, y, v, data, v_ind=1):
    for s in range(len(data[v_ind][0])):
        for a in range(ACTIVE_AGENTS):
            v.append(data[v_ind][a][s])
            x.append(data[-3][a][s][0])
            y.append(data[-3][a][s][1])
    return x, y, v

def create_csv():
    directory = '/home/socnav/frb_study_data'
    directory_fr = '/home/socnav/LAAS_study_data'
    algos = ["cv", "hst", "blind", "cohan", "static"]
    api = defaultdict(list)
    ape = defaultdict(list)
    aaa = defaultdict(list)
    aav = defaultdict(list)
    aaint = defaultdict(list)
    arint = defaultdict(list)
    aar = defaultdict(list)
    arr = defaultdict(list)
    tgps = defaultdict(list)
    aade = defaultdict(list)
    agps = defaultdict(list)
    rgps = defaultdict(list)
    raa = defaultdict(list)
    rav = defaultdict(list)
    rpi = defaultdict(list)
    rmd = defaultdict(list)
    rresp = defaultdict(list)
    group = defaultdict(list)
    site = defaultdict(list)
    helmet = defaultdict(list)
    date_time = defaultdict(list)
    csv_path = "/home/socnav/Downloads/both_sites_participant_order.csv"
    csv_data = pd.read_csv(csv_path)
    csv_data = csv_data.dropna(subset=['ResponseId'])
    csv_long = csv_data.melt(id_vars=["ResponseId"], value_vars=["Set 1", "Set 2", "Set 3", "Set 4", "Set 5"],
                            var_name="set_num", value_name="algorithm")
    
    heatmap_dict = {}
    heatmap_dict['fr'] = {}
    heatmap_dict['us'] = {}
    list_keys = ['x', 'y', 'v']
    for country in heatmap_dict:
        for k in algos:
            heatmap_dict[country][k] = {}
            for l in list_keys:
                heatmap_dict[country][k][l] = []

    group_counter = 1

    def add_to_csv(dir, item, model_upper, model_lower, done_step, video, fr):
        print(model_upper)
        am, rm, data = list_metrics([os.path.join(dir, item)], True, count_turning=False, done_step=done_step, video=video, fr=fr, data_path=dir)
        if fr:
            add_to_heatmap_lists(heatmap_dict['fr'][model_lower]['x'], heatmap_dict['fr'][model_lower]['y'], heatmap_dict['fr'][model_lower]['v'], data)
        else:
            add_to_heatmap_lists(heatmap_dict['us'][model_lower]['x'], heatmap_dict['us'][model_lower]['y'], heatmap_dict['us'][model_lower]['v'], data)
        aaa[model_upper].append(am[1][0])
        aaa[model_upper].append(am[1][1])
        aav[model_upper].append(am[2][0])
        aav[model_upper].append(am[2][1])
        aaint[model_upper].append(am[7][0])
        aaint[model_upper].append(am[7][1])
        arint[model_upper].append(am[8][0])
        arint[model_upper].append(am[8][1])
        aar[model_upper].append(am[9][0])
        aar[model_upper].append(am[9][1])
        arr[model_upper].append(am[10][0])
        arr[model_upper].append(am[10][1])
        api[model_upper].append(am[3][0])
        api[model_upper].append(am[3][1])
        tgps[model_upper].append(am[5][0])
        tgps[model_upper].append(am[5][1])
        agps[model_upper].append(am[4][0])
        agps[model_upper].append(am[4][1])
        rgps[model_upper].append(rm[2])
        rgps[model_upper].append(rm[2])
        rav[model_upper].append(rm[6])
        raa[model_upper].append(rm[5])
        rpi[model_upper].append(rm[4])
        rmd[model_upper].append(rm[3])
        rresp[model_upper].append(rm[7])
        rav[model_upper].append(rm[6])
        raa[model_upper].append(rm[5])
        rpi[model_upper].append(rm[4])
        rmd[model_upper].append(rm[3])
        rresp[model_upper].append(rm[7])
        aade[model_upper].append(am[6][0])
        aade[model_upper].append(am[6][1])
        group[model_upper].append(group_counter)
        site[model_upper].append(0)
        group[model_upper].append(group_counter)
        if fr:
            site[model_upper].append(1)
        else:
            site[model_upper].append(0)
        helmet[model_upper].append(0)
        helmet[model_upper].append(1)
        date_time[model_upper].append(d)
        date_time[model_upper].append(d)

    for d in os.listdir(directory):
        if 'consolidated' in d or 'lock' in d or 'example' in d:
            continue
        dir = os.path.join(directory, d + '/json')
        videos = VIDEOS
        for item in os.listdir(dir):
            if not '.json' in item:
                continue
            #print('item ', item, os.path.join(dir, item))
            if 'hst' in item:
               add_to_csv(dir, item, "HST", "hst", DONE_STEPS[0], videos[0], False)
            if 'cv' in item:
                add_to_csv(dir, item, "CV", "cv", DONE_STEPS[2], videos[2], False)
            elif 'blind' in item:
                add_to_csv(dir, item, "Blind", "blind", DONE_STEPS[4], videos[4], False)
            elif 'static' in item:
                add_to_csv(dir, item, "Static", "static", DONE_STEPS[3], videos[3], False)
            if 'cohan' in item:
                add_to_csv(dir, item, "CoHAN", "cohan", DONE_STEPS[1], videos[1], False)
        group_counter = group_counter + 1
    print("\n\n\n\n\nNOW ONTO FRANCE\n\n\n\n\n")
    for d in os.listdir(directory_fr):
        dir = os.path.join(directory_fr, d + '/json')
        videos = VIDEOS
        for item in os.listdir(dir):
            if not '.json' in item:
                continue
            if 'hst' in item:
               add_to_csv(dir, item, "HST", "hst", DONE_STEPS[0], videos[0], True)
            if 'cv' in item:
                add_to_csv(dir, item, "CV", "cv", DONE_STEPS[2], videos[2], True)
            elif 'blind' in item:
                add_to_csv(dir, item, "Blind", "blind", DONE_STEPS[4], videos[4], True)
            elif 'static' in item:
                add_to_csv(dir, item, "Static", "static", DONE_STEPS[3], videos[3], True)
            if 'cohan' in item:
                add_to_csv(dir, item, "CoHAN", "cohan", DONE_STEPS[1], videos[1], True)
        group_counter = group_counter + 1

    # Convert dictionaries to DataFrames
    def convert_metric_dict_to_df(metric_dict):
        df = pd.DataFrame(metric_dict)
        return df.T.reset_index().rename(columns={'index': 'algorithm'})

    pi_df = convert_metric_dict_to_df(api)
    agps_df = convert_metric_dict_to_df(agps)
    tgps_df = convert_metric_dict_to_df(tgps)
    aaint_df = convert_metric_dict_to_df(aaint)
    arint_df = convert_metric_dict_to_df(arint)
    aar_df = convert_metric_dict_to_df(aar)
    arr_df = convert_metric_dict_to_df(arr)
    rgps_df = convert_metric_dict_to_df(rgps)
    raa_df = convert_metric_dict_to_df(raa)
    rav_df = convert_metric_dict_to_df(rav)
    rpi_df = convert_metric_dict_to_df(rpi)
    rmd_df = convert_metric_dict_to_df(rmd)
    rresp_df = convert_metric_dict_to_df(rresp)
    pe_df = convert_metric_dict_to_df(ape)
    aa_df = convert_metric_dict_to_df(aaa)
    av_df = convert_metric_dict_to_df(aav)
    ade_df = convert_metric_dict_to_df(aade)
    group_df = convert_metric_dict_to_df(group)
    site_df = convert_metric_dict_to_df(site)
    helmet_df = convert_metric_dict_to_df(helmet)
    dt_df = convert_metric_dict_to_df(date_time)

    pi_long = pi_df.melt(id_vars=["algorithm"], var_name="ResponseId", value_name="path_irreg")
    aa_long = aa_df.melt(id_vars=["algorithm"], var_name="ResponseId", value_name="avg_accel")
    av_long = av_df.melt(id_vars=["algorithm"], var_name="ResponseId", value_name="avg_vel")
    agps_long = agps_df.melt(id_vars=["algorithm"], var_name="ResponseId", value_name="agent_gps")
    tgps_long = tgps_df.melt(id_vars=["algorithm"], var_name="ResponseId", value_name="team_gps")
    aaint_long = aaint_df.melt(id_vars=["algorithm"], var_name="ResponseId", value_name="agent_ppd")
    arint_long = arint_df.melt(id_vars=["algorithm"], var_name="ResponseId", value_name="robot_ppd")
    aar_long = aar_df.melt(id_vars=["algorithm"], var_name="ResponseId", value_name="agent_resp")
    arr_long = arr_df.melt(id_vars=["algorithm"], var_name="ResponseId", value_name="robot_resp")
    rgps_long = rgps_df.melt(id_vars=["algorithm"], var_name="ResponseId", value_name="robot_gps")
    raa_long = raa_df.melt(id_vars=["algorithm"], var_name="ResponseId", value_name="robot_accel")
    rav_long = rav_df.melt(id_vars=["algorithm"], var_name="ResponseId", value_name="robot_vel")
    rpi_long = rpi_df.melt(id_vars=["algorithm"], var_name="ResponseId", value_name="robot_pi")
    rmd_long = rmd_df.melt(id_vars=["algorithm"], var_name="ResponseId", value_name="robot_min_dist")
    rresp_long = rresp_df.melt(id_vars=["algorithm"], var_name="ResponseId", value_name="rob_avg_resp")
    ade_long = ade_df.melt(id_vars=["algorithm"], var_name="ResponseId", value_name="ade")
    group_long = group_df.melt(id_vars=["algorithm"], var_name="ResponseId", value_name="group")
    site_long = site_df.melt(id_vars=["algorithm"], var_name="ResponseId", value_name="site")
    helmet_long = helmet_df.melt(id_vars=["algorithm"], var_name="ResponseId", value_name="helmet")
    dt_long = dt_df.melt(id_vars=["algorithm"], var_name="ResponseId", value_name="date")

    merged_data = pi_long.merge(csv_long, on=["ResponseId", "algorithm"], how="left")
    merged_data = merged_data.merge(aa_long, on=["ResponseId", "algorithm"], how="left")
    merged_data = merged_data.merge(av_long, on=["ResponseId", "algorithm"], how="left")
    merged_data = merged_data.merge(aaint_long, on=["ResponseId", "algorithm"], how="left")
    merged_data = merged_data.merge(arint_long, on=["ResponseId", "algorithm"], how="left")
    merged_data = merged_data.merge(aar_long, on=["ResponseId", "algorithm"], how="left")
    merged_data = merged_data.merge(arr_long, on=["ResponseId", "algorithm"], how="left")
    merged_data = merged_data.merge(agps_long, on=["ResponseId", "algorithm"], how="left")
    merged_data = merged_data.merge(tgps_long, on=["ResponseId", "algorithm"], how="left")
    merged_data = merged_data.merge(rgps_long, on=["ResponseId", "algorithm"], how="left")
    merged_data = merged_data.merge(raa_long, on=["ResponseId", "algorithm"], how="left")
    merged_data = merged_data.merge(rav_long, on=["ResponseId", "algorithm"], how="left")
    merged_data = merged_data.merge(rpi_long, on=["ResponseId", "algorithm"], how="left")
    merged_data = merged_data.merge(rmd_long, on=["ResponseId", "algorithm"], how="left")
    merged_data = merged_data.merge(rresp_long, on=["ResponseId", "algorithm"], how="left")
    merged_data = merged_data.merge(ade_long, on=["ResponseId", "algorithm"], how="left")
    merged_data = merged_data.merge(group_long, on=["ResponseId", "algorithm"], how="left")
    merged_data = merged_data.merge(site_long, on=["ResponseId", "algorithm"], how="left")
    merged_data = merged_data.merge(helmet_long, on=["ResponseId", "algorithm"], how="left")
    merged_data = merged_data.merge(dt_long, on=["ResponseId", "algorithm"], how="left")

    merged_data.to_csv("objective_all_rob_edge_filter_individual_test_vel_extra_heading_hum_resp_only_on_god.csv")

    for country in heatmap_dict:
        for k in heatmap_dict[country]:
            print(country, k)
            heatmap_plot(heatmap_dict[country][k]['x'], heatmap_dict[country][k]['y'], heatmap_dict[country][k]['v'])

def on_segment(p, q, r):
    """Check if point q lies on segment pr"""
    return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
            min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

def orientation(p, q, r):
    """Return orientation of ordered triplet (p, q, r)
    0 -> colinear
    1 -> clockwise
    2 -> counterclockwise
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - \
          (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    return 1 if val > 0 else 2

def segments_intersect(p1, q1, p2, q2):
    """Return True if line segments p1q1 and p2q2 intersect"""

    # Find the four orientations
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special cases (colinear points)
    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, q2, q1): return True
    if o3 == 0 and on_segment(p2, p1, q2): return True
    if o4 == 0 and on_segment(p2, q1, q2): return True

    return False

if __name__ == '__main__':
    # data_processor = DataProcessor(DATA_PATH, USE_PREDICTIONS)
    # data_processor.read_data()
    # robot_metrics, agent_metrics, prediction_metrics = data_processor.calculate_metrics()
    # print(robot_metrics, agent_metrics, prediction_metrics)

    #heatmap_plot()

    if MAKE_CSV:
        create_csv()
    else:
        for d in DATA_DIR_LIST:
            hst_data_list = []
            cohan_data_list = []
            cvp_data_list = []
            static_data_list = []
            blind_data_list = []
            if LOAD_JSON:
                dir = os.path.join(DATA_DIR, d + '/json')
            else:
                dir = os.path.join(DATA_DIR, d + '/pkl')
            for item in os.listdir(dir):
                if 'hst' in item:
                    hst_data_list.append(os.path.join(dir, item))
                if 'cv' in item:
                    cvp_data_list.append(os.path.join(dir, item))
                elif 'blind' in item:
                    blind_data_list.append(os.path.join(dir, item))
                elif 'static' in item:
                    static_data_list.append(os.path.join(dir, item))
                if 'cohan' in item:
                    cohan_data_list.append(os.path.join(dir, item))

            if LOAD_JSON:
                videos = VIDEOS

                all_metrics(hst_data_list, cohan_data_list, cvp_data_list, static_data_list, blind_data_list, videos)
            else:
                #for d in DATA_DIR_LIST:
                jsons_path = os.path.join(DATA_DIR, d + '/json')
                if not os.path.exists(jsons_path):
                    print('dne')
                    os.mkdir(jsons_path)
                    save_to_jsons(hst_data_list, cohan_data_list, cvp_data_list, static_data_list, blind_data_list, jsons_path)
                else:
                    print('e ', jsons_path)