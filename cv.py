import numpy as np   
from controller_config import *
import torch
from utils import dynamics

from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.vectorized import contains

# import rospy

class CV(object):
    def __init__(self):
        super(CV, self).__init__()
        self.dt = None
        self.prediction_horizon = None
        self.wrap = np.vectorize(self._wrap)

        static_obs = STATIC_OBS

        self.polygons = [Polygon(obs) for obs in static_obs]
        self.multi_polygon = MultiPolygon(self.polygons)
        self.bounds = self.multi_polygon.bounds

        self.workstation_polygons = []
        for c in WORKSTATION_OBS:
            self.workstation_polygons.append(Polygon(self.construct_square_obstacle(c,l=1.25)))
        self.workstation_multi_polygon = MultiPolygon(self.workstation_polygons)
    
    def set_params(self):
        self.dt = DT
        self.vpref = VPREF
        self.prediction_horizon = PREDICTION_HORIZON
        self.rollout_steps = int(np.ceil(self.prediction_horizon / self.dt))
        self.prediction_length = int(np.ceil(self.prediction_horizon / self.dt)) + 1
        self.history_length = HISTORY_LENGTH

        self.q_obs = Q_OBS
        self.q_goal = Q_GOAL
        self.q_wind = Q_WIND

        self.sigma_h = SIGMA_H
        self.sigma_s = SIGMA_S
        self.sigma_r = SIGMA_R

        # Normalization factors for Q weights
        self.q_goal_norm = np.square(2 / float(self.prediction_horizon))

        # # Empirically found
        q_wind_norm = 0.1 * np.deg2rad(350)
        self.q_wind_norm = np.square(q_wind_norm)
        self.q_obs_norm = np.square(0.5)

        # Normalized weights
        self.Q_obs = self.q_obs / self.q_obs_norm
        self.Q_goal = self.q_goal / self.q_goal_norm
        self.Q_discrete = self.q_wind / self.q_wind_norm
        self.Q_dev = 0

        self.log_cost = LOG_COST
        self.discrete_cost_type = DISCRETE_COST_TYPE

        self.device = DEVICE

    def construct_square_obstacle(self, center, l=1.5):
        d = l/2
        return [(center[0]+d, center[1]+d), (center[0]-d, center[1]+d), (center[0]-d, center[1]-d), (center[0]+d, center[1]-d)]
    
    def construct_boundary(self, left, top, right, bottom, out=10.00):
        obs = [
            [(out, -out), (bottom, -out), (bottom, left), (out, left)], #bottom left
            [(bottom, -out), (top, -out), (top, left), (bottom, left)], #left
            [(top, -out), (-out, -out), (-out, left), (top, left)], #top left
            [(top, left), (-out, left), (-out, right), (top, right)], #top
            [(top, right), (-out, right), (-out, out), (top, out)], #top right
            [(bottom, right), (top, right), (top, out), (bottom, out)], #right
            [(out, right), (bottom, right), (bottom, out), (out, out)], #bottom right
            [(out, left), (bottom, left), (bottom, right), (out, right)]  #bottom
        ]
        return obs


    def construct_cv_predictions(self, states, prev_states, static=False):
        dist = (states[:,:2] - prev_states[:,:2])
        if static:
            dist = torch.zeros_like(states[:,:2]) + 1e-3
        steps = torch.linspace(1, PREDICTION_LENGTH, PREDICTION_LENGTH)
        steps = steps.reshape((steps.shape[0], 1))
        dist_tile = torch.tile(dist, (PREDICTION_LENGTH, 1, 1))
        dist_mult = dist_tile * steps.view((-1, 1, 1))
        steps_dist = dist_mult + states[:,:2]
        steps_dist = steps_dist.unsqueeze(0)
        steps_dist = steps_dist.unsqueeze(0)
        logits = torch.tensor([1])
        return steps_dist, logits
    
    def predictor_cost(self, state, actions, predictions):
        return np.array([0.0])

    def tracking_cost(self, state, actions, prediction, t):
        pos_after_action = dynamics(state, actions)[:,:2]
        dist = np.linalg.norm(prediction[t] - pos_after_action, axis=1)
        return torch.Tensor(dist)

    def static_cost(self, state, actions):
        final_cost = np.zeros((state.shape[0]))
        pos_after_action = dynamics(state, actions)[:,:2]
        for obst in OBSTACLE_LIST:
            bottom_left = obst[0]
            top_right = obst[1]
            inside = np.where(np.logical_and(np.logical_and((pos_after_action[:,0] > bottom_left[0]), (pos_after_action[:,0] < top_right[0])), np.logical_and((pos_after_action[:,1] > bottom_left[1]), (pos_after_action[:,1] < top_right[1]))), 1000, 0)
            final_cost = final_cost + inside
        return torch.Tensor(final_cost)

    def goal_cost(self, state, actions, goal):
        """Ratio of extra distance that needs to be travelled towards the goal"""
        init_dist = np.linalg.norm(goal-state[:,:2])
        pos_after_action = dynamics(state, actions)[:,:2]
        dist = np.linalg.norm(goal-pos_after_action, axis=1) # (N x T')
        st_dist = np.clip(np.array([self.vpref * self.dt]), 0, init_dist)
        opt_dist = init_dist - st_dist
        cost = (dist - opt_dist) / (2 * st_dist)
        cost = dist
        return torch.Tensor(self.Q_goal * (cost ** 2)) # (N, 1)

    def goal_cost_terminal(self, state, goal):
        """Ratio of extra distance that needs to be travelled towards the goal"""
        dist = torch.norm(goal[None,:]-state[:,:,:2], dim=2) # (N x T')
        cost = torch.sum(dist, dim=1)
        cost = self.Q_goal * (cost ** 2) # (N, 1)
        #cost = cost / torch.max(cost)
        return cost
    
    def collision_avoidance_cost_terminal(self,state):
       # Extract the (x, y) coordinates from the states
        xy_coords = state[..., :2].cpu().numpy()  # Shape: (N, T', 2)
        # Flatten the coordinates for bulk processing
        flattened_coords = xy_coords.reshape(-1, 2)

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

        collision_flags_stations = contains(self.workstation_multi_polygon, flattened_coords[:, 0], flattened_coords[:, 1])
        #collision_flags_stations[~within_bounds] = False  # Points outside bounds are not collisions

        # Assign costs based on collision flags
        costs = torch.where(
            torch.tensor(collision_flags, dtype=torch.bool),
            torch.tensor(10.0),  # Collision cost
            torch.tensor(0.0)   # No collision cost
        )

        # Reshape to (N, T') and sum over timesteps
        costs = costs.view(state.shape[0], state.shape[1]).sum(dim=1)

        costs_stations = torch.where(
            torch.tensor(collision_flags_stations, dtype=torch.bool),
            torch.tensor(1000.0),  # Collision cost
            torch.tensor(0.0)   # No collision cost
        )

        print("stations cost", costs_stations)

        # Reshape to (N, T') and sum over timesteps
        costs_stations = costs_stations.view(state.shape[0], state.shape[1]).sum(dim=1)

        return (costs+costs_stations).to(state.device)

    def obstacle_cost_terminal(self, state, predictions, logits, static=False):
        state_view = state
        dx = state_view[:, None, :, None, 0] - predictions[:,:,:state_view.shape[1],:,0] #(250, 1, 7, 1, 2) - (1, 20, 7, 14, 2)
        dy = state_view[:, None, :, None, 1] - predictions[:,:,:state_view.shape[1],:,1]

        vx = (predictions[:,:,1:state_view.shape[1]+1,:,0] - predictions[:,:,:state_view.shape[1],:,0]) / self.dt
        vy = (predictions[:,:,1:state_view.shape[1]+1,:,1] - predictions[:,:,:state_view.shape[1],:,1]) / self.dt

        # Heading of "other agent"
        obs_theta = torch.arctan2(vy, vx) # N x S x T' x H
        # Checking for static obstacles
        static_obs = (torch.norm(torch.stack((vx, vy), dim=-1), dim=-1) < 0.01) # N x S x T' x H
        # Alpha calculates whether ego agent is in front or behind "other agent"
        #alpha = self.wrap(torch.arctan2(dy, dx) - obs_theta + torch.pi/2.0) <= 0 # N x S x T' x H
        #alpha_arg = (torch.arctan2(dy, dx) - obs_theta + torch.pi/2.0)
        alpha = (torch.arctan2(dy, dx) - obs_theta + torch.pi/2.0)
        alpha = (torch.remainder(alpha + torch.pi, 2 * torch.pi) - torch.pi) <= 0
        #alpha = alpha_arg
        #alpha = self.wrap(alpha_arg.cpu()) <= 0 # N x S x T' x H
        #alpha = torch.from_numpy(alpha).to(self.device)
                                                                                                                                                                                                            # rospy.loginfo(" obs_theta:{} static_obs:{} alpha:{}".format(obs_theta.shape, static_obs.shape, alpha.shape))

        # Sigma values used to create 2D gaussian around obstacles for cost penalty
        sigma = torch.where(alpha, self.sigma_r, self.sigma_h)
        sigma = static_obs + torch.multiply(~static_obs, sigma) # N x S x T' x H
        sigma_s = 1.0 * static_obs + self.sigma_s * (~static_obs) # N x S x T' x H
                                                                                                                                                                                                            # rospy.loginfo("s:{} ss:{}".format(sigma.shape, sigma_s.shape))

        # Variables used in cost_obs function based on sigma and obs_theta
        a = torch.cos(obs_theta) ** 2 / (2 * sigma ** 2) + torch.sin(obs_theta) ** 2 / (2 * sigma_s ** 2)
        b = torch.sin(2 * obs_theta) / (4 * sigma ** 2) - torch.sin(2 * obs_theta) / (4 * sigma_s ** 2)
        c = torch.sin(obs_theta) ** 2 / (2 * sigma ** 2) + torch.cos(obs_theta) ** 2 / (2 * sigma_s ** 2)

        cost = torch.exp(-((a * dx ** 2) + (2 * b * dx * dy) +  (c * dy ** 2))) # N x S x T' x H
        cost = torch.mean(cost, axis=3)
        logits = torch.exp(logits) / sum(torch.exp(logits))
        cost = cost * logits[None, :, None]
        cost = torch.mean(cost, axis=2)
        cost = torch.sum(cost, axis=-1)
        #cost = cost / torch.max(cost)
        cost = -1 * cost
                                                                                                                                                                                                            # rospy.loginfo("c: {}\n\n".format(cost.shape))
        return self.Q_obs * (cost ** 2) # (N, S)

    def obstacle_cost(self, state, actions, predictions, logits, t):
        """
        Cost using 2D Gaussian around obstacles
        """
        # Distance to other agents
  #      rospy.loginfo("act: {} pred: {} state: {}\n\n".format(actions.shape, predictions.shape, state.shape))
        pos_after_action = dynamics(state, actions)[:,:2]
        dx = pos_after_action[:, None, None, None, 0] - predictions[:,:,t,:,0]
        dy = pos_after_action[:, None, None, None, 1] - predictions[:,:,t,:,1]

        vx = (predictions[:,:,t,:,0] - predictions[:,:,t-1,:,0]) / self.dt
        vy = (predictions[:,:,t,:,1] - predictions[:,:,t-1,:,1]) / self.dt
                                                                                                                                                                                                            # rospy.loginfo(" dx:{} dy:{}".format(dx.shape, dy.shape))
        # Heading of "other agent"
        obs_theta = torch.arctan2(vy, vx) # N x S x T' x H
        # Checking for static obstacles
        static_obs = (torch.norm(torch.stack((vx, vy), dim=-1), dim=-1) < 0.01) # N x S x T' x H
        # Alpha calculates whether ego agent is in front or behind "other agent"
        alpha_arg = (torch.arctan2(dy, dx) - obs_theta + torch.pi/2.0)
        alpha = self.wrap(alpha_arg) <= 0 # N x S x T' x H
        alpha = torch.from_numpy(alpha)
                                                                                                                                                                                                            # rospy.loginfo(" obs_theta:{} static_obs:{} alpha:{}".format(obs_theta.shape, static_obs.shape, alpha.shape))

        # Sigma values used to create 2D gaussian around obstacles for cost penalty
        sigma = torch.where(alpha, self.sigma_r, self.sigma_h)
        sigma = static_obs + torch.multiply(~static_obs, sigma) # N x S x T' x H
        sigma_s = 1.0 * static_obs + self.sigma_s * (~static_obs) # N x S x T' x H
                                                                                                                                                                                                            # rospy.loginfo("s:{} ss:{}".format(sigma.shape, sigma_s.shape))
        # Variables used in cost_obs function based on sigma and obs_theta
        a = torch.cos(obs_theta) ** 2 / (2 * sigma ** 2) + torch.sin(obs_theta) ** 2 / (2 * sigma_s ** 2)
        b = torch.sin(2 * obs_theta) / (4 * sigma ** 2) - torch.sin(2 * obs_theta) / (4 * sigma_s ** 2)
        c = torch.sin(obs_theta) ** 2 / (2 * sigma ** 2) + torch.cos(obs_theta) ** 2 / (2 * sigma_s ** 2)

        cost = torch.exp(-((a * dx ** 2) + (2 * b * dx * dy) +  (c * dy ** 2))) # N x S x T' x H
        cost = torch.mean(cost, axis=3)
        logits = np.exp(logits) / sum(np.exp(logits))
        cost = cost * logits[None, None, :]
        cost = torch.sum(cost, axis=-1)
        cost = -1 * cost
                                                                                                                                                                                                            # rospy.loginfo("c: {}\n\n".format(cost.shape))
        return self.Q_obs * (cost ** 2) # (N, S)
    
    def discrete_cost(self, state, actions, predictions): # (1+H) x 5, N x T' x 4, N x S x T' x H x 4
        N = actions.shape[0]
        S = predictions.shape[1]
        state_ = np.tile(state[None, None, None, None, 0, :2]-state[None, None, None, 1:, :2], (N, S, 1, 1, 1))
        dxdy = np.concatenate((state_, actions[:, None, :, None, :2] - predictions[:, :, :, :, :2]), axis=2)
        winding_nums = np.arctan2(dxdy[:, :, :, :, 1], dxdy[:, :, :, :, 0]) # N x S x T' x H
        winding_nums = winding_nums[:, :, 1:]-winding_nums[:, :, :-1]

        if self.discrete_cost_type == 'entropy':
            winding_nums = np.mean(winding_nums, axis=2) < 0 # N x S x H
            p = np.mean(winding_nums, axis=1) # N x H
            # Using mean entropy
            entropy = - (p * np.log(p+1e-8) + (1-p) * np.log(1-p+1e-8))
            entropy = np.mean(entropy, axis=1)[:, None]

            return self.Q_discrete * (entropy ** 2)
        else:
            winding_nums = np.abs(np.mean(winding_nums, axis=2)) # N x S x H

            # considering all agents we are in front of
            dxdy = state[None, 0, :2] - state[1:, :2]
            obs_theta = np.arctan2(state[1:, 3], state[1:, 2])
            alpha = self.wrap(np.arctan2(dxdy[:, 1], dxdy[:, 0]) - obs_theta + np.pi/2.0) >= 0 # N x S x H
            winding_nums = np.multiply(winding_nums, alpha)
            
            winding_nums = np.multiply(winding_nums, alpha)
            winding_nums = np.mean(winding_nums, axis=-1) # N x S

            return - self.Q_discrete * (winding_nums ** 2)
        

    @staticmethod
    def _wrap(angle):  # keep angle between [-pi, pi]
        while angle >= torch.pi:
            angle -= 2 * torch.pi
        while angle < -torch.pi:
            angle += 2 * torch.pi
        return angle