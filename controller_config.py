from controller_config import *
import torch
import numpy as np

METHOD = 'hst' #hst, cohan, cv, static, blind
DATA_TIME = '12_03_2026_00_00' #Format day_month_year_hour_minute for start of experiment. Keep the same for all methods in single experiment.
GOAL_SEQUENCE = 1 #Set to 1 every other baseline. Sequence 1 is the same but flipped across the x-axis

JSON = False #Save to JSON in controller SHOULD BE FALSE
LOAD_JSON = not JSON #Load JSON data instead of pkl, set to true after converting and saving pkl as json
MAKE_CSV = True
FILTER_BOTH_AGENTS = False

DATA_DIR_LIST = ['12_03_2026_00_00']
DATA_DIR_LIST_FR = ['12_03_2026_00_00']
VIDEOS = [True, True, True, True, True]

DATA_LIST_EXCLUDE = []
EXCLUDE_BLIND = True
RESPONSIBILITY_ONLY_TIMESTEPS = True
USE_MOCAP_HEADING = False

R_PR2 = 0.47
R_STRETCH = 0.24
R_HUM = 0.46

VPREF = 1.0
VMAX = 0.3
DT = 0.4
HISTORY_LENGTH = 7
WINDOW_LENGTH = 20
PREDICTION_LENGTH = WINDOW_LENGTH - HISTORY_LENGTH - 1
MAX_AGENT_NUM = 14
MODES = 20
PREDICTION_HORIZON = DT * (WINDOW_LENGTH - HISTORY_LENGTH - 1)
LOG_COST = True
ACTIVE_AGENTS = 2

HZ = 0.02

DEVICE = 'cuda:0'
USE_TERMINAL_COST = True

SIGMA_H = 0.3
SIGMA_S = 0.3
SIGMA_R = 0.6

Q_OBS = 10e3 #Use with terminal cost
#Q_OBS = 1e2 #For use with non-terminal cost
Q_GOAL = 1.0
Q_WIND = 5.0
Q_DEV = 1.0

NUM_SAMPLES = 2000
    
DISCRETE_COST_TYPE = "winding"

CONTROLLER_NODE_NAME = "social_controller"
CONTROLLER_COMMAND_TOPIC = "twist_command"
PREDICTION_TOPIC = 'hst/predictions'

NEED_PREDICTIONS = True
NEED_ODOM = False
NEED_LASER = False
USE_COHAN = False
COHAN_ONLY = False
COHAN_MPPI = False

CV_CUTOFF = 31714
HST_CUTOFF = 4390

SAVE_DATA = True
HUMAN_FRAME = "human"
DATA_DIR = '/home/socnav/frb_study_data/' #Keep constant now
DATA_PATH = None
VIDEO_DIR = 'realigned_goals.pkl'
VIDEO_NAME = '2_person_hst_1_fs.mp4'
METRICS = True
PREDICTIONS_VIDEO = True
USE_PREDICTIONS = True
BLIND = None #Set to true and run node.py for blind baseline
STATIC = None #Set to true for static prediction baseline
CV_PREDICTIONS = None #Set to true for CV prediction baseline. NOTE must also be true for STATIC baseline
HIGHEST_PROB_ONLY = True
CLOSEST_TO_GOAL = True

if METHOD == 'hst':
    BLIND = False
    STATIC = False
    COHAN = False
    CV_PREDICTIONS = False
elif METHOD == 'cohan':
    BLIND = False
    STATIC = False
    COHAN = True
    CV_PREDICTIONS = False
elif METHOD == 'cv':
    BLIND = False
    STATIC = False
    COHAN = False
    CV_PREDICTIONS = True
elif METHOD == 'static':
    BLIND = False
    STATIC = True
    COHAN = False
    CV_PREDICTIONS = True
elif METHOD == 'blind':
    BLIND = True
    STATIC = False
    COHAN = False
    CV_PREDICTIONS = False

DONE_STEPS = [[0], [0], [0], [0], [0]] #List of early cutoff steps for data analysis. Each list corresponds to the lists of data (HST, CV, COHAN, SM) in order, default value of 0 uses the last timestep.
AGENTS = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]

RADIUS = 0.2
COOLDOWN = 25 

STATIC_OBS = [
    [(-2.00, -1.76), (-10.00, -1.76), (-10.00, 1.77), (-2.00, 1.77)],   #top
    [(10.00, -1.68), (1.06, -1.68), (1.06, 1.79), (10.00, 1.79)],       #bottom
    [(1.16, -10.00), (-2.11, -10.00), (-2.11, -1.68), (1.16, -1.68)],   #left
    [(1.17, 1.69), (-2.34, 1.67), (-2.34, 10.00), (1.17, 10.00)],       #right
    [(-2.34, 1.77), (-10.00, 1.77), (-10.00, 10.00), (-2.34, 10.00)],     #top right
    [(-2.11, -10.00), (-10.00, -10.00), (-10.00, -1.76), (-2.11, -1.76)], #top left
    [(10.00, -10.00), (1.16, -10.00), (1.16, -1.68), (10.00, -1.68)],      #bottom left
    [(10.00, 1.79), (1.06, 1.79), (1.06, 10.00), (10.00, 10.00)]          #bottom right
]

ROBOT_BOUNDARY = [-1.75, -1.75, 1.75, 1.75]
LENGTHWAYS_COORDS = [(-1.45, -1.95), (1.45, -1.95), (-0.3, 1.95), (0.3, 1.95)]
HEIGHTWAYS_COORDS = [(-1.95, 0.0), (1.95, 0.0)]

ROBOT_BOUNDARY_FR = [3.0, 14.0, 6.5, 17.5]

WORKSTATION_OBS = [ #Centers of each workstation, used for creating static obstacles for the robot. Should actually be the same as agent goals
    [-2.00, -1.40],
    [-0.00, -2.00],
    [2.00, -0.40],
    [2.00, 0.50],
    [-0.00, 2.05],
    [-2.05, 1.45]
]

WORKSTATION_OBS_FR = [
    [6.73, 17.1],[4.84, 17.7],[2.82, 16.0],[2.82, 15.1],[4.82, 13.7],[6.73, 14.4]
]
AGENT_GOALS_FR = np.array(WORKSTATION_OBS_FR)

AGENT_GOALS = np.array(WORKSTATION_OBS)
GOALS_FRONT = torch.tensor([[-1.5, -1.4], [-0.1, -1.45], [1.38, -0.40], [1.33, 0.48], [-0.05, 1.45], [-1.55, 1.45]])

GOALS_FRONT_FR = torch.tensor([[5.8, 16.9],[4.85, 16.9], [3.66, 16.0], [3.66, 15.0], [4.85, 14.45], [5.8, 14.5]], dtype=torch.float32)
GOALS = GOALS_FRONT

GOAL_INDICES = torch.tensor([
    [2, 5, 2, 5, 1, 4, 1, 5, 2, 4, 0, 3, 1, 5, 0, 5, 0, 4, 1, 3, 0, 4, 2, 4, 2, 4, 2, 5, 1, 5, 1, 3, 1, 4, 1, 5, 0, 4, 0, 3],
    [3, 0, 3, 0, 4, 1, 4, 0, 3, 1, 5, 2, 4, 0, 5, 0, 5, 1, 4, 2, 5, 1, 3, 1, 3, 1, 3, 0, 4, 0, 4, 2, 4, 1, 4, 0, 5, 1, 5, 2],
    [0, 3, 1, 4, 2, 5, 0, 3, 1, 4, 2, 5, 1, 0, 5, 0, 5, 1, 4, 2, 5, 1, 3, 1, 3, 1, 3, 0, 4, 0, 4, 2, 4, 1, 4, 0, 5, 1, 5, 2]
])

REPEAT_GOALS = True
OBSTACLE_LIST = []