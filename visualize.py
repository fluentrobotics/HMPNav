from vis_utils import *
import numpy as np
import matplotlib.pyplot as plt
from controller_config import *

if __name__ == '__main__':    
    data_processor = DataProcessor(DATA_PATH, USE_PREDICTIONS)
    data_processor.read_data()
    if PREDICTIONS_VIDEO:
        data_processor.video(VIDEO_NAME, VIDEO_DIR, robot_states=True, agent_states=True, histories=True, goals=True, predictions=True, rollouts=True)
    

    