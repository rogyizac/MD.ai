import numpy as np

class Config(object):
    MODEL_UPDATE_SCHEDULER_INTERVAL = 10 
    
    MODEL_INFERENCE_TOP_N = 3 # Shows the top N inferences based on probability
    MODEL_UPDATE_MIN_THRESHOLD = 10 # Threshold for min number of items for model update
    MODEL_UPDATE_DIR = "data/model_update/" # Directory to scan for images and JSON for model update

    TEMP_DIR = "temp/" # Temp Directory

    CHEXNET_THRESHOLDS = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    # CHEXNET_THRESHOLDS = np.array([0.19362465, 0.07700258, 0.3401143 , 0.39875817, 0.08521137, 0.14014415, 0.02213187, 0.08226113])