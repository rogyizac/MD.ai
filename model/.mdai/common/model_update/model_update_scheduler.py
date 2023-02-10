from common.util.util import scanDirectory
from common.config.config import Config as cfg
import os

def model_update_scheduled_task():
    # Scan directory for json files and update when count reaches threshold
    # threshold = 10

    print("Scanning directory for JSON Files: " + cfg.MODEL_UPDATE_DIR) 
    print("Creating directory if it doesn't exist: " + cfg.MODEL_UPDATE_DIR)
    os.makedirs(cfg.MODEL_UPDATE_DIR, exist_ok=True)
    subfolders, files = scanDirectory(cfg.MODEL_UPDATE_DIR, [".json"])
    print("Total number of files: ", len(files)) 

    for file in files:
        file_details = file.split("/")
        print("File Names: " + cfg.MODEL_UPDATE_DIR + file_details[-1]) 

    if (len(files) >= cfg.MODEL_UPDATE_MIN_THRESHOLD):
        ## Update model here
        print("Updating model")
    else:
        print("Waiting for more files for model update operation")