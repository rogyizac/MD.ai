# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:57:14 2023

@author: rohan
"""

import torch
from torchvision import transforms
import imageio
import torch.nn.functional as F
# Import methods from helper files if any here
from densenet import densenet169
from PIL import Image
import numpy as np
import cv2


# set gpu device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

modelpath = r'C:\IUPUI\PLHILab\AIinRadiology\MDAI-skeleton-code\model\.mdai\model.pth'
model = densenet169()
model.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))
model.to(device)
model.eval()

# image = Image.open(r'C:\IUPUI\PLHILab\MURA-v1.1\MURA-v1.1\train\XR_ELBOW\patient00011\study1_negative\image2.png')
image = cv2.imread(r'C:\IUPUI\PLHILab\MURA-v1.1\MURA-v1.1\train\XR_ELBOW\patient00011\study1_negative\image2.png')


# Paste code for preprocessing the image here
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

image = preprocess(image)
image = image.unsqueeze(0)
image = image.to(device)


# Paste code for passing the image through the model
output = model(image)
# and generating predictions here
predicted_prob = output.item()
threshold = 0.5
predicted_class = 1 if predicted_prob >= threshold else 0

output.item()
