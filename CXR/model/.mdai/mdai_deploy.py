import os
from io import BytesIO
from PIL import Image
import pydicom
import numpy as np
import copy

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.ndimage import binary_dilation
from skimage import measure, draw

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

CLASSES = [ 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
       'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
       'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia',
       'Pneumothorax']

intensity_th = 0.9
img_width_exp, img_height_exp = 1024, 1024
img_resize = 256
crop = 224
transResize = 256
transCrop = 224

class DenseNet121(nn.Module):

    def __init__(self, classCount, isTrained):
	
        super(DenseNet121, self).__init__()
		
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)

        kernelCount = self.densenet121.classifier.in_features
		
        self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet121(x)
        return x
    
    
class MDAIModel:
    def __init__(self):

        # set gpu device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        modelpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model.pth.tar")
        self.model = DenseNet121(classCount = 15, isTrained = False)
        self.model = torch.nn.DataParallel(self.model).cuda()
        self.modelCheckpoint = torch.load(modelpath)
        self.model.load_state_dict(self.modelCheckpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        
    def predict(self, data):
        """
        See https://github.com/mdai/model-deploy/blob/master/mdai/server.py for details on the
        schema of `data` and the required schema of the outputs returned by this function.
        """
        input_files = data["files"]
        input_annotations = data["annotations"]
        input_args = data["args"]

        outputs = []

        for file in input_files:
            if file["content_type"] == "application/dicom":
                ds = pydicom.dcmread(BytesIO(file["content"]))
                image = ds.pixel_array
            elif file["content_type"] in ["image/jpeg", "image/png"]:
                ds = file["dicom_tags"]
                image = Image.open(BytesIO(file["content"]))
                image = np.asarray(image)
            else:
                continue
            
            # calculating crop sizes and rescale factors. use to recalculate bbox coordinates for original image size
            crop_del = int((img_resize - crop)/2)
            rescale_factor = int(image.shape[0]/img_resize)

            # repeat for 3 channels
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

            # Paste code for preprocessing the image here
            normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            transformList = []
            transformList.append(transforms.ToPILImage())
            transformList.append(transforms.Resize(transResize))
            transformList.append(transforms.CenterCrop(transCrop))
            transformList.append(transforms.ToTensor())
            transformList.append(normalize)  
            transformSequence=transforms.Compose(transformList)

            image = transformSequence(image)
            image = image.unsqueeze(0)
            image = image.to(self.device)

            # Paste code for passing the image through the model
            with torch.no_grad(): # Added this line
                output = self.model(image)
            # and generating predictions here
            # threshold = 0.5
            thresholds = torch.tensor([0.09353913,
                         0.020909226,
                         0.049312092,
                         0.024079245,
                         0.089281775,
                         0.018816797,
                         0.024021594,
                         0.0020436728,
                         0.18328501,
                         0.03943615,
                         0.60067524,
                         0.047590118,
                         0.028684067,
                         0.013894289,
                         0.048347842]).cuda()
            
            filtered_values = output[output > thresholds]
            filtered_indices = (output > thresholds).flatten().nonzero().flatten().tolist()
            filtered_labels = [CLASSES[i] for i in filtered_indices]
            
            # if no labels are picked then go with "No Finding"
            if len(filtered_values) == 0:
                filtered_values = [output.detach().cpu().numpy().flatten().tolist()[10]]
                filtered_indices = [10]
                filtered_labels = ['No Finding']
                
            # Zip the three lists together
            zipped_list = list(zip(filtered_values, filtered_indices, filtered_labels))

            # Sort by probabilities
            zipped_list.sort(key=lambda x: x[0], reverse=True)

            # Check if the label with the highest probability is 'no finding'
            if zipped_list[0][2] == 'No Finding':
                # If so, keep only this label, its probability and index
                zipped_list = [zipped_list[0]]
            else:
                # If 'no finding' is not the label with the highest probability, but its probability is > 0.5, remove it
                zipped_list = [item for item in zipped_list if item[2] != 'No Finding' or item[0] <= 0.5]

            # Unzip the list back into individual lists
            filtered_values, filtered_indices, filtered_labels = zip(*zipped_list)
            model_copy = copy.deepcopy(self.model)
            target_layers = [model_copy.module.densenet121.features.denseblock4.denselayer16]
            # Construct the CAM object once, and then re-use it on many images:
            cam = GradCAM(model=model_copy, target_layers=target_layers, use_cuda=True)
            
            heatmap_output = []
            for target in filtered_indices:
                targets = [ClassifierOutputTarget(target)]
                heatmap = cam(input_tensor=image, targets=targets)
                heatmap = heatmap[0, :]
                heatmap_output.append(heatmap)
                
            prediction_dict = {}
            for i in range(len(input_files)):
                prediction_dict[i] = []
                
            for k, npy, cls_prob in zip(filtered_indices, heatmap_output, filtered_values):

                # Define a threshold for 'high activation'
                threshold = 0.7
    
                # Use the threshold to create a binary map
                binary_map = (npy >= threshold).astype(np.int)
                
                # Label the regions in the binary map
                labels = measure.label(binary_map)
                
                # Iterate over the detected regions
                for region in measure.regionprops(labels):
                    # Get coordinates and dimensions of bounding box
                    minr, minc, maxr, maxc = region.bbox
                    x, y = minc, minr
                    width, height = maxc - minc, maxr - minr
                    
                    prediction_sent = '%d %.2f %.1f %.1f %.1f %.1f' % (k, cls_prob, x*rescale_factor,
                                                           y*rescale_factor,
                                                           width*rescale_factor,
                                                           height*rescale_factor)
                    img_id = 0
                    prediction_dict[img_id].append(prediction_sent)

            # loop through outputs and return
            for i in range(len(prediction_dict)):
                prediction = prediction_dict[i]
                print(prediction)
                for pred in prediction:
                    result = {
                        "type": "ANNOTATION",
                        "study_uid": str(ds.StudyInstanceUID),
                        "series_uid": str(ds.SeriesInstanceUID),
                        "instance_uid": str(ds.SOPInstanceUID),
                        "class_index": int(pred.split(" ")[0]), # Enter ouput class index here,
                        "probability": float(pred.split(" ")[1]),# Add probility value here if generated model else delete field,
                        "data": {"x": int(float(pred.split(" ")[2])), "y": int(float(pred.split(" ")[3])), "width": int(float(pred.split(" ")[4])), "height": int(float(pred.split(" ")[5]))}
                    }

                    outputs.append(result)
        # if output label has "No Finding" Keep only that label and remove rest
        if outputs[0]['class_index'] == 10:
            if len(outputs) > 1:
                outputs = outputs[:1]
        # keep only top 3 labels
        if len(outputs) > 1:
            outputs = outputs[:3]
            
        del model_copy
                    
        return outputs
