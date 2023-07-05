import numpy as np
from os import listdir
import skimage.transform
from io import BytesIO

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import cv2
import os
from collections import OrderedDict

import skimage
from skimage.io import *
from skimage.transform import *

import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.ndimage import binary_dilation
import glob
import pydicom
import re

# Import methods from helper files if any here
from common.config.config import Config as cfg

intensity_th = 0.9
img_width_exp, img_height_exp = 1024, 1024
img_resize = 256
crop = 224


class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()

        if torch.cuda.is_available():
            print("Cuda is available. Model will run on GPU.")
            device = torch.device("cuda")
        else:
            print("No GPU found. Device will run on CPU.")
            device = torch.device("cpu")

        self.densenet121 = torchvision.models.densenet121(weights=None)
        pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
        )

        state_dict = torch.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), "densenet121-a639ec97.pth"), map_location=device)

        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        self.densenet121.load_state_dict(state_dict)

        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


class ChestXrayDataSet_plot(Dataset):
    def __init__(self, test_X, transform=None):
        self.X = np.uint8(test_X*255)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image
        """
        current_X = np.tile(self.X[index],3)
        image = self.transform(current_X)
        return image

    def __len__(self):
        return len(self.X)


# ======= Grad CAM Function =========
class PropagationBase(object):

    def __init__(self, model, cuda=False):
        self.model = model
        self.model.eval()
        if cuda:
            self.model.cuda()
        self.cuda = cuda
        self.all_fmaps = OrderedDict()
        self.all_grads = OrderedDict()
        self._set_hook_func()
        self.image = None

    def _set_hook_func(self):
        raise NotImplementedError

    def _encode_one_hot(self, idx):
        one_hot = torch.FloatTensor(1, self.preds.size()[-1]).zero_()
        one_hot[0][idx] = 1.0
        return one_hot.cuda() if self.cuda else one_hot

    def forward(self, image):
        self.image = image
        self.preds = self.model.forward(self.image)
#         self.probs = F.softmax(self.preds)[0]
#         self.prob, self.idx = self.preds[0].data.sort(0, True)
        return self.preds.cpu().data.numpy()

    def backward(self, idx):
        self.model.zero_grad()
        one_hot = self._encode_one_hot(idx)
        self.preds.backward(gradient=one_hot, retain_graph=True)


class GradCAM(PropagationBase):

    def _set_hook_func(self):

        def func_f(module, input, output):
            self.all_fmaps[id(module)] = output.data.cuda()

        def func_b(module, grad_in, grad_out):
            self.all_grads[id(module)] = grad_out[0].cuda()

        for module in self.model.named_modules():
            module[1].register_forward_hook(func_f)
            module[1].register_backward_hook(func_b)

    def _find(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('Invalid layer name: {}'.format(target_layer))

    def _normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm.item()

    def _compute_grad_weights(self, grads):
        grads = self._normalize(grads)
        self.map_size = grads.size()[2:]
        return nn.AvgPool2d(self.map_size)(grads)

    def generate(self, target_layer):
        fmaps = self._find(self.all_fmaps, target_layer)
        grads = self._find(self.all_grads, target_layer)
        weights = self._compute_grad_weights(grads)
        gcam = torch.FloatTensor(self.map_size).zero_().cuda()
        for fmap, weight in zip(fmaps[0], weights[0]):
            gcam += fmap * weight.data

        gcam = F.relu(Variable(gcam))

        gcam = gcam.data.cpu().numpy()
        gcam -= gcam.min()
        gcam /= gcam.max()
        gcam = cv2.resize(gcam, (self.image.size(3), self.image.size(2)))

        return gcam

    def save(self, filename, gcam, raw_image):
        gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
        gcam = gcam.astype(np.float) + raw_image.astype(np.float)
        gcam = gcam / gcam.max() * 255.0
        cv2.imwrite(filename, np.uint8(gcam))


class MDAIModel:

    def __init__(self):

        if torch.cuda.is_available():
            print("Cuda is available. Model will run on GPU.")
            device = torch.device("cuda")
        else:
            print("No GPU found. Device will run on CPU.")
            device = torch.device("cpu")

        modelpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "DenseNet121_aug4_pretrain_WeightBelow1_1_0.829766922537.pkl")
        pkl_files = glob.glob(modelpath)
        pkl_file = pkl_files[0]

        self.model = DenseNet121(8)  # Load model here
        self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load(pkl_file, map_location=device), strict=False)

    def predict(self, data):
        """
        See https://github.com/mdai/model-deploy/blob/master/mdai/server.py for details on the
        schema of `data` and the required schema of the outputs returned by this function.
        """
        input_files = data["files"]
        input_annotations = data["annotations"]
        input_args = data["args"]

        outputs = []
        test_X = []
        thresholds = cfg.CHEXNET_THRESHOLDS
        for file in input_files:
            if file["content_type"] == "application/dicom":
                ds = pydicom.dcmread(BytesIO(file["content"]))
                image = ds.pixel_array.astype(np.uint8)
            elif file["content_type"] in ["image/jpeg", "image/png"]:
                ds = file["dicom_tags"]
                image = Image.open(BytesIO(file["content"]))
                image = np.asarray(image)
            else:
                continue

            # Paste code for preprocessing the image here
            if image is not None:
                # if image.shape != (img_width_exp,img_height_exp):
                if len(image.shape)==3:
                    img = image[:,:,0]
                else:
                    img = image[:,:]

            crop_del = int((img_resize - crop)/2)
            rescale_factor = int(image.shape[0]/img_resize)

            img_resized = skimage.transform.resize(img,(img_resize,img_resize))
            test_X.append((img_resized).reshape(img_resize,img_resize,1))
            test_X = np.array(test_X)
            
            # preprocess image transforms
            test_dataset = ChestXrayDataSet_plot(test_X = test_X,transform=transforms.Compose([
                                                    transforms.ToPILImage(),
                                                    transforms.CenterCrop(crop),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                    ]))
            # initialize
            heatmap_output = []
            image_id = []
            output_class = []

            # gcam = GradCAM(model=model, cuda=True)
            gcam = GradCAM(model=self.model, cuda=True)
            for index in range(len(test_dataset)):
                input_img = Variable((test_dataset[index]).unsqueeze(0), requires_grad=True)
                probs = gcam.forward(input_img)
                sorted_index_probs = np.argsort(probs)
                sorted_probs = np.take(probs, sorted_index_probs)
                # n = 3
                n = cfg.MODEL_UPDATE_MIN_THRESHOLD 
                top_n_probs = np.expand_dims(sorted_probs[0, -n:], 0)
                print(f"Probabilities : {probs}")
                print(f"Sorted Index : {sorted_index_probs}")
                print(f"Sorted Probs : {sorted_probs}")
                print(f"Top N = {n} Probs : {top_n_probs}")
                print(f"Thresholds : {thresholds}")
                sorted_threshold = np.take(thresholds, sorted_index_probs[0])
                top_n_threhold = sorted_threshold[-n:]
                print(f"Sorted Threshold : {sorted_threshold}")
                print(f"Top N Threshold : {top_n_threhold}")

                activate_classes = sorted_index_probs[top_n_probs > top_n_threhold]
                activate_classes_probs = top_n_probs[top_n_probs > top_n_threhold]

                print(f"Activated Classes : {activate_classes}")
                for activate_class in activate_classes:
                    gcam.backward(idx=activate_class)
                    output = gcam.generate(target_layer="module.densenet121.features.denseblock4.denselayer16")
                    #### this output is heatmap ####
                    if np.sum(np.isnan(output)) > 0:
                        print("fxxx nan")
                    heatmap_output.append(output)
                    image_id.append(index)
                    output_class.append(activate_class)

            # ======= Plot bounding box =========
            img_width, img_height = crop, crop

            class_index = ['Atelectasis', 'Cardiomegaly', 'Effusion',
                           'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                           'Pneumothorax']

            prediction_dict = {}
            for i in range(len(input_files)):
                prediction_dict[i] = []

            for img_id, k, npy, cls_prob in zip(image_id, output_class, heatmap_output, activate_classes_probs):

                data = npy

                if np.isnan(data).any():
                    continue

                # Find local maxima
                neighborhood_size = 100
                threshold = .1

                data_max = filters.maximum_filter(data, neighborhood_size)
                maxima = (data == data_max)
                data_min = filters.minimum_filter(data, neighborhood_size)
                diff = ((data_max - data_min) > threshold)
                maxima[diff == 0] = 0
                for _ in range(5):
                    maxima = binary_dilation(maxima)

                labeled, num_objects = ndimage.label(maxima)
                xy = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects+1)))

                # create pixel threshold based on intensity
                thresholded_data_l = (data > np.max(data)*intensity_th).astype(np.int64)

                for point in xy:
                    centroid_x = int(point[0])
                    centroid_y = int(point[1])

                    if data[centroid_x, centroid_y] > np.max(data)*.9:
                        # find box boundaries
                        left, right, upper, lower = centroid_x, centroid_x, centroid_y, centroid_y

                        # check adjacent pixel value and update coordinate
                        while left > 0 and thresholded_data_l[max(0, left), centroid_y] == 1:
                            left -= 1
                        while right < crop and thresholded_data_l[min(crop, right), centroid_y] == 1:
                            right += 1
                        while upper > 0 and thresholded_data_l[centroid_x, max(0, upper)] == 1:
                            upper -= 1
                        while lower < crop and thresholded_data_l[centroid_x, min(crop, lower)] == 1:
                            lower += 1

                        prediction_sent = '%d %.2f %.1f %.1f %.1f %.1f' % (k, cls_prob, (left+crop_del)*rescale_factor,
                                                                           (upper+crop_del)*rescale_factor,
                                                                           (right-left)*rescale_factor,
                                                                           (lower-upper)*rescale_factor)
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
        return outputs