import torch
from torch.nn import DataParallel

import os
import cv2
import pydicom
import numpy as np
from io import BytesIO
from PIL import Image

import albumentations as A
from albumentations.pytorch.transforms import ToTensor

# helpers
from src.inference import PytorchInference
from src.models.unet import ResnetSuperVision


class MDAIModel:

    # configs
    seg_classes = 1
    backbone_arch = 'resnet34'

    empty_threshold = 0.2
    empty_score_threshold = 0.8
    area_threshold = 400
    mask_score_threshold = 0.4

    EMPTY = '-1'
    FINAL_SIZE = (1024, 1024)

    def __init__(self):

        # set gpu device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        modelpath1 = os.path.join(os.path.dirname(os.path.dirname(__file__)), "epoch_58_fold0.pth")
        modelpath2 = os.path.join(os.path.dirname(os.path.dirname(__file__)), "epoch_59_fold1.pth")
        modelpath3 = os.path.join(os.path.dirname(os.path.dirname(__file__)), "epoch_58_fold2.pth")
        modelpath4 = os.path.join(os.path.dirname(os.path.dirname(__file__)), "epoch_59_fold3.pth")

        self.model1 = ResnetSuperVision(self.seg_classes, self.backbone_arch)
        self.model2 = ResnetSuperVision(self.seg_classes, self.backbone_arch)
        self.model3 = ResnetSuperVision(self.seg_classes, self.backbone_arch)
        self.model4 = ResnetSuperVision(self.seg_classes, self.backbone_arch)

        self.model1.load_state_dict(torch.load(modelpath1))
        self.model2.load_state_dict(torch.load(modelpath2))
        self.model3.load_state_dict(torch.load(modelpath3))
        self.model4.load_state_dict(torch.load(modelpath4))

        self.model1 = DataParallel(self.model1).to(self.device)
        self.model2 = DataParallel(self.model2).to(self.device)
        self.model3 = DataParallel(self.model3).to(self.device)
        self.model4 = DataParallel(self.model4).to(self.device)

        self.model1.eval()
        self.model2.eval()
        self.model3.eval()
        self.model3.eval()

        self.models = [self.model1, self.model2, self.model3, self.model4]

    def predict(self, data):
        """
        See https://github.com/mdai/model-deploy/blob/master/mdai/server.py for details on the
        schema of `data` and the required schema of the outputs returned by this function.
        """
        input_files = data["files"]
        input_annotations = data["annotations"]
        input_args = data["args"]

        outputs_mdai = []

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

            # Assuming image is your grayscale image
            image = np.stack((image,)*3, axis=-1)

            # transforms
            transforms_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "val_transforms_768.yaml")
            _transform = A.load(transforms_config_path, 'yaml')
            _transform = A.Compose([_transform, ToTensor()])
            transformed_image = _transform(image = image)
            transformed_image = transformed_image['image'].to(self.device)
            transformed_image = transformed_image.unsqueeze(0)

            def to_numpy(images):
                return images.data.cpu().numpy()

            # Ensemble
            runner = PytorchInference(self.device)
            outputs = {"predictions":[], "empty":[]}
            for model in self.models:
                prediction, empty = runner.tta(model, transformed_image)
                outputs["predictions"].append(prediction)
                outputs["empty"].append(empty)

            # Stack tensors along a new dimension
            stacked_tensors = torch.stack(outputs["predictions"])

            # Calculate the mean along that new dimension
            prediction = torch.mean(stacked_tensors, dim=0)
            empty = torch.mean(torch.stack(outputs['empty']))

            prediction = np.moveaxis(to_numpy(prediction), 0, -1)
            empty = to_numpy(empty)
            
            
            # if total segmentated area is < than threshold area
            if np.sum(prediction > self.empty_score_threshold) < self.area_threshold:
                mask = np.zeros(prediction.shape)
                mask = np.array(mask > self.mask_score_threshold).astype(np.uint8)
            mask = np.array(prediction > self.mask_score_threshold).astype(np.uint8)
            
            mask = mask.squeeze()
            mask = cv2.resize(src=mask * 255, dsize=self.FINAL_SIZE, interpolation=cv2.INTER_NEAREST)
            
            if mask.sum() == 0:
                predicted_class = 0
            else:
                predicted_class = 1
            predicted_prob = empty
            
            predicted_prob = 1 - predicted_prob if predicted_class == 0 else predicted_prob
            # predicted_prob = np.round(predicted_prob,2)
            predicted_prob = float(np.round(predicted_prob,2))

            result = {
                "type": "ANNOTATION",
                "study_uid": str(ds.StudyInstanceUID),
                "series_uid": str(ds.SeriesInstanceUID),
                "instance_uid": str(ds.SOPInstanceUID),
                "class_index": int(predicted_class),
                "data": {"mask": mask.tolist()},
                "probability": [{"class_index" : predicted_class, "probability" : predicted_prob}]
            }

            outputs_mdai.append(result)
        return outputs_mdai
