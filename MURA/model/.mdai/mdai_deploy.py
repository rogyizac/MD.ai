# Import python packages here
import torch
from torchvision import transforms

import os
from io import BytesIO
import numpy as np
from PIL import Image
import pydicom

# Import methods from helper files if any here
from densenet import densenet169


class MDAIModel:
    def __init__(self):

        # set gpu device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        modelpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model.pth")
        self.model = densenet169()
        self.model.load_state_dict(torch.load(modelpath))
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

            # repeat for 3 channels
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

            # Paste code for preprocessing the image here
            preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

            image = preprocess(image)
            image = image.unsqueeze(0)
            image = image.to(self.device)

            # Paste code for passing the image through the model
            output = self.model(image)
            # and generating predictions here
            predicted_prob = output.item()
            threshold = 0.5
            predicted_class = 1 if predicted_prob >= threshold else 0
            predicted_prob = 1 - predicted_prob if predicted_class == 0 else predicted_prob
            predicted_prob = round(predicted_prob,2)

            result = {
                "type": "ANNOTATION",
                "study_uid": str(ds.StudyInstanceUID),
                "series_uid": str(ds.SeriesInstanceUID),
                "instance_uid": str(ds.SOPInstanceUID),
                "class_index": int(predicted_class),
                "probability": [{"class_index" : predicted_class, "probability" : predicted_prob}]
            }

            outputs.append(result)
        return outputs
