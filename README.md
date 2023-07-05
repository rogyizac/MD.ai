# MD.ai
A comprehensive sample code that can be referenced for use in MD.ai (https://md.ai/) with respective weights.
Refer the docs at MD.ai for code templates (https://docs.md.ai/models/interface-code/).

MDai Integrations,
- CXR - CheXnet (https://github.com/zoogzog/chexnet) multilabel with gradcam based bounding boxes
- Pneumothorax (https://github.com/amirassov/kaggle-pneumothorax/tree/master) - segmentation model
- CBIS-DDSM - Only Calcification (https://github.com/rogyizac/BreastCancerDetectionCBIS-DDSM) - classification
- MURA (https://github.com/rogyizac/DenseNet-MURA-PyTorch) - classification
  
P.S: Use Local Labels option in MD.ai to enable bounding boxes. Feel free to contact me if you face issues.

You can find deployment instructions in the MD.ai documentations.

You can find the model weights [here](https://app.box.com/s/d6p945s2s7b4d9oqhycsoo77hu092dwm) in this link except for CBIS_DDSM_Calcification since its around 500MB which box doesnt allow me to upload.

For CBIS_DDSM_Calcification weights you can train using the code [here](https://github.com/rogyizac/BreastCancerDetectionCBIS-DDSM) and get the weights.
