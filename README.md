# Glimpse Model Evaluation Repository

This repo contains tools for evaluating Glimpse’s models by generating evaluation metrics such as precision, recall, and mean average precision (mAP). These metrics provide a data-driven way to compare a models accuracy/performance to other models. The repo is designed to be run on the same folder of videos each time. 

Each video has its own ground truth json located in src/ground_truth/data/ground_truth that contains the correct labels for each frame. Using ‘eval.py’ in the ‘src’ folder, Glimpse's object detection model will be ran on each video in the ‘videos’ folder and generate a predictions json in the same format as the ground truth. The json format is as follows:
```json
[
    [
    "train_index (int)",
    "class_prediction (int)",
    "confidence_score (float 0-1)",
    "x1 (int)",
    "y1 (int)",
    "x2 (int)",
    "y2 (int)"
    ]
]
```
The script will then compare the predicted labels to the ground truth labels to calculate precision, recall, and mAP. 

To provide further information about a models performance, each video has had its attributes tagged (such as night/day time, violent/non-violent, weather, etc.), allowing for metrics to be generated for each tag. Knowing the specific video contexts that a given model struggles/excels on allows the Glimpse team to collect data to cover these weaknesses. Each videos respective tags can be found in src/ground_truth/data/video_attributes.json, and a full list of possible attributes can be found in attributes_list.txt in the root directory. 

After running the 'eval.py' script, a models results are stored in a dated folder in the root directory that contains the model config and weights that were used, as well as a ‘performance.json’ file containing the metrics for each video. As of 12/14/2022, this repo only generates metrics for Glimpse’s object detection model.

## Environment

This repo uses the same environment used in Glimpse’s ml-deployment repo. If the 'sdeploy' environment used in ml-deployment is already set up on your local machine, the following instructions can be skipped.

### Requirements

- Ubuntu 20.04
- CUDA 11.3
- Conda (Up to date)
- (optional, if training)  RTX 3080 GPU or > 

### Install Instructions

First, make sure CUDA 11.3 has been properly installed at the system level of your workstation. There is a variety of ways to do it, 
but ensure that while in your conda environment (this environment is created in the next step) the command
``` torch.cuda.is_available() ``` returns True.

1. ) Some packages are needed at the system level of Ubuntu to properly crop videos and local install model packages. In a bash terminal run:
     - ```apt-get install ffmpeg libsm6 libxext6 gcc ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6  -y```
     
2. ) With miniconda3 installed, at the root directory of evaluate-models run:

     -  ```conda env create -f environment.yml```
     -  ```conda activate sdeploy```
     -  ```MMCV_WITH_OPS=1 pip install -e src/mmcv/```
     -  ```pip install -v -e src/mmaction2/```
     -  ```pip install -v -e src/mmdetection/```
     
          **ADVISORY:**  If you install any more packages into this environment, be sure to use pip. Using conda will cause package conflicts and break the environment.

## Running eval.py
1. ) In a terminal, cd’ed to the desired directory to store the repo, run:
- ```git clone https://github.com/Survai-hrf/evaluate-models.git```

2. ) Access the google cloud [survai-dataset](https://console.cloud.google.com/storage/browser/survai-dataset). Inside model_artifacts/od, download:
- ```MAIN_epoch_54.pth (or the model weights that you wish to test)```
- ```mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py```

Place these files in the 'model_artifacts' folder inside the root directory of the evaluate-models repo.

3. ) Cd into the 'evaluate-models' repo that was just cloned and run:
- ```python src/eval.py 0```
NOTE: '0' is a placeholder for the video_id argument, which will eventually get overwritten by the real file name when the sript iterates through the 'videos' folder.

### Arguments
eval.py accepts the following arguments:
- video_id (required, automatically generated)
- --folder
- --gen-video
- --gt
- --pred
- --config
- --checkpoint

#### --folder
Specifies the path to the local folder containing the ground truth videos. The script will automatically filter out other file types, so it is okay to pass a folder containing other files like txt or csv. Default path is:
- ```videos```

#### --gen-video
Adding to the command line will output the input video with overlayed bounding boxes into a ‘video_overlays’ folder in the root directory of this repo.

#### --gt
Specifies the path to the folder containing each videos ground truth json. Default path is:
- ```src/ground_truth/data/ground_truth```

#### --pred
Specifies the path to store predicted labels. Default path is:
- ```src/ground_truth/data/predictions```

#### --config
Specifies the path to folder containing model config. Default path is:
- ```model_artifacts```

#### --checkpoint
Specifies the path to the folder containing model checkpoints. Default path is:
- ```model_artifacts```

### NOTES
- These defaults will work if the repo is consistent with the format on github. Any changes to the format of the ground truth data will require you to specify these changes using their respective arguments in the command line when running 'eval.py'.
- As of 12/22/2022, there is only one video in the ground_truth dataset, so each attributes metrics will all be the same until a new video is added. 

## Adding Videos to the Ground Truth Dataset
1. ) Use ‘extract_frames.ipynb’ in the ‘tools’ folder to extract the individual frames from the video you wish to add to the ground truth dataset.

2. ) Label the objects in each frame using your preferred annotation tool/service and export the annotations in COCO json format. While COCO format is not required, if a different format is used, a new script will need to be written to convert the non-COCO json to the required data format for ‘eval.py’ (see the format depicted in the top section).

3. ) For ground truth annotations in COCO format, use ‘coco_to_ground_truth.ipynb’ in the ‘tools’ folder to convert the COCO json to the required format. 

4. ) Add the json that is generated using ‘coco_to_ground_truth.ipynb’ to src/ground_truth/data/ground_truth

5. ) To ‘attributes.json’, add the video with its attribute tags in the following format:
```json
{
    "video_id_1": [
        "attribute 1",
        "attribute 2",
        "attribute 3..."
    ],
    "video_id_2": [
        "attribute 1",
        "attribute 2",
        "attribute 3..."
    ]
}
```
Please reference ‘attributes_list.txt’ for a list of all possible video attributes.

6. ) Add the video to the 'videos' folder located in the root directory.

DO NOT include videos with the following:
- Montages (ex: compilations)
- Professional footage (ex: local news footage)
- Low-hanging fruit (ex: 20 second video of a giant brawl)
- Don’t include any videos that are already in the training dataset
- Civilian misconduct outside of verified hate group or known violent group
- Shorter than 1:00 long

