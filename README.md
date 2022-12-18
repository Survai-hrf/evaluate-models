# SurvAI Model Evaluation Repository

This repo contains tools for evaluating SurvAI’s models by generating evaluation metrics such as precision, recall, and mean average precision (mAP). These metrics provide a data-driven way to compare a models accuracy/performance to other models. The repo is designed to be run on the same folder of videos each time. 

Each video has its own ground truth json located in src/ground_truth/data/ground_truth that contains the correct labels for each frame. Using ‘eval.py’ in the ‘src’ folder, SurvAI's object detection model will be ran on each video in the ‘videos’ folder and generate a predictions json in the same format as the ground truth. The json format is as follows:
```json
[
    "train_index (int)",
    "class_prediction (int)",
    "confidence_score (float 0-1)",
    "x1 (int)",
    "y1 (int)",
    "x2 (int)",
    "y2 (int)"
]
```
The script will then compare the predicted labels to the ground truth labels to calculate precision, recall, and mAP. 

To provide further information about a models performance, each video has had its attributes tagged (such as night/day time, violent/non-violent, weather, etc.), allowing for metrics to be generated for each tag. Knowing the specific video contexts that a given model struggles/excels on allows the SurvAI team to collect data to cover these weaknesses. Each videos respective tags can be found in src/ground_truth/data/ground_truth/video_attributes.json, and a full list of possible attributes can be found in attributes_list.json in the root directory. 

After running the 'eval.py' script, a models results are stored in a dated folder in the root directory that contains the model config and weights that were used, as well as a ‘performance.json’ file containing the metrics for each video. As of 12/14/2022, this repo only generates metrics for SurvAI’s object detection model.

## Environment

This repo uses the same environment used in SurvAI’s ml-deployment repo. If the 'sdeploy' environment used in ml-deployment is already set up on your local machine, the following instructions can be skipped.

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

2. ) Cd into the 'evaluate-models' repo that was just cloned and run:
- ```python src/eval.py```

eval.py accepts the following arguments:
- video_id (required, automatically generated)
- --folder
- --gen-video
- --gt
- --pred
- --config
- --checkpoint

### --folder
Specifies the path to the local folder containing the ground truth videos. The script will automatically filter out other file types, so it is okay to pass a folder containing other files like txt or csv. Default is:
- ```src/ground_truth/videos```

### --gen-video
Adding to the command line will output the input video with overlayed bounding boxes into a ‘video_overlays’ folder in the root directory of this repo.

### --gt
Specifies the path to the folder containing each videos ground truth json. Default is:
- ```src/ground_truth/jsons/ground_truth```

### --pred
Specifies the path to store predicted labels. Default is:
- ```src/ground_truth/jsons/predictions```

### --config
Specifies the path to folder containing model config. Default is:
- ```model_artifacts```

### --checkpoint
Specifies the path to the folder containing model checkpoints. Default is:
- ```model_artifacts```

NOTE: These defaults will work if the repo is consistent with the format on github. Any changes in the format of the ground truth data will require you to specify these changes with their respective arguments.

## Adding Videos to the Ground Truth Dataset
1. ) Use ‘extract_frames.ipynb’ in the ‘tools’ folder to extract the individual frames from the video you wish to add to the dataset.

2. ) Label the objects in each frame and export the annotations in coco json format. While coco format is not required, if a different format is used, a new script will need to be written to convert the non coco json to the required data format for ‘eval.py’.

3. ) For ground truth annotations in coco format, use ‘coco_to_ground_truth.ipynb’ in the ‘tools’ folder to convert the coco json to the required format. 

4. ) Add the output json from ‘coco_to_ground_truth.ipynb’ to src/ground_truth/data/ground_truth

5. ) To ‘attributes.json’, add the video with its tags in the following format:
```json
{
    "video_id": [
        "attribute 1",
        "attribute 2"
    ]
}
```
Please reference ‘attributes_list.txt’ file for a list of all possible video attributes.

DON'T include videos with the following:
- Montages (ex: compilations)
- Professional footage (ex: local news footage)
- Low-hanging fruit (ex: 20 second video of a giant brawl)
- Don’t include any videos that are already in the training dataset
- Civilian misconduct outside of verified hate group or known violent group
- Shorter than 1:00 long

