import json
import argparse
import glob
import datetime
import os
import shutil
import mimetypes
from cv2 import VideoCapture
import traceback
mimetypes.init()

from ground_truth.src.calculate_stats import mean_average_precision
from ground_truth.src.calculate_stats import get_attribute_stats
from ground_truth.src.calculate_stats import get_overall_stats
from ground_truth.src.run_model import perform_video_od


def parse_args():
    '''
    This script will generate evaluation statistics for a given model and store them in a json.
    '''
    parser = argparse.ArgumentParser(
        description='Evaluate a models performance')
    parser.add_argument('video_id', help='unique id for saving video and video info')
    parser.add_argument('--folder', default='src/ground_truth/videos', help='path/to/folder/of/videos')
    parser.add_argument('--gen-video', default=False, action='store_true', help='generates video overlay')
    parser.add_argument('--gt', default='src/ground_truth/jsons/ground_truth', help='path/to/ground/truth/folder')
    parser.add_argument('--pred', default='src/ground_truth/jsons/predictions', help='path/to/store/predictions')
    parser.add_argument('--config', default='model_artifacts', help='path/to/config/folder')
    parser.add_argument('--checkpoint', default='model_artifacts', help='path/to/checkpoint/folder')
    args = parser.parse_args()
    return args


def evaluate_model(video_id, folder='', gen_video=False, gt='', pred='', config='', checkpoint=''):

    # make directory to store predictions
    if not os.path.exists(pred):
        os.makedirs(pred)

    # run model to generate predictions
    perform_video_od(video_id, gen_video, folder, config, checkpoint, pred)

    # load model predictions and ground_truths
    with open(f'{pred}/{video_id}_predictions.json') as file:
        predictions = json.load(file)

    with open(f'{gt}/{video_id}.json') as file:
        ground_truth = json.load(file)

    # compute map
    mean_average_precision(pred_boxes=predictions, true_boxes=ground_truth, iou_threshold=0.5, 
                            box_format="midpoint", num_classes=7, video_id=video_id, json_store=video_stats, folder=folder)


if __name__ == '__main__':

    args = parse_args()

    # dicts for storing precision, recall and map for each video (video stats), and each attribute (attribute_stats)
    video_stats = {}
    attribute_stats = {}

    # create directory to store performance data
    storage_path = f'src/ground_truth/{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}_model'

    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    # read video_attributes
    with open('src/ground_truth/jsons/ground_truth/video_attributes.json') as file:
        attributes = json.load(file)   


    #iterate videos folder and run evaluate_model on each video
    for subdir, dirs, files in os.walk(args.folder):
        print('iterating all files in sub directories looking for videos...')
        for file in files:

            filepath = subdir + os.sep + file

            mimestart = mimetypes.guess_type(filepath)[0]

            if mimestart != None:
                mimestart = mimestart.split('/')[0]

                #if file is a video
                if mimestart in ['video']:
                    #verify its a working video 
                    try:
                        capture = VideoCapture(filepath)
                        print(filepath)
                        evaluate_model(os.path.splitext(file)[0], filepath, args.gen_video, 
                                            args.gt, args.pred, args.config, args.checkpoint)   
                    except Exception as e:
                        print(f"broken video: {filepath}")
                        print(e)
                        print(traceback.format_exc())


    # calculate stats for each attribute and all videos combined
    get_attribute_stats(attributes, video_stats, attribute_stats)
    get_overall_stats(video_stats, video_stats)

    # aggregate stats for videos and attributes into one dictionary
    performance = {
        'videos': video_stats,
        'attributes': attribute_stats
    }

    # write performance data to storage path
    with open(f'{storage_path}/performance.json', 'w+') as file:
        json.dump(performance, file)

    shutil.copy(glob.glob(f"{args.config}/*.py")[0], storage_path)
    shutil.copy(glob.glob(f"{args.checkpoint}/*.pth")[0], storage_path)

    # remove folder storing predictions
    shutil.rmtree(args.pred)

