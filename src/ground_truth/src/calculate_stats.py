import torch
from collections import Counter
from moviepy.editor import VideoFileClip




def intersection_over_union(box_preds, box_labels, box_format='midpoint'):
    # box_preds shape is (N, 4), where N is the number of bboxes
    # box_labels shape is (N, 4)

    if box_format == 'midpoint':
        box1_x1 = box_preds[..., 0:1] - box_preds[..., 2:3] / 2
        box1_y1 = box_preds[..., 1:2] - box_preds[..., 3:4] / 2
        box1_x2 = box_preds[..., 0:1] + box_preds[..., 2:3] / 2
        box1_y2 = box_preds[..., 1:2] + box_preds[..., 3:4] / 2
        box2_x1 = box_labels[..., 0:1] - box_labels[..., 2:3] / 2
        box2_y1 = box_labels[..., 1:2] - box_labels[..., 3:4] / 2
        box2_x2 = box_labels[..., 0:1] + box_labels[..., 2:3] / 2
        box2_y2 = box_labels[..., 1:2] + box_labels[..., 3:4] / 2

    elif box_format == 'corners':
        box1_x1 = box_preds[..., 0:1]
        box1_y1 = box_preds[..., 1:2]
        box1_x2 = box_preds[..., 2:3]
        box1_y2 = box_preds[..., 3:4]
        box2_x1 = box_preds[..., 0:1]
        box2_y1 = box_preds[..., 1:2]
        box2_x2 = box_preds[..., 2:3]
        box2_y2 = box_preds[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y1 - box1_y2))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y1 - box2_y2))

    return intersection / (box1_area + box2_area - intersection + 1e-6)



def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20, video_id='', json_store='', folder=''
):
    """
    Calculates mean average precision 
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []
        ious = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )
                
                ious.append(iou)
    

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # false positive if iou is lower than threshold
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

        # get video duration
        clip = VideoFileClip(folder)
        duration = clip.duration

        json_store[video_id] = {
            'precision': precisions.tolist()[len(precisions)-1],
            'recall': recalls.tolist()[len(recalls)-1],
            'map': float(sum(average_precisions) / len(average_precisions)),
            'duration': duration
        }

    #print('ious: ', ious)
    print('precision: ', precisions.tolist()[len(precisions)-1])
    print('recall: ', recalls.tolist()[len(recalls)-1])
        
    #print(average_precisions)
    print('mAP: ', sum(average_precisions) / len(average_precisions))

    return sum(average_precisions) / len(average_precisions)



def get_attribute_stats(attributes, video_stats, json_store):
    '''
    calculate precision, recall, and map for each attribute
    '''

    # for each attribute, create a list of videos that share that attribute
    shared_attributes = {}

    for k,v in attributes.items():
        for attribute in v:
            if attribute in shared_attributes:
                shared_attributes[attribute].append(k)
            else:
                shared_attributes[attribute] = [k]

    # for each video that shares attribute, aggregate their statstics and average them
    for attribute, videos_list in shared_attributes.items():

        precision = []
        recall = []
        map = []

        for video in videos_list:
            data = video_stats.get(video)

            for k,v in data.items():

                if k == 'precision':
                    precision.append(v)
                if k == 'recall':
                    recall.append(v)
                if k == 'map':
                    map.append(v)
        
            # store in dictionary to be exported later
            json_store[attribute] = {
                'precision': sum(precision)/len(precision),
                'recall': sum(recall)/len(recall),
                'map': sum(map)/len(map)
            }



def get_overall_stats(video_stats, json_store):
    '''
    calculate the combined precision, recall, and map
    '''

    precision = []
    recall = []
    map = []

    for video, stats in video_stats.items():
        data = video_stats.get(video)

        for k,v in data.items():

            if k == 'precision':
                precision.append(v)
            if k == 'recall':
                recall.append(v)
            if k == 'map':
                map.append(v)
        
    json_store['overall'] = {
        'precision': sum(precision)/len(precision),
        'recall': sum(recall)/len(recall),
        'map': sum(map)/len(map) 
    }
    



