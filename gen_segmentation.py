#
# Authors: Wouter Van Gansbeke
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import time
import json
import argparse
import datetime
from termcolor import colored
from tqdm import tqdm
import skimage.io as io
import cv2

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils.voc_classes import *


def get_most_confident_predictions(predictions):
    """ Get most confident predictions.
    """
    memo = {}
    for i, p in enumerate(predictions):
        img_id = p['image_id']
        p['id'] = i
        p['area'] = int(p['bbox'][2]*p['bbox'][3])
        if img_id not in memo or memo[img_id]['score'] < p['score']:
            memo[img_id] = p
    return list(memo.values())


def get_confident_predictions(COCO_pred, threshold):
    """ Get confident predictions based on threshold.
    """
    anns_out = []
    count = 0
    for k, anns in COCO_pred.imgToAnns.items():
        if anns[0]['score'] < threshold:
            anns[0]['id'] = count
            anns_out.append(anns[0])
            count += 1
        else:
            for ann in anns:
                if ann['score'] >= threshold:
                    ann['id'] = count
                    anns_out.append(ann)
                    count += 1
                else:
                    break # anns are sorted
    return anns_out


def get_args_parser():
    parser = argparse.ArgumentParser('Get confident predictions in COCO format', 
            add_help=False)
    parser.add_argument('--input_file', default='/path')
    parser.add_argument('--gt_file', default='/path')
    parser.add_argument('--output_file', default='out_file.json')
    parser.add_argument('--save_dir', default='output_dir/')
    parser.add_argument('--threshold', default=1.0, type=float)
    parser.add_argument('--mask_type', default='polygon', type=str)
    parser.add_argument('--image_dir', default='/VOCdevkit/VOC12/JPEGImages/', type=str,
            help='Path to images, only necessary when grabcut is used.')

    return parser


if __name__ == "__main__":
    start_time = time.time()

    # get arguments
    args = get_args_parser()
    args = args.parse_args()
    print(colored(args, 'red'))

    # read predictions from mask r-cnn
    with open(args.input_file, "r") as f:
        predictions = json.load(f)

    # add image info to json file (no image info in result file)
    with open(args.gt_file, 'r') as f:
        gt_content = json.load(f)
    gt_object = COCO(args.gt_file)
    pred_object = gt_object.loadRes(args.input_file)
    predictions = {'annotations': get_confident_predictions(pred_object, args.threshold),
                   'images': gt_content['images'],
                   'categories': gt_content['categories']}
    print('{} object found for training'.format(len(predictions['annotations'])))

    # save
    with open(args.output_file, "w") as outfile:
        json.dump(predictions, outfile)
    print('File saved')

    # check bounding box AP of output (for validation) with coco api
    pred_object = COCO(args.output_file)
    gt_object = COCO(args.gt_file)
    results_cls = COCOeval(gt_object, pred_object, 'bbox')
    results_cls.params.catIds = list(ID2VOC.keys())
    results_cls.evaluate()
    results_cls.accumulate()
    results_cls.summarize()
    print(colored('initial AP50 result is {}'.format(results_cls.stats[1]), 'yellow'))

    # make mapping
    # TODO: handle this through the coco api
    id2img = {}
    for img in gt_content['images']:
        id2img[img['id']] = img['file_name'].split('.')[0]

    # generate segmentation maps with most confident object masks
    for k, img_anns in tqdm(pred_object.imgToAnns.items()):
        # load ann
        mask_res = False
        for j, ann in enumerate(img_anns):
            image_id = id2img[k]
            category = ann['category_id']
            bbox = ann['bbox']
            mask = pred_object.annToMask(ann)
            h, w = mask.shape

            # check no overlap with existing mask
            if mask_res is not False and (mask_res[mask]).any():
                continue
            if j == 0: mask_res = np.zeros((h, w)).astype(np.uint8)

            # take background into account (class id is 0)
            # we explore different types of masks
            if args.mask_type == 'polygon' or args.mask_type == 'binary_mask':
                mask_res[mask == 1] = 1+category

            elif args.mask_type == 'bbx':
                bbx_mask = np.zeros((h, w))
                bbx_mask[round(bbox[1]):round(bbox[1]+bbox[3])+1, 
                         round(bbox[0]):round(bbox[0]+bbox[2])+1] = 1
                mask_res[bbx_mask == 1] = 1+category

            elif args.mask_type == 'grabcut':
                img_gc = io.imread(os.path.join(args.image_dir, image_id+'.jpg'))
                rect = [int(x+0.5) for x in bbox]
                bgdModel = np.zeros((1, 65), np.float64)
                fgdModel = np.zeros((1, 65), np.float64)
                try:
                    grabcut_mask = np.zeros(img_gc.shape[:2], np.uint8)
                    mask_out,_, _ = cv2.grabCut(img_gc, grabcut_mask, rect, 
                            bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
                except cv2.error as e:
                    mask_out = np.zeros((h, w)).astype(np.uint8)
                    print(colored('CV2 error for image {}'.format(k), 'yellow'))
                mask = np.where((mask_out==2) | (mask_out==0), 0, 1).astype('uint8')
                mask_res[mask == 1] = 1+category

        assert ( mask_res.shape == (h, w) )

        # convert mask to png and save
        im = Image.fromarray(mask_res)
        im.save(os.path.join(args.save_dir, image_id+'.png'))
    print("All images saved")

    print("Offline evaluation on segmentation val set ...")
    from segmentation.utils.evaluate_utils import eval_segmentation_supervised_offline_distill
    from data.voc.pascal_voc import VOC12
    true_val_dataset = VOC12(split='val', transform=None)
    true_train_dataset = VOC12(split='trainaug', transform=None)

    # 21 classes in pascal voc
    eval_stats = eval_segmentation_supervised_offline_distill(21, args.save_dir, 
            true_val_dataset, verbose=True)
    eval_stats = eval_segmentation_supervised_offline_distill(21, args.save_dir, 
            true_train_dataset, verbose=True)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total time {}'.format(total_time_str))
