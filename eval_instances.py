#
# Authors: Wouter Van Gansbeke
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import time
import json
import torch
import argparse
import datetime
from tqdm import tqdm
from termcolor import colored
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils.util import iou_box, get_bbx_info_coco


def get_args_parser():
    parser = argparse.ArgumentParser('Eval instance segmentation')
    parser.add_argument('--input_file', default='/path')
    parser.add_argument('--gt_file', default='/path')
    parser.add_argument('--eval_most_confident', action='store_true')
    return parser

def get_most_confident_predictions(predictions_voc):
    """ Get most confident predictions.
    """
    memo = {}
    for i, p in enumerate(predictions_voc):
        img_id = p['image_id']
        p['id'] = i # remap
        p['area'] = int(p['bbox'][2]*p['bbox'][3]) # A = h*w
        if img_id not in memo or memo[img_id]['score'] < p['score']:
            memo[img_id] = p
    return list(memo.values())


if __name__ == "__main__":
    start_time = time.time()

    # get arguments
    args = get_args_parser()
    args = args.parse_args()
    print(colored(args, 'red'))

    if args.eval_most_confident:
        os.makedirs('tmp', exist_ok=True)

        # read in files
        with open(args.input_file, "r") as f:
            predictions = json.load(f)
        with open(args.gt_file, "r") as f:
            gt = json.load(f)

        # get most confident predictions
        predictions = {'annotations': get_most_confident_predictions(predictions),
                       'images': gt['images'],
                       'categories': gt['categories']}
        with open('tmp/sel.json', "w") as f:
            json.dump(predictions, f)
        print('File saved')

        # map to ground truth object based on max bbox iou overlap
        gt_object = COCO(args.gt_file)
        id2img = {}
        for img in predictions['images']:
            id2img[img['id']] = img['file_name'].split('.')[0]
        anns_sel = defaultdict(list)
        for ann in tqdm(predictions['annotations']):
            bbox = ann["bbox"]
            img_id = ann['image_id']
            img_anns = gt_object.imgToAnns[img_id]
            bbx_objs, class_objs, hard_objs = get_bbx_info_coco(img_anns)
            assert(len(bbx_objs) != 0)
            ious = iou_box([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]], bbx_objs)
            anns_sel[img_id].append(img_anns[torch.argmax(ious).item()])

        res = []
        for an in anns_sel.values():
            for x in an:
                res.append(x)
        anns_sel = {'annotations': res,
                    'images': gt['images'],
                    'categories': gt['categories']}
        print('Found {} objects'.format(len(anns_sel['annotations'])))
        with open('tmp/gt_sel.json', "w") as outfile:
            json.dump(anns_sel, outfile)
        print('File saved')

        # evaluate on selection
        gt_sel = COCO('tmp/gt_sel.json')
        pred_sel = COCO('tmp/sel.json')
        results_cls = COCOeval(gt_sel, pred_sel, 'segm')
        results_cls.evaluate()
        results_cls.accumulate()
        results_cls.summarize()
        print('Eval classes ... ')
        print(gt_sel.cats)
        print(colored('segm AP50 result is {}'.format(results_cls.stats[1]), 'yellow'))

    else:
        # compute segm results in COCO-style format
        gt_object = COCO(args.gt_file)
        pred_object = gt_object.loadRes(args.input_file)
        results_cls = COCOeval(gt_object, pred_object, 'segm')
        results_cls.evaluate()
        results_cls.accumulate()
        results_cls.summarize()
        print('Eval classes ... ')
        print(gt_object.cats)
        print(colored('segm AP50 result is {}'.format(results_cls.stats[1]), 'yellow'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total time {}'.format(total_time_str))
