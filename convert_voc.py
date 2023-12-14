#
# Authors: Wouter Van Gansbeke
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import json
import argparse
from tqdm import tqdm
from termcolor import colored

import numpy as np
from utils.voc_classes import *
from utils.util import *

# Generate instance segmentation ground truth in COCO-style format.
# The 2913 object segmentation files in VOC 2012 are used.


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate instances on PASCAL VOC')
    parser.add_argument('--out_file', default=None, help='path to output file')
    parser.add_argument('--agnostic', action='store_true', help='set class to 0')
    args = parser.parse_args()

    # print args
    print(colored(args, 'red'))

    # get dataset
    from data.voc.pascal_voc_instance import VOC12
    dataset = VOC12(split='instance', transform=None)
    print(dataset)
    print(colored('Found {} images in dataset'.format(len(dataset)), 'yellow'))

    # containers
    all_anns = []
    all_imgs = []
    gt_idx = 0

    # construct instance annotations in coco-style format
    for j, sample in enumerate(tqdm(dataset)):
        img = sample['image']
        instance = sample['instance'] 
        semseg = sample['semseg'] 
        img_name = sample['meta']['image']
        img_w, img_h = sample['meta']['im_size'][::-1]

        for idx in np.unique(instance):
            if idx in [0, 255]: continue # handle background or ignore id
            mask_i = instance == idx
            if not args.agnostic:
                # assign mask id using a majority vote within each object
                cls_i = np.argmax(np.bincount(semseg[mask_i])) - 1 # to make compatible
                assert cls_i != 255
            else:
                cls_i = 0 # class agnostic
            # generate polygon for annotation file
            ann_info = {'id': cls_i, 'iscrowd': 0}
            ann_i = get_annotation_entry(gt_idx, img_name, 
                            ann_info, binary_mask=mask_i, image_size=(img_w, img_h), 
                            tolerance=2, bounding_box=None)
            if ann_i is None: continue
            all_anns.append(ann_i)
            gt_idx += 1

        image_i = get_image_entry(img_name, img_name, dataset, image_size=(img_h, img_w))
        all_imgs.append(image_i)

    if not args.agnostic:
        all_categories = [{'supercategory': '', 'id': i, 'name': x} for i,x in ID2VOC.items()]
    else:
        all_categories = [{'supercategory': '', 'id': 0, 'name': ''}]

    if args.out_file is not None:
        # save gt in json format
        print('number of annotations', sum([x['iscrowd']==0 for x in all_anns]))
        gt_data = {}
        gt_data["images"] = all_imgs # same as prediction
        gt_data["annotations"] = all_anns
        gt_data["categories"] = all_categories # calss agnostic if args.agnostic is true
        with open(args.out_file, "w") as outfile:
                json.dump(gt_data, outfile)
        print('File with ground truth constructed.')
