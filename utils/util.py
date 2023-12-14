#
# Authors: Wouter Van Gansbeke
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import json
import torch
import numpy as np
from skimage import measure
from skimage.measure import find_contours
from pycocotools import mask
from PIL import Image
import xml.etree.ElementTree as ET
from utils.voc_classes import *

EPS=1e-8

def iou_box(b1, b2):
    """ Compute IoU between 2 bounding boxes.
    """

    # handle input type
    if isinstance(b1, (list)):
        b1 = torch.tensor(b1, dtype=torch.float32)
    if isinstance(b2, (list)):
        b2 = torch.tensor(b2, dtype=torch.float32)
    if isinstance(b1, (np.ndarray)):
        b1 = torch.from_numpy(b1.astype(np.float32))
    if isinstance(b2, (np.ndarray)):
        b2 = torch.from_numpy(b2.astype(np.float32))

    # compute intersection and union
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
    b2 = b2.T
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[0], b2[1], b2[2], b2[3]
    intersection = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                   (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + EPS
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + EPS
    union = w1 * h1 + w2 * h2 - intersection + EPS

    # compute iou 
    return intersection/union


def get_annotation_entry(annotation_id, image_id, category_info, binary_mask, 
                           image_size=None, tolerance=2, bounding_box=None):
    """ Construct annotation in COCO-style format.
    """

    # based on waspinator/pycococreator impl.
    if image_size is not None:
        binary_mask = resize_binary_mask(binary_mask, image_size)

    binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    area = mask.area(binary_mask_encoded)
    if area < 1:
        return None

    if bounding_box is None:
        bounding_box = mask.toBbox(binary_mask_encoded)

    assert not category_info["iscrowd"]
    is_crowd = 0
    segmentation = binary_mask_to_polygon(binary_mask, tolerance)
    if not segmentation:
        return None

    annotation_info = {
        "score": 1.0,
        "id": annotation_id,
        "image_id": image_id,
        "category_id": int(category_info["id"]),
        "iscrowd": is_crowd,
        "area": area.tolist(),
        "bbox": bounding_box.tolist(),
        "segmentation": segmentation,
        "width": binary_mask.shape[1],
        "height": binary_mask.shape[0],
    } 
    return annotation_info


def get_image_entry(img_name, image_id, dataset, image_size):
    """ Construct image info in COCO-style format.
    """

    if 'VOC'in dataset.__class__.__name__.upper():
        file_name = img_name

    elif 'COCO'in dataset.__class__.__name__.upper():
        file_name = dataset.coco.imgs[image_id]['file_name']

    else:
        raise NotImplementedError

    return {'file_name': file_name, 
            'height': image_size[0],
            'width': image_size[1],
            'id': image_id}


def binary_mask_to_polygon(binary_mask, tolerance=0):
    """ Convert binary mask to polygon.
    """

    polygons = []
    padded_binary_mask = np.pad(binary_mask, pad_width=1, 
            mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def close_contour(contour):
    # Helper function: construct polygon from mask.
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def resize_binary_mask(array, new_size):
    # Helper function: resize mask to original image size.
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)


def get_coco20k(sel_file, all_annotations_file, annotations_file, annotations_file_agn):
    """ Construct annotation file based on the selected COCO20k-filenames.
    """

    from pycocotools.coco import COCO
    print('Get COCO20k dataset')

    # load all annotations
    all_anns_obj = COCO(all_annotations_file)

    # load selected images
    with open(sel_file, "r") as f:
        sel_20k = f.readlines()
        sel_20k = [s.replace("\n", "") for s in sel_20k]
    im20k = [int(s.split("_")[-1].split(".")[0]) for s in sel_20k]
    anns_sel = []
    imgs_sel = []
    for img_id in im20k:
        anns_sel.extend(all_anns_obj.imgToAnns[img_id])
        imgs_sel.append(all_anns_obj.imgs[img_id])
    train_coco = {}
    train_coco["images"] = imgs_sel
    train_coco["annotations"] = anns_sel
    train_coco["categories"] = list(all_anns_obj.cats.values())
    with open(annotations_file, "w") as outfile:
        json.dump(train_coco, outfile)

    # handle agnostic case
    for ann in train_coco["annotations"]:
        ann['category_id'] = 0
    train_coco["categories"] = [{'supercategory': '', 'id': 0, 'name': ''}] # agnostic
    with open(annotations_file_agn, "w") as outfile_agn:
        json.dump(train_coco, outfile_agn)


def get_img_info(ann, dataset):
    """ Construct image info.
    """

    if "VOC" in dataset.__class__.__name__.upper():
        img_name = ann["annotation"]["filename"]
        height = int(ann['annotation']['size']['height'])
        width = int(ann['annotation']['size']['width'])

    elif "COCO" in dataset.__class__.__name__.upper():
        img_id = ann[0]['image_id']
        img_name = str(img_id)
        height = dataset.coco.imgs[img_id]['height']
        width = dataset.coco.imgs[img_id]['width']

    else:
        raise NotImplementedError

    return img_name, height, width


def get_bbx_info_voc(ann_file):
    """ Get bounding box info from xml file for VOC
    """
    # based on gt extraction for VOC in LOST from xml file
    ann_file = open(ann_file)
    tree=ET.parse(ann_file)
    root = tree.getroot()
    bbox_all = []
    class_all = []
    hards_all = []
    for obj in root.iter('object'):
        hards_all.append(int(obj.find('difficult').text))
        cls = obj.find('name').text
        if cls not in VOC2ID: assert False
        class_all.append(VOC2ID[cls])
        bbox = obj.find("bndbox")
        bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
        bbox[0] -= 1.0
        bbox[1] -= 1.0
        bbox_all.append(bbox)
    return bbox_all, class_all, hards_all


def get_bbx_info_coco(ann, rm_iscrowd=True):
    """ Extract ground truth boxes and its class
    """
    hard_objs = []
    class_objs = []
    bbx_objs = []
    for obj_i in ann:
        hard_i = 0
        if rm_iscrowd and obj_i["iscrowd"] == 1: hard_i = 1
        hard_objs.append(hard_i)
        class_objs.append(obj_i["category_id"])
        bbox = obj_i["bbox"]
        c = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] # COCO to VOC format
        bbx_objs.append([int(round(x)) for x in c])
    return bbx_objs, class_objs, hard_objs


def threshold_attention(attentions, threshold):
    """ Keep only a certain percentage of the attention mass.
        see arxiv.org/abs/2104.14294
    """

    nh = attentions.shape[0]
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]
    return th_attn
