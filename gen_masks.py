#
# Authors: Wouter Van Gansbeke
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import json
import argparse
from tqdm import tqdm
from termcolor import colored

import scipy
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets
import models.models_vit

from utils.voc_classes import *
from utils.util import *


def mask_generator(Q, K, cols, args):
    """ Generate masks based on self-attention in transformers.
    """
    # define variables
    bs, L, N, _ = Q.size()

    # compute attention
    A = (Q @ K.transpose(-2, -1))

    # compute graph G_cls
    G_cls = A[0, :, 0, 1:].reshape(L, -1) # select CLS token
    G_cls = torch.mean(G_cls[torch.arange(args.heads)], dim=0, keepdim=True) # average heads
    G_cls *= threshold_attention(G_cls, 0.5).view(bs, -1) # for better visualization

    # compute graph G_i
    K_i = K[0, :, 1:] # leave out CLS token: N-1 tokens
    G_i = (K_i @ K_i.transpose(-2, -1)).sum(dim=0)

    # select topk patches in G_cls (i.e., set of proposed patches P in paper)
    G_cls = G_cls[0].view(-1)
    topk_ids = torch.topk(G_cls.float(), k=int(args.topk*(N-1)), 
                          dim=0, largest=True, sorted=True)[1]
    source = topk_ids[0] # most discriminative patch

    # diffuse over similar patches P in graph G_i (see paper LOST)
    source_nodes = topk_ids[G_i[source, topk_ids] > 0.0]
    G_diff = torch.sum(G_i[source_nodes], dim=0)
    G_diff_th = (G_diff > 0).view(-1, cols)
    labels = scipy.ndimage.label(G_diff_th.cpu().numpy())[0]

    # select most discriminative component in G_diff
    obj_mask = labels == labels[source // cols, source % cols]
    return obj_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate masks')
    parser.add_argument("--dataset", default="VOCClass", type=str,
                        help="specify your dataset")
    parser.add_argument("--dataset_root", default="/esat/rat/wvangans/Datasets/", type=str,
                        help="path to your dataset")
    parser.add_argument("--year", default="2012", type=str,
                        help="year of dataset")
    parser.add_argument("--set", type=str, default="trainval",
                        help="dataset split for dataloader")
    parser.add_argument("--include_test", action="store_true",
                        help="include test set")
    parser.add_argument("--model", default="small", type=str, 
                        help="transformer model")
    parser.add_argument("--patch_size", type=int, default=16,
                        help="patch size of transformer")
    parser.add_argument("--heads", type=int, default=5,
                        help="amount of heads to include")
    parser.add_argument("--pred_json_path", default=None,
                        help="output file")
    parser.add_argument("--gt_json_path_coco", default=None,
                        help="coco ground truth")
    parser.add_argument("--gt_json_path_coco_agn", default=None,
                        help="class agnostic coco ground truth")
    parser.add_argument("--resize", type=int, default=640,
                       help="resize input to a specific size")
    parser.add_argument("--topk", type=float, default=0.4,
                        help="topk selection")
    args = parser.parse_args()

    # get pretrained vit model
    if args.model == "small":
        args.model = "vit_small_patch" + str(args.patch_size)
        args.pretrained_weights = \
            "pretrained/dino_vitsmall{}_pretrain.pth".format(args.patch_size)
    else:
        raise NotImplementedError

    # print args
    print(colored(args, "red"))

    # create model
    model = models.models_vit.__dict__[args.model]()

    # load pretrained weights
    if args.pretrained_weights:
        checkpoint = torch.load(args.pretrained_weights, map_location="cpu")

        print("checkpoint loaded from", args.pretrained_weights)
        if "model" in checkpoint.keys(): # handle moco or dino weights
            checkpoint_model = checkpoint["model"]
        else:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print("Removing key {} from pretrained checkpoint".format(k))
                del checkpoint_model[k]
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        assert set(msg.missing_keys) == {"head.weight", "head.bias"}

    # set model in eval mode
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    model.to("cuda")
    model.head = nn.Identity()

    # get transformations
    transform = transforms.Compose([
            transforms.Resize(args.resize),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # get dataset
    if args.dataset == "VOCClass":
        from data.voc.voc_base import VOCDetection
        dataset = VOCDetection(
            args.dataset_root,
            year=args.year,
            image_set=args.set,
            transform=transform,
            download=False,
        )
        if args.year == "2007" and args.include_test:
            # add 2007 set
            from torch.utils.data import ConcatDataset
            dataset_test = VOCDetection(
                args.dataset_root,
                year="2007",
                image_set="test",
                transform=transform,
                download=False,
            )
            dataset = ConcatDataset([dataset, dataset_test])

        elif args.set == "trainaug_seg" and args.year == "2012" and args.include_test:
            # add 2012 segmentation validation set
            from torch.utils.data import ConcatDataset
            dataset_test = VOCDetection(
                args.dataset_root,
                year="2012",
                image_set="val_seg", # make sure split is defined in voc_base.py
                transform=transform,
                download=False,
            )
            dataset = ConcatDataset([dataset, dataset_test])

    elif args.dataset == "COCO20k":
        # make sure MS-COCO is in the correct location
        # benchmark from arxiv.org/abs/2007.02662
        prefix = os.path.join(args.dataset_root, "MS-COCO2014")
        all_annfile = os.path.join(prefix, "annotations/instances_train2014.json")
        annfile, annfile_agn = args.gt_json_path_coco, args.gt_json_path_coco_agn   
        sel20k = "data/coco/coco20k_files.txt"
        if not os.path.exists(annfile) or not os.path.exists(annfile_agn):
            from utils.util import get_coco20k
            get_coco20k(sel20k, all_annfile, annfile, annfile_agn)
        root_path = os.path.join(prefix, "train2014")
        dataset = torchvision.datasets.CocoDetection(
                root_path, annFile=annfile, transform=transform)

    else:
        raise NotImplementedError

    print(dataset)
    print(colored("Found {} images in dataset".format(len(dataset)), "yellow"))

    # containers
    all_imgs = []
    all_anns = []
    pred_idx = 0

    for j, (img, ann) in enumerate(tqdm(dataset)):

        # pad image to a multiple of the patch-size
        or_h, or_w = img.shape[1], img.shape[2]
        size_im = (
            img.shape[0],
            int(np.ceil(or_h / args.patch_size) * args.patch_size),
            int(np.ceil(or_w / args.patch_size) * args.patch_size))
        img_padded = torch.zeros(size_im)
        img_padded[:, :or_h, :or_w] = img
        img = img_padded
        cols = img.shape[-1] // args.patch_size

        # get q, k features
        _, _, queries, keys = model.get_last_selfattention(img.unsqueeze(0).to("cuda"))

        # generate mask proposals
        pred_mask = mask_generator(queries, keys, cols, args)

        # create predictions
        if args.pred_json_path is not None:

            # nearest neighbor interpolation
            pred_mask = nn.functional.interpolate(
              torch.from_numpy(pred_mask[None, None, :, :]).float(),
              scale_factor=args.patch_size, 
              mode="nearest")[0, 0, :or_h, :or_w].numpy()

            # create annotation entry (coco style)
            img_name, img_h, img_w = get_img_info(ann, dataset)
            ann_info = {'id': 0, 'iscrowd':0} # class agnostic
            idx = ann[0]["image_id"] if "COCO" in args.dataset else j
            ann_i = get_annotation_entry(pred_idx, idx, 
                       ann_info, binary_mask=pred_mask, image_size=(img_w, img_h), 
                       tolerance=2, bounding_box=None) # flip img_size for resize with PIL.
            image_i = get_image_entry(img_name, idx, dataset, image_size=(img_h, img_w))

            all_anns.append(ann_i)
            all_imgs.append(image_i)
            pred_idx += 1

    if args.pred_json_path is not None:
        # save prediction json with pred bbox and pred mask
        print("number of predictions", sum([x["iscrowd"]==0 for x in all_anns]))
        train_data = {}
        train_data["images"] = all_imgs
        train_data["annotations"] = all_anns
        train_data["categories"] = [{'supercategory': '', 'id': 0, 'name': ''}] # agnostic
        with open(args.pred_json_path, "w") as outfile:
                json.dump(train_data, outfile)
        print("File with predictions constructed.")
