#
# Authors: Wouter Van Gansbeke
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import sys
import json
import argparse
from termcolor import colored
from tqdm import tqdm
import skimage.io as io

import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils.voc_classes import *
from utils.util import *
import models.models_vit


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


@torch.no_grad()
def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    """ Hungarian matching of clustered predictions with targets
        output: matched pairs with cluster id to target id
    """
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    assert (preds_k == targets_k)  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res


def _majority_match(flat_preds, flat_targets):
    """ Majority matching of clustered predictions with targets
        output: matched pairs with cluster id to target id
    """
    from collections import Counter
    res = []
    for c1 in np.unique(flat_preds):
        gt_i = flat_targets[flat_preds == c1]
        c2, _ = Counter(gt_i).most_common()[0]
        res.append((c1, c2))
    return res


def run_kmeans(x, nmb_clusters, max_points, val_features=None, use_faiss=True, 
               verbose=False, seed=None):
    """ Run kmeans algorithm with faiss or sklearn 
    """
    if use_faiss:
        import faiss
        n_data, d = x.shape
        clus = faiss.Clustering(d, nmb_clusters)
        if seed is None: clus.seed = np.random.randint(1234)
        else: clus.seed = seed
        clus.niter = 20
        clus.max_points_per_centroid = max_points
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = False
        flat_config.device = 0
        index = faiss.GpuIndexFlatIP(res, d, flat_config)

        # perform the training
        clus.train(x, index)
        _, I = index.search(val_features, 1)
        centroids = faiss.vector_float_to_array(clus.centroids).reshape(nmb_clusters, d)
        preds = I.ravel()

    else:
        if seed is None: seed = np.random.randint(1234)
        kmeans = MiniBatchKMeans(n_clusters = nmb_clusters, random_state = seed,
                             batch_size = 1000)
        kmeans.fit(x)
        preds = kmeans.predict(val_features)
        centroids = None

    return preds, centroids


def kmeans(features, features_val, labels_val, nmb_clusters, use_faiss=True, 
            num_trials=1, whiten_feats=True, return_all_metrics=False, seed=None):
    """ Kmeans clustering to nmb_clusters of features
        output: centroid ids after hungarian matching with labels
    """
    from sklearn import metrics
    import scipy.cluster.vq as vq

    # make sure all features are in numpy format
    features = features.cpu().numpy()
    features_val = features_val.cpu().numpy()
    labels_val = labels_val.cpu().numpy()

    # l2-normalize whitened features
    if whiten_feats:
        features = vq.whiten(features)
        features_val = vq.whiten(features_val)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    features_val = features_val / np.linalg.norm(features_val, axis=1, keepdims=True)

    # keep track of clustering metrics: acc, nmi, ari
    acc_lst, nmi_lst, ari_lst = [], [], []
    for i in range(num_trials):
        if seed is None: curr_seed = i
        else: curr_seed = seed

        # run kmeans
        num_elems = labels_val.shape[0]
        I, centroids = run_kmeans(features, nmb_clusters=nmb_clusters, 
                max_points=int(features.shape[0]/nmb_clusters), 
                val_features=features_val, 
                use_faiss=use_faiss, verbose=False, seed=curr_seed)
        pred_labels = np.array(I)
        if nmb_clusters == C:
            # number of clusters equals number of classes C in dataset
            match = _hungarian_match(pred_labels, 
                                     labels_val, 
                                     nmb_clusters, 
                                     nmb_clusters)

        else:
            # number of clusters excedes number of classes C in dataset
            assert nmb_clusters > C
            match = _majority_match(pred_labels, labels_val)

        reordered_preds = np.zeros(num_elems)
        for pred_i, target_i in match:
            reordered_preds[pred_labels == int(pred_i)] = int(target_i)

        # gather performance metrics
        acc = int((reordered_preds == labels_val).sum()) / float(num_elems)
        if return_all_metrics:
            nmi_lst.append(metrics.normalized_mutual_info_score(labels_val, pred_labels))
            ari_lst.append(metrics.adjusted_rand_score(labels_val, pred_labels))
        print(colored(
            'Computed KMeans RUN {} with CLS ACC {:.2f}'.format(i, 100*acc), 'yellow'))
        acc_lst.append(acc)

    # return performance metrics
    if return_all_metrics:
        return {'ACC': 100*np.mean(acc_lst), 'NMI': 100*np.mean(nmi_lst), 
                'ARI': 100*np.mean(ari_lst)}, reordered_preds
    else:
        return {'ACC': 100*np.mean(acc_lst)}, reordered_preds


@torch.no_grad()
def process(preds, prefix, patch_size=8, gt_file=False):
    """ Forward images to obtain features and targets.
    """
    id2img = {}
    for img in preds['images']:
        id2img[img['id']] = img['file_name'].split('.')[0]
    annotations = preds['annotations']
    features_all = torch.FloatTensor(len(annotations), model.norm.normalized_shape[0])
    targets_all = torch.LongTensor(len(annotations))

    gt_coco = False
    if 'COCO' in prefix.upper():
        # we are working with coco
        gt_coco = COCO(gt_file)

    ptr = 0
    for ann in tqdm(annotations):
        bbox = ann["bbox"]
        img_id = id2img[ann['image_id']]

        if gt_coco:
            image = io.imread(os.path.join(prefix, img_id+".jpg"))
            index = int(img_id.split('_')[-1])
            img_anns = gt_coco.imgToAnns[index]
            gt_bbox, gt_class, gt_hards = get_bbx_info_coco(img_anns)

        else:
            image = io.imread(os.path.join(prefix, 'JPEGImages', img_id+".jpg"))
            gt_bbox, gt_class, gt_hards = get_bbx_info_voc(os.path.join(
                prefix, 'Annotations', img_id+".xml"))

        image = transform_val(image)
        image_cropped = image[:, round(bbox[1]):round(bbox[1]+bbox[3]), 
                                 round(bbox[0]):round(bbox[0]+bbox[2])]
        size_im = (
            1,
            image_cropped.shape[0],
            int(np.ceil(image_cropped.shape[1] / patch_size) * patch_size),
            int(np.ceil(image_cropped.shape[2] / patch_size) * patch_size),
        )
        padded = torch.zeros(size_im)
        padded[:, :, :image_cropped.shape[1], :image_cropped.shape[2]] = image_cropped
        image_cropped = padded

        ious = iou_box([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]], gt_bbox)
        gt_id = gt_class[torch.argmax(ious).item()]
        out = model(image_cropped.cuda())
        features_all[ptr:ptr+1] = out.cpu()
        targets_all[ptr:ptr+1] = torch.tensor(gt_id)
        ptr += 1
    return features_all, targets_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Cluster predictions')
    parser.add_argument('--dataset', default='VOCClass', type=str)
    parser.add_argument("--num_classes", type=int, default=20, help='dataset classes')
    parser.add_argument("--overcluster", type=int, default=1)
    parser.add_argument('--model', default='small', type=str)
    parser.add_argument("--patch_size", type=int, default=8)

    parser.add_argument('--dataset_root1', default='/path/to/VOC12', type=str)
    parser.add_argument('--dataset_root2', default='/path/to/VOC07/', type=str)
    parser.add_argument('--input_file1', default=None, help='coco style input')
    parser.add_argument('--input_file2', default=None, help='coco style input')
    parser.add_argument('--gt_file1', default=None)
    parser.add_argument('--gt_file2', default=None)
    parser.add_argument('--output_file', default=None)
    parser.add_argument('--save_dir', default='clusters/')
    args = parser.parse_args()

    # make output dir to save features
    os.makedirs(args.save_dir, exist_ok=True)

    # get amount of classes C in dataset (for clustering later on)
    if 'VOC' in args.dataset.upper():
        C = 20
    elif 'COCO' in args.dataset.upper():
        C = 80
    else:
        raise NotImpelementedError

    assert(args.num_classes == C)

    # get pretrained vit model (patch size 8 works slightly better than 16 for clustering)
    if args.model == 'small':
        args.model = 'vit_small_patch' + str(args.patch_size)
        args.pretrained_weights = \
            '../pretrained/dino_vitsmall{}_pretrain.pth'.format(args.patch_size)
    else:
        raise NotImpelementedError

    # print args
    print(colored(args, 'red'))

    # create model
    model = models.models_vit.__dict__[args.model]()

    # load pretrained weights
    if args.pretrained_weights:
        checkpoint = torch.load(args.pretrained_weights, map_location='cpu')

        print('checkpoint loaded from', args.pretrained_weights)
        if 'model' in checkpoint.keys(): # handle moco or dino weights
            checkpoint_model = checkpoint['model']
        else:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print('Removing key {} from pretrained checkpoint'.format(k))
                del checkpoint_model[k]
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

    # set model in eval mode
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    model.to('cuda')
    model.head = nn.Identity()
    transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # read files
    with open(args.input_file1, "r") as f: # eg. voc2012
        predictions1 = json.load(f)
    if args.input_file2 is not None:
        with open(args.input_file2, "r") as f: # eg. voc2007 or a different support set
            predictions2 = json.load(f)

    if not 'annotations' in predictions1: # a json file without image info is used.
        with open(args.gt_file1, 'r') as f:
            gt1 = json.load(f)
        predictions1 = {'annotations': get_most_confident_predictions(predictions1),
                        'images': gt1['images'],
                        'categories': gt1['categories']}
    if args.input_file2 is not None and not 'annotations' in predictions2:
        with open(args.gt_file2, 'r') as f:
            gt2 = json.load(f)
        predictions2 = {'annotations': get_most_confident_predictions(predictions2),
                        'images': gt2['images'],
                        'categories': gt2['categories']}

    if not os.path.exists(os.path.join(args.save_dir, 'features1.pth')):
        features1, targets1 = process(predictions1, prefix=args.dataset_root1, 
                gt_file=args.gt_file1)
        torch.save(features1, os.path.join(args.save_dir, 'features1.pth'))
        torch.save(targets1, os.path.join(args.save_dir, 'labels1.pth'))
        if args.input_file2 is not None:
            features2, targets2 = process(predictions2, prefix=args.dataset_root2,
                gt_file=args.gt_file2)
            torch.save(features2, os.path.join(args.save_dir, 'features2.pth'))
            torch.save(targets2, os.path.join(args.save_dir, 'labels2.pth'))
            features = torch.cat((features1, features2), dim=0) # order is important
            targets = torch.cat((targets1, targets2), dim=0)
    else:
        print(colored('Loading features ... ', 'yellow'))
        features1 = torch.load(os.path.join(args.save_dir, 'features1.pth'))
        targets1 = torch.load(os.path.join(args.save_dir, 'labels1.pth'))
        if args.input_file2 is not None:
            features2 = torch.load(os.path.join(args.save_dir, 'features2.pth'))
            targets2 = torch.load(os.path.join(args.save_dir, 'labels2.pth'))
            features = torch.cat((features1, features2), dim=0)
            targets = torch.cat((targets1, targets2), dim=0)

    if not args.input_file2:
        features = features1
        targets = targets1

    # remap targets from [start, end] to [0, end-start] range
    targets_to_id = {v:i for i,v in enumerate(np.unique(targets))}
    id_to_targets = {v:i for i,v in targets_to_id.items()}
    targets = torch.tensor([targets_to_id[x] for x in targets.numpy()])

    # cluster all features
    res, reordered_preds = kmeans(features, features, targets,
        nmb_clusters=args.num_classes*args.overcluster, use_faiss=False, num_trials=1, 
        whiten_feats=True, return_all_metrics=True, seed=1234)
    print(colored('Result of KMeans evaluation is {}'.format(res), 'yellow'))

    # undo mapping: map back to [start, end] range
    reordered_preds = [id_to_targets[int(x)] for x in reordered_preds]

    # set pointer and save OD json files
    all_categories = [{'supercategory': '', 'id': i, 'name': x} for i,x in ID2VOC.items()]
    ptr = 0
    anns = []
    for ann in predictions1['annotations']:
        ann['category_id'] = int(reordered_preds[ptr])
        if 'score' not in ann: ann['score']  = 1.0
        anns.append(ann)
        ptr += 1
    preds_out = {'images': predictions1['images'],
                 'annotations': anns,
                 'categories': all_categories}

    with open(args.output_file, "w") as outfile:
        json.dump(preds_out, outfile)
    print('File saved')

    # compute initial AP
    if args.gt_file1 is not None:
        pred_object = COCO(args.output_file)
        gt_object = COCO(args.gt_file1)
        results_cls = COCOeval(gt_object, pred_object, 'bbox')
        results_cls.params.catIds = list(ID2VOC.keys())
        results_cls.evaluate()
        results_cls.accumulate()
        results_cls.summarize()
        print(colored('initial AP50 results is {}'.format(results_cls.stats[1]), 'yellow'))
