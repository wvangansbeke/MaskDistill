#
# Authors: Wouter Van Gansbeke
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from torchvision.models.resnet import resnet50, resnet18
from utils.collate import collate_custom


def get_backbone(p):
    if p['model_kwargs']['pretraining'] == 'imagenet_supervised':
        print('Loaded model with ImageNet supervised initialization.')
        return resnet50(pretrained=True)

    elif p['model_kwargs']['pretraining'] == 'random':
        print('Loaded model with random initialization.')
        return resnet50(pretrained=False)

    elif p['model_kwargs']['pretraining'] == 'moco':
        print('Loading model with MoCo initialization')
        print('State dict found at {}'.format(p['model_kwargs']['pretraining_path']))
        model = resnet50(pretrained=False)
        checkpoint = torch.load(p['model_kwargs']['pretraining_path'], map_location='cpu')
        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]

            elif k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.fc'):
                # remove prefix
                state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        return model

    else:
        raise NotImplementedError('Model with pretraining {} not implemented.'.format(p['model_kwargs']['pretraining']))


def get_model(p):
    if 'VOCSegmentation' in p['train_db_name']:
        if 'use_fcn' in p['model_kwargs'] and p['model_kwargs']['use_fcn']:
            # FCN model with dilation 6 in the head following MoCo suppl. mat.
            print('Using FCN for PASCAL')
            from models.fcn_model import Model
            return Model(get_backbone(p), p['num_classes'] + int(p['has_bg']))

        else:
            # We use a deeplabv3 model
            from models.deeplabv3_model import DeepLabV3
            print('Using DeepLab v3 for PASCAL')
            return DeepLabV3(get_backbone(p), p['num_classes'] + int(p['has_bg']))

    else:
        raise ValueError('No model for train dataset {}'.format(p['train_db_name']))


def get_train_dataset(p, transform=None):
    if p['train_db_name'] == 'VOCSegmentation':
        from data.dataloaders.pascal_voc import VOC12
        dataset = VOC12(split=p['train_db_kwargs']['split'], transform=transform)

    elif p['train_db_name'] == 'VOCSegmentationDistill':
        from data.dataloaders.pascal_voc_distill import VOC12Distill
        dataset = VOC12Distill(split=p['train_db_kwargs']['split'], 
                distill_dir=p['train_db_kwargs']['distill_dir'],
                transform=transform)

    else:
        raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))
    
    return dataset


def get_val_dataset(p, transform=None):
    if p['val_db_name'] == 'VOCSegmentation':
        from data.dataloaders.pascal_voc import VOC12
        dataset = VOC12(split='val', transform=transform)
    
    else:
        raise ValueError('Invalid validation dataset {}'.format(p['val_db_name']))
    
    return dataset


def get_train_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'], 
            batch_size=p['train_db_kwargs']['batch_size'], pin_memory=True, 
            collate_fn=collate_custom, drop_last=True, shuffle=True)


def get_val_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],
            batch_size=p['val_db_kwargs']['batch_size'], pin_memory=True, 
            collate_fn=collate_custom, drop_last=False, shuffle=False)


def get_train_transformations(p):
    if p['train_db_kwargs']['transforms'] == 'fblib_pascal':
        import data.dataloaders.fblib_transforms as fblib_tr
        return transforms.Compose([fblib_tr.RandomHorizontalFlip(),
                                       fblib_tr.ScaleNRotate(rots=(-5,5), scales=(.75,1.25),
                                        flagvals={'semseg': cv2.INTER_NEAREST, 'image': cv2.INTER_CUBIC}),
                                       fblib_tr.FixedResize(resolutions={'image': tuple((512,512)), 'semseg': tuple((512,512))},
                                        flagvals={'semseg': cv2.INTER_NEAREST, 'image': cv2.INTER_CUBIC}),
                                       fblib_tr.ToTensor(),
                                        fblib_tr.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    
    elif p['train_db_kwargs']['transforms'] == 'vanilla_pascal':
        import data.dataloaders.vanilla_transforms as tr
        return transforms.Compose([tr.RandomScale((0.5, 2.0)), 
                                    tr.RandomCrop((513, 513)), tr.RandomHorizontalFlip(),
                                    tr.ToTensor(), tr.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    else:
        raise NotImplementedError

    
def get_val_transformations(p):
    if p['val_db_kwargs']['transforms'] == 'fblib_pascal':
        import data.dataloaders.fblib_transforms as fblib_tr
        return transforms.Compose([fblib_tr.FixedResize(resolutions={'image': tuple((512,512)), 
                                                            'semseg': tuple((512,512))},
                                                flagvals={'image': cv2.INTER_CUBIC, 'semseg': cv2.INTER_NEAREST}),
                                    fblib_tr.ToTensor(),
                                    fblib_tr.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    
    elif p['val_db_kwargs']['transforms'] == 'vanilla_pascal':
        import data.dataloaders.vanilla_transforms as tr
        return transforms.Compose([tr.Resize((513, 513)),
                                    tr.ToTensor(), tr.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    
    else:
        raise NotImplementedError


def get_optimizer(p, parameters):
    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(parameters, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(parameters, **p['optimizer_kwargs'])
    
    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer


def adjust_learning_rate(p, optimizer, epoch):
    lr = p['optimizer_kwargs']['lr']
    
    if p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'poly':
        lambd = pow(1-(epoch/p['epochs']), 0.9)
        lr = lr * lambd

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
