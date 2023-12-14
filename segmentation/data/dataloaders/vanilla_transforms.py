import numpy.random as random
import numpy as np
import torch
import cv2
import math
import torchvision


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        w, h = sample['image'].shape[:2] 
        W, H = self.size

        if (W, H) == (w, h): 
            return sample
        
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)

            for elem in sample.keys():
                if elem == 'meta':
                    continue

                elif elem == 'semseg':
                    sample[elem] = cv2.resize(sample[elem], (h, w), interpolation=cv2.INTER_NEAREST)
                
                elif elem == 'depth':
                    sample[elem] = scale * cv2.resize(sample[elem], (h, w), interpolation=cv2.INTER_NEAREST)
                    
                else:
                    sample[elem] = cv2.resize(sample[elem], (h, w), interpolation=cv2.INTER_CUBIC)
        
        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        for elem in sample.keys():
            if elem == 'meta':
                continue
            tmp = sample[elem]
            if tmp.ndim == 2:
                sample[elem] = tmp[crop[0]:crop[2], crop[1]:crop[3]]
            else:
                sample[elem] = tmp[crop[0]:crop[2], crop[1]:crop[3], :]

        return sample

        
    def __str__(self):
        return 'RandomCrop'


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        w, h = sample['image'].shape[:2] 
        W, H = self.size

        if (W, H) == (w, h): 
            return sample
        
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)

            for elem in sample.keys():
                if elem == 'meta':
                    continue

                elif elem == 'semseg':
                    sample[elem] = cv2.resize(sample[elem], (h, w), interpolation=cv2.INTER_NEAREST)
                
                elif elem == 'depth':
                    sample[elem] = scale * cv2.resize(sample[elem], (h, w), interpolation=cv2.INTER_NEAREST)
                    
                else:
                    sample[elem] = cv2.resize(sample[elem], (h, w), interpolation=cv2.INTER_CUBIC)
        
        mid_w, mid_h = w // 2, h // 2
        sw, sh = mid_w - W // 2, mid_h - H // 2
        
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        for elem in sample.keys():
            if elem == 'meta':
                continue

            tmp = sample[elem]
            if tmp.ndim == 2:
                sample[elem] = tmp[crop[0]:crop[2], crop[1]:crop[3]]

            else:
                sample[elem] = tmp[crop[0]:crop[2], crop[1]:crop[3], :]

        return sample

    def __str__(self):
        return 'CenterCrop'


class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):

        if random.random() < 0.5:
            for elem in sample.keys():
                if 'meta' in elem:
                    continue
                else:
                    tmp = sample[elem]
                    tmp = cv2.flip(tmp, flipCode=1)
                    sample[elem] = tmp

        return sample

    def __str__(self):
        return 'RandomHorizontalFlip'


class AddIgnoreRegions(object):
    """Add Ignore Regions"""

    def __call__(self, sample):
        """
        for elem in sample.keys():
            tmp = sample[elem]
            if elem == 'depth':
                tmp[tmp == 0] = 255.
                sample[elem] = tmp
        """
        return sample

    def __str__(self):
        return 'AddIgnoreRegions'


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        self.to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, sample):

        for elem in sample.keys():
            if elem == 'meta':
                continue
            
            tmp = sample[elem]

            if elem == 'image':
                sample[elem] = self.to_tensor(tmp.astype(np.uint8)) # Between 0 .. 255 so cast as uint8 to ensure compatible w/ imagenet weight
            
            elif elem == 'semseg':
                sample[elem] = torch.from_numpy(tmp).long()

            else:
                sample[elem] = torch.from_numpy(tmp).float()

        return sample

    def __str__(self):
        return 'ToTensor'


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.normalize = torchvision.transforms.Normalize(self.mean, self.std)

    def __call__(self, sample):
        sample['image'] = self.normalize(sample['image'])    
        return sample

    def __str__(self):
        return 'Normalize([%.3f,%.3f,%.3f],[%.3f,%.3f,%.3f])' %(self.mean[0], self.mean[1], self.mean[2], self.std[0], self.std[1], self.std[2])


class Resize(object):
    def __init__(self, size=(512, 1024)):
        self.h = size[0]
        self.w = size[1]

    def __call__(self, sample):
        for elem in sample:
            if elem == 'meta':
                continue

            elif elem == 'semseg':
                sample[elem] = cv2.resize(sample[elem], (self.h, self.w), interpolation=cv2.INTER_NEAREST)
            
            elif elem == 'depth':
                sample[elem] = cv2.resize(sample[elem], (self.h, self.w), interpolation=cv2.INTER_NEAREST)
                
            else:
                sample[elem] = cv2.resize(sample[elem], (self.h, self.w), interpolation=cv2.INTER_CUBIC)

        return sample

    def __str__(self):
        return 'Resize(scales=' + str(self.size) + ')'


class RandomScale(object):
    def __init__(self, scales):
        self.scales = scales

    def __call__(self, sample):
        W, H = sample['image'].shape[:2]

        if type(self.scales) == tuple:
            scale  = (self.scales[1] - self.scales[0]) * random.random() + self.scales[0] 

        elif type(self.scales) == list:
            scale = random.choice(self.scales)

        w, h = int(W * scale), int(H * scale)

        for elem in sample:
            if elem == 'meta':
                continue

            elif elem == 'semseg':
                sample[elem] = cv2.resize(sample[elem], (h, w), interpolation=cv2.INTER_NEAREST)
            
            elif elem == 'depth':
                sample[elem] = scale * cv2.resize(sample[elem], (h, w), interpolation=cv2.INTER_NEAREST)
                
            else:
                sample[elem] = cv2.resize(sample[elem], (h, w), interpolation=cv2.INTER_CUBIC)

        return sample

    def __str__(self):
        return 'RandomScale(scales=' + str(self.scales) + ')'
