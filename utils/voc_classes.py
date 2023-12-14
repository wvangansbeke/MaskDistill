#
# Authors: Wouter Van Gansbeke
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor", "background"]

# make dictionaries for easy access
VOC2ID = {c:i for i,c in enumerate(VOC_CLASSES)}
ID2VOC = {i:c for i,c in enumerate(VOC_CLASSES)}
