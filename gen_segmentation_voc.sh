#!/usr/bin/sh

# define paths
PRED_VOC=detectron2/out/voc/mask_rcnn_trainaug.json
GT_VOC=detectron2/out/voc/gt_trainaug.json
CONF_PRED_VOC=detectron2/out/voc/conf_pred_trainaug.json # output file
OUTDIR=OUT # Segmentation maps will be saved here.

mkdir -p ${OUTDIR}

# We select the most confident region proposals from mask r-cnn.
# Confidence threshold can be set to 0.9 or higher.
python gen_segmentation.py \
    --threshold 0.9 \
    --input_file ${PRED_VOC} \
    --output_file ${CONF_PRED_VOC} \
    --gt_file ${GT_VOC} \
    --save_dir ${OUTDIR}
