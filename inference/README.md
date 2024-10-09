## Table of Contents
- [Nipple Detection](#nipple_detection)
- [Pectoral Muscle Segmentation](#pectoral_muscle_segmentation)
- [View-Orientation Classification](#view_orient_classification)


## Nipple Detection

To run nipple detection on a set of images:

```
python inference/nipple_detection_inference.py -c path/to/checkpoint -i path/to/image/directory
```

Options:
- `-c`, `--checkpoint_path`: Path to the model checkpoint (default: `D:\Code\MammoQC\runs\detect\train4\weights\best.pt`)
- `-i`, `--image_dir`: Path to the input image directory (default: `inference/inference_imgs`)

## Pectoral Muscle Segmentation

To perform pectoral muscle segmentation:

```
python inference/pectoral_muscle_segmentation_inference.py -c path/to/checkpoint -i path/to/image/directory
```

Options:
- `-c`, `--checkpoint_path`: Path to the model checkpoint (default: `checkpoints/pectoral-segmentation-unet-512-epoch=06-val_dice_coeff=0.97.ckpt`)
- `-i`, `--image_dir`: Path to the input image directory (default: `inference/inference_imgs`)

## View-Orientation Classification

To classify mammogram views and orientations:

```
python inference/view_orientation_clasiffication_inference.py -c path/to/checkpoint -i path/to/image/directory
```

Options:
- `-c`, `--checkpoint_path`: Path to the model checkpoint (default: `checkpoints/res2next-mammography-epoch=09-val_loss=0.00.ckpt`)
- `-i`, `--image_dir`: Path to the input image directory (default: `inference/inference_imgs`)