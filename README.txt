# Lung Segmentation Challenge — YOLO boxes as SAM prompts on synthetic chest radiographs

## Project Overview
The goal of this experiment is to implement a lung segmentation pipeline that uses a You Only Look Once (YOLO) detector
to generate bounding boxes for the Segment Anything Model (SAM) and evaluate the pipeline on synthetic anterior-posterior
(AP) chest radiographs.

The baseline pipeline uses YOLO to generate bounding boxes for each lung, which are then provided to SAM as prompts for
generating pixel-level segmentation masks. Segmentation performance is evaluated using per-image, per-lung intersection
over union (IoU). The baseline results are compared with an oracle evaluation in which SAM is provided with bounding boxes
derived from the ground-truth masks rather than YOLO predictions.

The next step is to improve segmentation performance through fine-tuning, using only the train set for fitting and the
val set for model selection. The final fine-tuned model is evaluated on the test set, with performance reported
using per-lung IoU, including the mean, median, distribution, and worst-case examples.

Lastly, one additional improvement is proposed based on evidence from the experiments but was not implemented due to
time constraints.

## Files

```
data/
  images/{train,val,test}/<patient>.png   8-bit grayscale 512×512 AP DRR (PNG)
  masks/{train,val,test}/<patient>.png    8-bit PNG label map, same size: 0 = background, 1 = RIGHT lung, 2 = LEFT lung
  labels/{train,val,test}/<patient>.txt   YOLO boxes, one line per lung: `<class> <cx> <cy> <w> <h>` normalized to [0, 1]
  dataset.yaml                             Ultralytics dataset file (train/val/test relative to this folder)
  split.csv                                patient id, split, source CT series, slice thickness, mask areas
  boxes.csv                                pixel boxes (x0 y0 x1 y1, x1/y1 exclusive) and mask areas for every lung
models/
  yolo_lung_drr.pt                       Ultralytics YOLO11n lung detector trained on the DRR train split (the shipped detector)
  yolo_lung_realcxr.pt                   Ultralytics YOLO11n lung detector trained on real chest radiographs (Montgomery), for comparison
  sam_vit_b_01ec64.pth                     SAM ViT-B checkpoint (Meta, unchanged)
SHA256SUMS                                 checksums of every file in the kit

| split | patients / images | masks | labels |
|---|---|---|---|
| train | 210 | 210 | 210 |
| val | 45 | 45 | 45 |
| test | 45 | 45 | 45 |

One image per patient; no patient appears in more than one split.

## Conventions

* **Class ids:** `0 = right lung`, `1 = left lung`. **Mask values:** `0` background, `1` right lung, `2` left lung.
  These are the patient's anatomical sides.
* **Display:** every image is in radiological orientation — the patient's **right** lung is on the **left** side of the
  image (as a radiologist views a chest film), superior at the top. Do **not** use horizontal-flip augmentation unless
  you also swap the class ids; a flipped image with unchanged labels is wrong.
* **Boxes** are the tight axis-aligned bounding box of each mask (min/max of its pixels), as in the paper. Detected
  boxes are the SAM prompts; the paper also evaluates "oracle" prompts built from the ground-truth masks.
* **IoU (per image, per lung):** `IoU = |P ∩ G| / |P ∪ G|` where `P` is your predicted binary mask for that lung and `G`
  the ground-truth mask (`masks/... == 1` for the right lung, `== 2` for the left). If the detector produces no box for
  a lung, `P` is empty and the IoU for that lung is 0. Report the mean over all (image, lung) pairs of the split,
  plus per-lung means, medians and the number of misses. `boxes.csv` lets you compute box IoU the same way.
* **Ground truth** is the *visible lung field*, the convention of radiologist-drawn lung masks (JSRT, Montgomery): the lung that
  a frontal radiograph shows, bounded medially by the cardiac / mediastinal silhouette and inferiorly by the diaphragm dome,
  hila included. It was derived from 3-D CT segmentations projected through the same X-ray geometry that produced the image:
  lung tissue along the ray, minus rays that cross the abdominal cavity (below the dome) and rays where mediastinal tissue
  outweighs lung (behind the heart and spine). Lung hidden behind the heart or below the dome is therefore NOT labelled.
* **Boundary construction (v6).** Each lung is one closed smooth curve: apex, lateral edge and costophrenic angle follow the
  projected lung; the diaphragm dome is the smoothed upper envelope of the projected abdominal cavity; the cardio-mediastinal
  border is the smoothed occlusion edge with hilar vessel intrusions filled (the hila belong to the field) but never crossing
  the heart or the aorta; finally every contour point is snapped to the strongest dark-to-bright edge of the image within a few
  pixels, so mask borders coincide with the visible edges. Masks never overlap, have no holes, and are one component per lung.
* The two detectors are Ultralytics models: `from ultralytics import YOLO; YOLO('models/yolo_lung_drr.pt').predict(img, imgsz=512)`.
  SAM: `from segment_anything import sam_model_registry, SamPredictor` with `sam_model_registry['vit_b'](checkpoint=...)`;
  feed a 3-channel copy of the grayscale image and `predictor.predict(box=np.array([x0, y0, x1, y1]), multimask_output=False)`.

## Repository Structure
The expected directory structure is:

lung_challenge_kit_v6/
├── data/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── masks/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── boxes.csv
│
├── models/
│   ├── yolo_lung_drr.pt
│   ├── yolo_lung_realcxr.pt
│   ├── sam_vit_b_finetuned.pth
│   └── sam_vit_b_01ec64.pth
│
├── Code/
│   ├── LungSegmentation.py
│   └── FineTuneFinal.pth
│
├── IoU_Values_Output.xlsx
├── OracleIoU_Values_Output.xlsx
├── HolobeamData.xlsx
├── Holobeam Coding Challenge Write Up.pdf
├── 10_worst_test_cases.png
└── README.md


The dataset and model checkpoints
are not included in this repository because of their file sizes.
They should be placed in the directories shown above before
running the code.

## Environment

Python ≥ 3.10

Required Python packages include:

PyTorch
numpy
pillow
torch
torchvision
ultralytics
segment-anything
openpyxl

The implementation was run on CPU because the computer did not have an NVIDIA GPU available.
However, a GPU would be reccommended.

## Baseline Evaluation

The baseline uses the provided pretrained YOLO detector and SAM ViT-B checkpoint.

For each test image:

- YOLO predicts bounding boxes for the lungs.
- The predicted bounding boxes are passed to SAM as box prompts.
- SAM generates a segmentation mask for each detected lung.
- Each predicted mask is compared with its corresponding ground-truth mask.
- IoU is calculated separately for each lung.

Image → YOLO Box → SAM → Mask → IoU

Produces a per-image, per-lung IoU and mean IoUs.

## Oracle Evaluation

An oracle experiment was performed to determine how much segmentation performance
was affected by YOLO bounding box localization.

Instead of using YOLO-predicted bounding boxes, the oracle experiment uses the
ground-truth bounding boxes provided in boxes.csv as SAM prompts.

Image → Ground Truth Box → SAM → Mask → IoU

The SAM model and IoU calculation remain unchanged. Produces a per-image, per-lung IoU and mean IoUs.

## SAM Fine-Tuning

The SAM mask decoder was fine-tuned using the training dataset. The image encoder and prompt encoder were frozen,
while the mask decoder was updated during training.

The training configuration was:

Trainable component: SAM mask decoder
Frozen components: Image encoder, prompt encoder
Loss function: BCEWithLogitsLoss
Optimizer: AdamW
Learning rate: 1e-5
Epochs: 5
Training data: 210 images
Validation data: 45 images

Outputs the final pre-lung IoU values. 

## Evaluation
Segmentation performance was measured using IoU:

IoU = Intersection / Union

Where the intersection represents pixels shared by the predicted and ground-truth masks,
and the union represents all pixels belonging to either mask.

IoU was calculated separately for each lung in each image.

If a lung was not detected by YOLO, its IoU was assigned a value of
0 in accordance with the challenge evaluation criteria.

## Results

Overall mean IoU (after Fine Tune): 0.9090321974949248
Overall mean IoU (with YOLO):  0.8914163477874033
Overall mean IoU (with Oracle): 0.8904524285148037

The negligible difference between the YOLO and oracle results suggests that bounding box localization 
was not a major source of error in the baseline. The optimization outlined increases the IoU indicating 
an improved SAM model. 

Note: These are not the only results reported during the experimentation. 
Additional metrics, including median IoU, per-lung performance, IoU distribution, 
and worst-case examples, are also reported for the final fine-tuned model. The three
mean IoU values above provide the primary comparison between the baseline, 
oracle, and fine-tuned configurations.

## Limitations and Future Work

The primary limitation of this implementation was computational ability. 
The experiments were performed entirely on CPU, which significantly increased
training time and limited the number of configurations that could be evaluated.

With additional computational power, future experiments would include:

- Comparing Adam and AdamW using the validation set
- Testing multiple learning rates
- Evaluating additional numbers of epochs
- Investigating alternative loss functions such as Dice loss
- Evaluating fine-tuning of additional SAM components

These experiments would allow the fine-tuning configuration to be selected more 
systematically based on validation performance.

## Reference

Khalili, M. et al. "Automatic lung segmentation in chest X-ray images using SAM with prompts from YOLO." 
IEEE Access, vol. 12, 2024, pp. 122805–122816.
