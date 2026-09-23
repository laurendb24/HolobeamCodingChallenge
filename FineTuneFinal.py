# %% Imports and Environment

from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

import numpy as np
from PIL import Image
from pathlib import Path
import csv
import torch
import matplotlib.pyplot as plt

torch.set_num_threads(16)

device = torch.device("cpu")

print("Device:", device)
print("PyTorch threads:", torch.get_num_threads())
# %% Load SAM

SAM = sam_model_registry["vit_b"](
    checkpoint="lung_challenge_kit_v6/models/sam_vit_b_01ec64.pth")
SAM.to(device)

print("SAM loaded")
# %% Configure SAM for Training

for parameter in SAM.image_encoder.parameters():
    parameter.requires_grad = False
for parameter in SAM.prompt_encoder.parameters():
    parameter.requires_grad = False
for parameter in SAM.mask_decoder.parameters():
    parameter.requires_grad = True
SAM.mask_decoder.train()
# %% Load Training and Validation Images

TrainImages = sorted(
    Path("lung_challenge_kit_v6/data/images/train").glob("*.png"))
ValImages = sorted(
    Path("lung_challenge_kit_v6/data/images/val").glob("*.png"))
# %% Load Training and Validation Boxes

TrainBoxes = []
ValBoxes = []

with open(
    "lung_challenge_kit_v6/data/boxes.csv",
    newline="") as file:
    reader = csv.DictReader(file)
    for row in reader:
        Box = [
            float(row["x0"]),
            float(row["y0"]),
            float(row["x1"]),
            float(row["y1"])]
        if row["split"] == "train":
            TrainBoxes.append([
                row["pid"],
                int(row["class"]),
                Box,
                str(
                    Path("lung_challenge_kit_v6/data/masks/train")
                    / f"{row['pid']}.png")])
        elif row["split"] == "val":

            ValBoxes.append([
                row["pid"],
                int(row["class"]),
                Box,
                str(
                    Path("lung_challenge_kit_v6/data/masks/val")
                    / f"{row['pid']}.png")])
# %% Resizing for SAM

Transform = ResizeLongestSide(
    SAM.image_encoder.img_size)

print("SAM image size:", SAM.image_encoder.img_size)


# %% Loss Function and Optimizer

LossFunction = torch.nn.BCEWithLogitsLoss()

optimize = torch.optim.AdamW(
    SAM.mask_decoder.parameters(),
    lr=1e-5)

print("Loss function: BCEWithLogitsLoss")
print("Optimizer: AdamW")
print("Learning rate: 1e-5")
# %% Store Embedding for Future 

TrainEmbeddingFolder = Path(
    "lung_challenge_kit_v6/embeddings/train")

ValEmbeddingFolder = Path(
    "lung_challenge_kit_v6/embeddings/val")

TrainEmbeddingFolder.mkdir(
    parents=True,
    exist_ok=True)

ValEmbeddingFolder.mkdir(
    parents=True,
    exist_ok=True)
# %% Training

for i, ImagePath in enumerate(TrainImages):

    print(
        f"Processing training image "
        f"{i + 1}/{len(TrainImages)}: {ImagePath.stem}")
    ImageArray = np.array(
        Image.open(ImagePath).convert("RGB"))
    InputImage = torch.as_tensor(
        ImageArray,
        device=device)
    InputImage = InputImage.permute(2, 0, 1).contiguous()
    InputImage = InputImage[None, :, :, :]
    InputImage = SAM.preprocess(InputImage)
    with torch.no_grad():
        ImageEmbedding = SAM.image_encoder(InputImage)
    EmbeddingPath = (
        TrainEmbeddingFolder
        / f"{ImagePath.stem}.pt")
    torch.save(
        ImageEmbedding.cpu(),
        EmbeddingPath)
# %% Validation

for i, ImagePath in enumerate(ValImages):

    print(
        f"Processing validation image "
        f"{i + 1}/{len(ValImages)}: {ImagePath.stem}")

    ImageArray = np.array(Image.open(ImagePath).convert("RGB"))
    InputImage = torch.as_tensor(
        ImageArray,
        device=device)
    InputImage = InputImage.permute(2, 0, 1).contiguous()
    InputImage = InputImage[None, :, :, :]
    InputImage = SAM.preprocess(InputImage)

    with torch.no_grad():
        ImageEmbedding = SAM.image_encoder(
            InputImage)

    EmbeddingPath = (
        ValEmbeddingFolder
        / f"{ImagePath.stem}.pt")

    torch.save(
        ImageEmbedding.cpu(),
        EmbeddingPath)
# %% Train Mask Decoder

NumberOfEpochs = 5

for Epoch in range(NumberOfEpochs):
    SAM.mask_decoder.train()
    TrainLoss = []

    print(
        f"\nStarting epoch "
        f"{Epoch + 1}/{NumberOfEpochs}")
    
    TrainingPIDs = sorted(
        set(lung[0] for lung in TrainBoxes))

    for PID in TrainingPIDs:
        EmbeddingPath = (
            TrainEmbeddingFolder
            / f"{PID}.pt")
        ImageEmbedding = torch.load(
            EmbeddingPath,
            map_location=device)
        lungs = [
            lung
            for lung in TrainBoxes
            if lung[0] == PID]
        for lung in lungs:
            ClassID = lung[1]
            Box = lung[2]
            MaskPath = lung[3]

            BoxTensor = torch.tensor(
                Box,
                dtype=torch.float32,
                device=device)

            BoxTensor = Transform.apply_boxes_torch(
                BoxTensor[None, :],
                (512, 512))
            
            with torch.no_grad():
                SparseEmbeddings, DenseEmbeddings = (
                    SAM.prompt_encoder(
                        points=None,
                        boxes=BoxTensor,
                        masks=None))

            LowResMasks, IoUPredictions = (
                SAM.mask_decoder(
                    image_embeddings=ImageEmbedding,
                    image_pe=SAM.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=SparseEmbeddings,
                    dense_prompt_embeddings=DenseEmbeddings,
                    multimask_output=False))

            GroundTruth = np.array(
                Image.open(MaskPath))

            if ClassID == 0:
                GroundTruth = (
                    GroundTruth == 1)
            else:
                GroundTruth = (
                    GroundTruth == 2)

            GroundTruthTensor = torch.tensor(
                GroundTruth,
                dtype=torch.float32,
                device=device)

            GroundTruthTensor = (
                torch.nn.functional.interpolate(
                    GroundTruthTensor[None, None, :, :],
                    size=(256, 256),
                    mode="nearest"))

            Loss = LossFunction(
                LowResMasks,
                GroundTruthTensor)

            optimize.zero_grad()
            Loss.backward()
            optimize.step()
            TrainLoss.append(
                Loss.item()
            )

    print(
        f"Epoch {Epoch + 1}/{NumberOfEpochs} "
        f"- Average training loss: "
        f"{np.mean(TrainLoss):.4f}")
# %% Save Fine Tuned Model

FineTunedModelPath = (
    "lung_challenge_kit_v6/models/"
    "sam_vit_b_finetuned.pt")

torch.save(
    SAM.state_dict(),
    FineTunedModelPath)

# %% Validate Fine Tuned SAM

SAM.eval()
FineTunedPredictor = SamPredictor(SAM)
ValidationIoUs = []

for ImagePath in ValImages:

    PID = ImagePath.stem

    ImageArray = np.array(
        Image.open(ImagePath).convert("RGB"))
    FineTunedPredictor.set_image(ImageArray)
    lungs = [
        lung
        for lung in ValBoxes
        if lung[0] == PID]

    for lung in lungs:

        ClassID = lung[1]
        Box = lung[2]
        MaskPath = lung[3]

        PredictedMasks, _, _ = (
            FineTunedPredictor.predict(
                box=np.array(Box),
                multimask_output=False))

        PredictedMask = PredictedMasks[0]

        GroundTruth = np.array(
            Image.open(MaskPath))

        if ClassID == 0:
            GroundTruth = (
                GroundTruth == 1)
        else:
            GroundTruth = (
                GroundTruth == 2)

        Intersection = np.logical_and(
            PredictedMask,
            GroundTruth).sum()

        Union = np.logical_or(
            PredictedMask,
            GroundTruth).sum()

        if Union > 0:
            IoU = Intersection / Union
        else:
            IoU = 0
        ValidationIoUs.append(IoU)
# %% Fine Tuned Validation Results

print("\nFINE-TUNED SAM VALIDATION RESULTS")
print("Mean IoU:", np.mean(ValidationIoUs))
print("Median IoU:", np.median(ValidationIoUs))
print("Minimum IoU:", np.min(ValidationIoUs))
print("Maximum IoU:", np.max(ValidationIoUs))
print("Validation examples:", len(ValidationIoUs))
# %% Load Original SAM

OriginalSAM = sam_model_registry["vit_b"](
    checkpoint="lung_challenge_kit_v6/models/sam_vit_b_01ec64.pth")

OriginalSAM.to(device)
OriginalSAM.eval()
OriginalPredictor = SamPredictor(OriginalSAM)

print("\nOriginal SAM loaded.")


# %% Original SAM Validation

OriginalValIoUs = []

for ImagePath in ValImages:

    PID = ImagePath.stem

    ImageArray = np.array(
        Image.open(ImagePath).convert("RGB"))

    OriginalPredictor.set_image(
        ImageArray)

    lungs = [
        lung
        for lung in ValBoxes
        if lung[0] == PID]
    for lung in lungs:
        ClassID = lung[1]
        Box = lung[2]
        MaskPath = lung[3]
        Masks, _, _ = (
            OriginalPredictor.predict(
                box=np.array(Box),
                multimask_output=False))

        PredictedMask = Masks[0]
        GroundTruth = np.array(
            Image.open(MaskPath))

        if ClassID == 0:
            GroundTruth = (
                GroundTruth == 1)
        else:
            GroundTruth = (
                GroundTruth == 2)

        Intersection = np.logical_and(
            PredictedMask,
            GroundTruth).sum()

        Union = np.logical_or(
            PredictedMask,
            GroundTruth).sum()
        if Union > 0:
            IoU = Intersection / Union
        else:
            IoU = 0
        OriginalValIoUs.append(IoU)
# %% Original SAM Validation Results

print("\nORIGINAL SAM VALIDATION RESULTS")
print("Mean IoU:", np.mean(OriginalValIoUs))
print("Median IoU:",np.median(OriginalValIoUs))
print("Minimum IoU:", np.min(OriginalValIoUs))
print("Maximum IoU:", np.max(OriginalValIoUs))
print("Validation examples:", len(OriginalValIoUs))


# %% Final Test Evaluation

SAM.eval()

FineTunedPredictor = SamPredictor(SAM)
TestImages = sorted(
    Path("lung_challenge_kit_v6/data/images/test").glob("*.png"))
TestModel = YOLO(
    "lung_challenge_kit_v6/models/yolo_lung_drr.pt")
TestResults = TestModel.predict(
    "lung_challenge_kit_v6/data/images/test",
    imgsz=512)

FinalResults = []

for i, (ImagePath, Result) in enumerate(
    zip(TestImages, TestResults)):
    PID = ImagePath.stem
    ImageArray = np.array(
        Image.open(ImagePath).convert("RGB"))
    FineTunedPredictor.set_image(
        ImageArray)
    GroundTruth = np.array(
        Image.open(
            f"lung_challenge_kit_v6/data/masks/test/{PID}.png"))
    Boxes = Result.boxes.xyxy.cpu().numpy()
    Classes = (
        Result.boxes.cls
        .cpu()
        .numpy()
        .astype(int))

    for ClassID in [0, 1]:
        if ClassID == 0:
            LungSide = "right"
            TrueMask = (
                GroundTruth == 1)
        else:
            LungSide = "left"
            TrueMask = (
                GroundTruth == 2)

        MatchingIndices = np.where(
            Classes == ClassID)[0]

        if len(MatchingIndices) == 0:
            FinalResults.append({
                "PID": PID,
                "Lung": LungSide,
                "ClassID": ClassID,
                "Detected": False,
                "IoU": 0.0})
            continue

        Box = Boxes[
            MatchingIndices[0]]

        PredictedMasks, _, _ = (
            FineTunedPredictor.predict(
                box=Box,
                multimask_output=False))
        PredictedMask = PredictedMasks[0]

        Intersection = np.logical_and(
            PredictedMask,
            TrueMask).sum()

        Union = np.logical_or(
            PredictedMask,
            TrueMask).sum()

        if Union > 0:
            IoU = Intersection / Union
        else:
            IoU = 0.0
        FinalResults.append({
            "PID": PID,
            "Lung": LungSide,
            "ClassID": ClassID,
            "Detected": True,
            "IoU": IoU})
    print(
        f"Processed {i + 1}/{len(TestImages)}: {PID}")
# %% Final Test 

AllIoUs = np.array([
    result["IoU"]
    for result in FinalResults])
RightIoUs = np.array([
    result["IoU"]
    for result in FinalResults
    if result["Lung"] == "right"])
LeftIoUs = np.array([
    result["IoU"]
    for result in FinalResults
    if result["Lung"] == "left"])
Misses = [
    result
    for result in FinalResults
    if not result["Detected"]]

print("\nFINAL TEST RESULTS")
print("Total lung cases:", len(AllIoUs))
print("Missed lungs:", len(Misses))
print("Overall mean IoU:",np.mean(AllIoUs))
print("Overall median IoU:", np.median(AllIoUs))
print("Overall minimum IoU:", np.min(AllIoUs))
print("Overall maximum IoU:", np.max(AllIoUs))
print("\nRIGHT LUNG")
print("Mean:",np.mean(RightIoUs))
print("Median:", np.median(RightIoUs))
print("\nLEFT LUNG")
print("Mean:", np.mean(LeftIoUs))
print("Median:", np.median(LeftIoUs))
# %% Distributions

IoUValues = np.array([
    result["IoU"]
    for result in FinalResults])

print("\nIoU DISTRIBUTION")
print("IoU >= 0.90:", np.sum(IoUValues >= 0.90))
print("IoU 0.80-0.90:", np.sum((IoUValues >= 0.80)&(IoUValues < 0.90)))
print("IoU 0.70-0.80:", np.sum((IoUValues >= 0.70)&(IoUValues < 0.80)))
print("IoU < 0.70:", np.sum(IoUValues < 0.70))
# %% 10 Worst Test Cases

WorstResults = sorted(
    FinalResults,
    key=lambda result: result["IoU"])

print("\n10 WORST CASES")

for result in WorstResults[:10]:
    print(
        result["PID"],
        "|",
        result["Lung"],
        "| IoU:",
        round(result["IoU"], 4),
        "| Detected:",
        result["Detected"])
# %% 10 Worst-Case Visualizations 

Worst10 = WorstResults[:10]

print(
    "\nCreating",
    len(Worst10),
    "worst-case visualizations...")

Figure, Axes = plt.subplots(
    10,
    3,
    figsize=(12, 35))

for Row, Case in enumerate(Worst10):

    PID = Case["PID"]
    Side = Case["Lung"]
    IoU = Case["IoU"]

    print(
        f"Creating image {Row + 1}/10: "
        f"{PID} | {Side} | IoU = {IoU:.4f}")

    ImageIndex = next(i for i, ImagePath in enumerate(TestImages)
        if ImagePath.stem == PID)

    ImageArray = np.array(
        Image.open(
            TestImages[ImageIndex]).convert("RGB"))

    Result = TestResults[ImageIndex]
    Boxes = Result.boxes.xyxy.cpu().numpy()
    Classes = (
        Result.boxes.cls
        .cpu()
        .numpy()
        .astype(int))

    if Side == "right":
        TargetClass = 0
        GroundTruthLabel = 1
    else:
        TargetClass = 1
        GroundTruthLabel = 2

    TargetBox = None

    for j, ClassID in enumerate(Classes):
        if ClassID == TargetClass:
            TargetBox = Boxes[j]
            break

    FineTunedPredictor.set_image(
        ImageArray)

    if TargetBox is not None:

        Masks, _, _ = (
            FineTunedPredictor.predict(
                box=TargetBox,
                multimask_output=False))

        PredictedMask = Masks[0]

    else:
        PredictedMask = np.zeros(
            ImageArray.shape[:2],
            dtype=bool)

    MaskPath = (
        Path("lung_challenge_kit_v6/data/masks/test")
        / f"{PID}.png")
    GroundTruth = np.array(
        Image.open(MaskPath))
    GroundTruthMask = (
        GroundTruth == GroundTruthLabel)

    Axes[Row, 0].imshow(
        ImageArray)

    if TargetBox is not None:

        x0, y0, x1, y1 = TargetBox

        Rectangle = plt.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            linewidth=2)
        Axes[Row, 0].add_patch(
            Rectangle)
    Axes[Row, 0].set_title(
        f"{PID} | {Side} | IoU = {IoU:.4f}")
    Axes[Row, 0].axis("off")
    Axes[Row, 1].imshow(
        ImageArray)
    Axes[Row, 1].imshow(
        PredictedMask,
        alpha=0.45)
    Axes[Row, 1].set_title(
        "Fine-tuned SAM")
    Axes[Row, 1].axis("off")
    Axes[Row, 2].imshow(
        ImageArray)
    Axes[Row, 2].imshow(
        GroundTruthMask,
        alpha=0.45)
    Axes[Row, 2].set_title(
        "Ground Truth")
    Axes[Row, 2].axis("off")


# %% Save Worst Case Figure
plt.tight_layout()
plt.savefig(
    "10_worst_test_cases.png",
    dpi=300,
    bbox_inches="tight")
plt.show()
print("Saved: 10_worst_test_cases.png")