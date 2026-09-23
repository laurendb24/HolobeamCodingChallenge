# %% Import Libraries
from ultralytics import YOLO
import numpy as np
from PIL import Image
from openpyxl import Workbook

# %% Load YOLO Model
model = YOLO("lung_challenge_kit_v6/models/yolo_lung_drr.pt")  

# %% Run Model On Images
results = model.predict(
    "lung_challenge_kit_v6/data/images/test",
    imgsz=512
    )

# %% Bounding Boxes and ClassIDs
for result in results:
    BoundingBoxes = result.boxes.xyxy.cpu().numpy() 

for result in results:
    ClassID = result.boxes.cls.cpu().numpy().astype(int)
# %% Save Bounding Boxes and Class IDs
all_BoundingBoxes = []
for result in results:
    BoundingBoxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes in (x1, y1, x2, y2) format
    all_BoundingBoxes.append(BoundingBoxes)

all_ClassID = []
for result in results:
    ClassID = result.boxes.cls.cpu().numpy().astype(int)
    all_ClassID.append(ClassID)

# %% Calculate Confidence
all_Confidence = []
for result in results:
    Confidence = result.boxes.conf.cpu().numpy()
    all_Confidence.append(Confidence)

# %% Download SAM
from segment_anything import sam_model_registry, SamPredictor
SAM = sam_model_registry["vit_b"](checkpoint="lung_challenge_kit_v6/models/sam_vit_b_01ec64.pth")

predictor = SamPredictor(SAM)

# %%  Save Masks
all_SamMasks = []
for i, result in enumerate(results):
    print(f"Processing image {i} of {len(results)}")
    ImagePath = result.path
    image = Image.open(ImagePath).convert("RGB")
    ImageArray = np.array(image)

    predictor.set_image(ImageArray)

    BoundingBoxes = all_BoundingBoxes[i]
    ImageMasks = []
    for j, box in enumerate(BoundingBoxes):
        masks, scores, logits = predictor.predict(
            box=box,
            multimask_output=False,
        )
        ImageMasks.append(masks[0])

    all_SamMasks.append(ImageMasks)

print(len(all_SamMasks))
print(len(all_SamMasks[0]))
print(all_SamMasks[0][0].shape)
# %% Calculate IoU
all_IoU = []
ImageName = []
Lung = []
for i, result in enumerate(results):
    ImagePath = result.path
    MaskPath = ImagePath.replace("images","masks")
    GroundTruth = np.array(Image.open(MaskPath))
    
    RightLung = GroundTruth == 1
    LeftLung = GroundTruth == 2

    Classes = all_ClassID[i]
    ImageMasks = all_SamMasks[i]

    for j, mask in enumerate(ImageMasks):
        ClassID = Classes[j]
        PredictedMask = ImageMasks[j]
        if ClassID == 0:
            GroundTruthMask = RightLung
            Side = "Right"
        elif ClassID == 1:
            GroundTruthMask = LeftLung
            Side = "Left"
        Intersection = PredictedMask & GroundTruthMask
        Union = PredictedMask | GroundTruthMask
        IntersectionCount = np.sum(Intersection)
        UnionCount = np.sum(Union)

        if UnionCount == 0:
            IoU = 0
        else: 
            IoU = IntersectionCount / UnionCount

        all_IoU.append(IoU)
        ImageName.append(ImagePath)
        Lung.append(Side)

# %% Print Stats
RightIoU = []
LeftIoU = []

for i in range(len(all_IoU)):
    if Lung[i] == "Right":
        RightIoU.append(all_IoU[i])
    elif Lung[i] == "Left":
        LeftIoU.append(all_IoU[i])

print("Right lung mean:", np.mean(RightIoU)) 
print("Right lung median:", np.median(RightIoU))

print("Left lung mean:", np.mean(LeftIoU)) 
print("Left lung median:", np.median(LeftIoU)) 

print("Overall mean:", np.mean(all_IoU)) 
print("Overall median:", np.median(all_IoU))

# %% Save Data to Excel
workbook = Workbook()
sheet = workbook.active

sheet["A1"] = "Image"
sheet["B1"] = "Lung"
sheet["C1"] = "IoU"

for i in range(len(all_IoU)):
    sheet.cell(row=i + 2, column=1, value=ImageName[i])
    sheet.cell(row=i + 2, column=2, value=Lung[i])
    sheet.cell(row=i + 2, column=3, value=all_IoU[i])

workbook.save("IoU_Values_Output.xlsx")
# %% Organize Oracle Data
import csv

OracleBoxes = {}
OracleClasses = {}

with open("lung_challenge_kit_v6/data/boxes.csv", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        if row["split"] == "test":
            pid = row["pid"]
            ClassID = int(row["class"])
            box = [
                float(row["x0"]),
                float(row["y0"]),
                float(row["x1"]),
                float(row["y1"])]
            if pid not in OracleBoxes:
                OracleBoxes[pid] = []
            if pid not in OracleClasses:
                OracleClasses[pid] = []
            OracleBoxes[pid].append(box)
            OracleClasses[pid].append(ClassID)
# %% Save Masks
OracleMasks = []

for result in results:
    ImagePath = result.path
    ImageName = ImagePath.split("\\")[-1]
    PID = ImageName.replace(".png", "")

    image = Image.open(ImagePath).convert("RGB")
    ImageArray = np.array(image)

    predictor.set_image(ImageArray)

    OracleBox = OracleBoxes[PID]

    ImageMasks = []
    for box in OracleBox:
        masks, scores, logits = predictor.predict(
            box=np.array(box),
            multimask_output=False
        )
        ImageMasks.append(masks[0])
    OracleMasks.append(ImageMasks)
print(len(OracleBoxes))

# %% Calculate IoU
all_OracleIoU = []
OracleImageName = []
OracleLung = []

for i, result in enumerate(results):
    ImagePath = result.path
    MaskPath = ImagePath.replace("images", "masks")
    GroundTruth = np.array(Image.open(MaskPath))
    ImageName = ImagePath.split("\\")[-1]
    PID = ImageName.replace(".png", "")

    RightLung = GroundTruth == 1
    LeftLung = GroundTruth == 2

    Classes = OracleClasses[PID]
    ImageMasks = OracleMasks[i]

    for j, mask in enumerate(ImageMasks):
        ClassID = Classes[j]
        PredictedMask = ImageMasks[j]
        if ClassID == 0:
            GroundTruthMask = RightLung
            Side = "Right"
        elif ClassID == 1:
            GroundTruthMask = LeftLung
            Side = "Left"
        Intersection = PredictedMask & GroundTruthMask
        Union = PredictedMask | GroundTruthMask
        IntersectionCount = np.sum(Intersection)
        UnionCount = np.sum(Union)

        if UnionCount == 0:
            IoU = 0
        else: 
            IoU = IntersectionCount / UnionCount

        all_OracleIoU.append(IoU)
        OracleImageName.append(ImagePath)
        OracleLung.append(Side)

print(len(all_OracleIoU))
print(OracleLung.count("Right"))
print(OracleLung.count("Left"))

#%% Save to Excel
workbook = Workbook() 
sheet = workbook.active 

sheet["A1"] = "Image"
sheet["B1"] = "Lung"
sheet["C1"] = "IoU"

for i in range(len(all_OracleIoU)):
    sheet.cell(row=i + 2, column=1, value=OracleImageName[i])
    sheet.cell(row=i + 2, column=2, value=OracleLung[i])
    sheet.cell(row=i + 2, column=3, value=all_OracleIoU[i])

workbook.save("OracleIoU_Values_Output.xlsx")
# %% Calculate Stats
OracleRightIoU = []
OracleLeftIoU = []
for i in range(len(all_OracleIoU)):
    if OracleLung[i] == "Right":
        OracleRightIoU.append(all_OracleIoU[i])
    elif OracleLung[i] == "Left":
        OracleLeftIoU.append(all_OracleIoU[i])

print("Right lung mean:", np.mean(OracleRightIoU)) 
print("Right lung median:", np.median(OracleRightIoU))

print("Left lung mean:", np.mean(OracleLeftIoU)) 
print("Left lung median:", np.median(OracleLeftIoU)) 

print("Overall mean:", np.mean(all_OracleIoU)) 
print("Overall median:", np.median(all_OracleIoU))
# %% Calculate Misses
OracleMisses = 0

for i in range(len(all_OracleIoU)):
    if all_OracleIoU[i] == 0:
        OracleMisses += 1

print("Oracle Misses:", OracleMisses)