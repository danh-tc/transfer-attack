"""Project-wide constants shared by crafting and evaluation scripts."""

# Fixed square crafting/eval resolution used for every model (surrogate and all
# 6 targets) so that every model sees a bit-identical perturbation. See plan
# doc section 0 ("Canvas / resize strategy") for rationale. 800 is divisible
# by 32, which every candidate backbone (ResNet/Darknet/CSPDarknet/Swin)
# downsamples by, so no padding-divisor surprises.
CANVAS = 800

COCO_ANN_FILE = "data/coco/annotations/instances_val2017.json"
COCO_IMG_DIR = "data/coco/val2017"

# ASR "is this a real detection" confidence cutoff — conventional default used
# by most mmdet demo/visualization tooling.
DEFAULT_SCORE_THR = 0.3
# Matching IoU threshold for both ASR and as the reference point when reading
# COCO mAP (which itself averages over 0.5:0.05:0.95) so ASR and mAP-drop stay
# conceptually aligned.
DEFAULT_IOU_THR = 0.5

# Original OSFD paper / config/attack_faster_rcnn.yaml hyperparameters for a
# Faster R-CNN surrogate (kept verbatim).
EPSILON = 5.0
ALPHA = 1.0
STEPS = 200
MU = 1.0        # MI momentum decay
K = 3.0         # OSFD amplification factor
THETA = 7.0     # RRB max rotation angle (degrees)
L_S = 10        # RRB rotation-center jitter (pixels)
RHO = 0.8       # RRB adaptive resizing scale-vs-bbox-size factor
S_MAX = 1.10    # RRB max resizing scale
SIGMA = 6.0     # RRB gaussian blur std (pixel units, image in [0,255])
