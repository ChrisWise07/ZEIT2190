import torch
import os

from art.estimators.object_detection import PyTorchFasterRCNN
from torchvision import transforms

COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

#Create ART object detector
FRCNN = PyTorchFasterRCNN(
    clip_values=(0, 255), attack_losses=["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
)

TRANSFORM = transforms.Compose([transforms.ToTensor(),])

DEVICE: torch.device
if not torch.cuda.is_available():
    DEVICE = torch.device("cpu")
else:
    cuda_idx = torch.cuda.current_device()
    DEVICE = torch.device(f"cuda:{cuda_idx}")


ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

DATA_DIRECTORY = ROOT_DIRECTORY + "/data/"

PATCHED_IMAGE_PATH = DATA_DIRECTORY + "temp/patch_image.jpg"

INITIAL_PREDICTIONS_DIRECTORY = DATA_DIRECTORY + "initial_predictions/"

IMAGES_DIRECTORY = DATA_DIRECTORY + "images/"

PATCHES_DIRECTORY = DATA_DIRECTORY + "patches_adv/"

ADVERSARIAL_PREDICTIONS_DIRECTORY = DATA_DIRECTORY + "predictions_adv/"

ADVERSARIAL_IMAGES_DIRECTORY =  DATA_DIRECTORY + "images_adv/"

TRAINING_DATA_DIRECTORY =  DATA_DIRECTORY + "attack_training_data/"

TRAINING_PROGRESS_FILE_PATH = DATA_DIRECTORY + "training_progress_file.txt"

PLOTS_DIRECTORY = DATA_DIRECTORY + "plots/"