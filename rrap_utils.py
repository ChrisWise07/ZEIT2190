from os import path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import sys
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim

from torchvision import transforms
from PIL import Image

TRANSFORM = transforms.Compose([transforms.ToTensor(),])
DEVICE = torch.device("cpu")
ROOT_DIRECTORY = "/mnt/c/Users/Chris Wise/Documents/Programming/ZEIT2190/rrap/"
PATCHED_IMAGE_PATH = (ROOT_DIRECTORY + "data/temp/patch_image.jpg")

sys.path.append(ROOT_DIRECTORY)
from differential_color_functions import rgb2lab_diff, ciede2000_diff


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

def extract_predictions(predictions_):
        # Get the predicted class
        predictions_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(predictions_["labels"])]
        print("predicted classes:", predictions_class)

        # Get the predicted bounding boxes
        predictions_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(predictions_["boxes"])]

        # Get the predicted prediction score
        predictions_score = list(predictions_["scores"])
        print("predicted score:", predictions_score)

        # Get a list of index with score greater than threshold
        threshold = 0.5
        predictions_t = [predictions_score.index(x) for x in predictions_score if x > threshold][-1]

        predictions_boxes = predictions_boxes[: predictions_t + 1]
        predictions_class = predictions_class[: predictions_t + 1]

        return predictions_class, predictions_boxes

def save_image(img, path):
        plt.axis("off")
        plt.imshow(img.astype(np.uint64), interpolation="nearest")
        plt.savefig(path, bbox_inches="tight", pad_inches = 0)

def plot_image_with_boxes(img, boxes, pred_cls, path):
        text_size = 2
        text_th = 2
        rect_th = 2

        for i in range(len(boxes)):
                # Draw Rectangle with the coordinates
                cv2.rectangle(img, (int(boxes[i][0][0]), int(boxes[i][0][1])), (int(boxes[i][1][0]), int(boxes[i][1][1])), 
                        color=(0, 255, 0), thickness=rect_th)

                # Write the prediction class
                cv2.putText(img, pred_cls[i], (int(boxes[i][0][0]), int(boxes[i][0][1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)

        save_image(img, path)

def generate_predictions(object_detector, image):
        #generate predictions
        predictions = object_detector.predict(x=image)

        # Process predictions
        return extract_predictions(predictions[0])   

def plot_predictions(object_detector, image, path):       


    predictions_class, predictions_boxes = generate_predictions(object_detector, image)

    # Plot predictions
    plot_image_with_boxes(img=image[0].copy(), 
                          boxes=predictions_boxes, 
                          pred_cls=predictions_class, 
                          path = path)

    return predictions_boxes[0]

def get_rgb_diff(image_path, image_size):
        image_tensor = TRANSFORM(Image.open(image_path).resize(image_size))
        image_tensor = torch.stack([image_tensor], dim=0)
        return rgb2lab_diff(image_tensor,DEVICE) 

def get_perceptibility_loss_between_image_and_patched_image(og_image, patched_image):
        save_image(patched_image, PATCHED_IMAGE_PATH)
        patched_image_rgb_diff = get_rgb_diff(PATCHED_IMAGE_PATH, image_size=og_image.get_image_size())
        d_map=ciede2000_diff(og_image.get_image_rbg_diff(), patched_image_rgb_diff, DEVICE).unsqueeze(1)
        colour_dis=torch.norm(d_map.view(1,-1),dim=1)
        return colour_dis.sum()