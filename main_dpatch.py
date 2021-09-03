from typing import Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import imagehash
import PIL
import torch
import tensorflow as tf

from art.estimators.object_detection import PyTorchFasterRCNN
from art.attacks.evasion import DPatch
from art.attacks.evasion import RobustDPatch
from differential_color_functions import rgb2lab_diff, ciede2000_diff
from math import pi, cos
from PIL import Image

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
    print("\npredicted classes:", predictions_class)

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

    return predictions_class, predictions_boxes, predictions_class


def plot_image_with_boxes(img, boxes, pred_cls, file_location_name):
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

    save_image(img, file_location_name)


def save_image(img, file_location_name):
        plt.axis("off")
        plt.imshow(img.astype(np.uint64), interpolation="nearest")
        plt.savefig(file_location_name, bbox_inches="tight", pad_inches = 0)


def generate_print_predictions(object_detector, image, file_location_name):
    #generate predictions
    predictions = object_detector.predict(x=image)

    # Process predictions
    predictions_class, predictions_boxes, predictions_class = extract_predictions(predictions[0])

    # Plot predictions
    plot_image_with_boxes(img=image[0].copy(), boxes=predictions_boxes, 
                            pred_cls=predictions_class, file_location_name = file_location_name)

    return predictions_boxes[0]

def quantization(x):
   """quantize the continus image tensors into 255 levels (8 bit encoding)"""
   x_quan=torch.round(x*255)/255 
   return x_quan

def colour_perception_loss(image, patch):
    device  = torch.device("cpu")
    inputs_LAB=rgb2lab_diff(image,device)
    mask_isadv= torch.zeros(1,dtype=torch.uint8).to(device)
    alpha_c_init = 0.5
    alpha_c_min=alpha_c_init/10
    alpha_c=alpha_c_min+0.5*(alpha_c_init-alpha_c_min)*(1+cos(1/pi))

    d_map=ciede2000_diff(inputs_LAB,rgb2lab_diff(image+patch,device),device).unsqueeze(1)
    color_dis=torch.norm(d_map.view(1,-1),dim=1)
    color_loss=color_dis.sum()
    color_loss.backward() 
    grad_color=patch.grad.clone()
    patch.grad.zero_()
    #update patch based on colour difference
    patch.data[mask_isadv]=patch.data[mask_isadv]-alpha_c* (grad_color.permute(1,2,3,0)/torch.norm(grad_color.view(1,-1),dim=1)).permute(3,0,1,2)[mask_isadv]        

    #make sure colours are still between 0 and 1 after patch is applied.
    patch.data=(patch.data).clamp(0,1)
    #round to nearest 1/255 so colours rendered in standard RGB
    patch=quantization(patch.data)

    return patch

def main(root_dir):

    # Create ART object detector
    frcnn = PyTorchFasterRCNN(
        clip_values=(0, 255), attack_losses=["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
    )

    # Create attack
    #attack = RobustDPatch(estimator=frcnn, max_iter=20, batch_size=1)
    
    with os.scandir(root_dir + "images") as entries:
        i = 0

        for entry in entries:
            print("-"*60)

            #Read image
            image = cv2.imread(entry.path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.stack([image], axis=0).astype(np.float32)
            image_copies = np.vstack([image]*1).astype(np.float32)
            image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)

            with Image.open(entry.path) as im:
                hash = imagehash.colorhash(im, binbits=4)

            #Print predictions and return prediction boxes
            print("\nPredictions for image {}:".format(i))
            predictions_boxes = generate_print_predictions(object_detector = frcnn, image = image, 
                                                           file_location_name = (root_dir + "predictions/image_{}".format(i)))
            
            #Get centre of image and patch
            top_left_bbox = predictions_boxes[0]
            bottom_right_bbox = predictions_boxes[1]
            centre_point = (int((top_left_bbox[0] + bottom_right_bbox[0]) / 2),
                            int((top_left_bbox[1] +  bottom_right_bbox[1]) / 2))
            #Here the coordinates are store (y,x) as somewhere in Roboust Dpatch they are treated as (y,x) 
            patch_dimensions = (40,40)
            top_left_coordinates = (int(centre_point[1] - (patch_dimensions[1]/2)),
                                  int(centre_point[0] - (patch_dimensions[0]/2)))


            print("\n\nTraining adversarial patch for image {}:\n".format(i))
            # Create attack
            attack = RobustDPatch(estimator=frcnn, max_iter=1, patch_location=top_left_coordinates, 
                                  sample_size=1, learning_rate=5.0, batch_size=1, rotation_weights=(1,1,1,1), 
                                  brightness_range= (0.1,1.0))
            """
            patch_adv = attack.generate(x=image_copies, y=None)
            image_adv = attack.apply_patch(x=image)
            """

            for j in range(1):

                for k in range(1):
                    #train to trick object detector
                    patch_adv = attack.generate(x=image_copies, y=None)

                    #train to not be perceptible
                    #patch_adv = colour_perception_loss(image_tensor, patch_adv)


                image_adv = attack.apply_patch(x=image)
                #generate predictions
                predictions = frcnn.predict(x=image_adv)
                # Process predictions
                predictions_class, predictions_boxes, predictions_class = extract_predictions(predictions[0])
            

            #Save adversarial patch
            save_image(patch_adv, (root_dir + "/patches_adv/patch_{}".format(i)))
    
            #Print adversarial predictions
            print("\nAdversarial predictions for image {}:".format(i))
            generate_print_predictions(object_detector = frcnn, image = image_adv, 
                                       file_location_name = (root_dir + "predictions_adv/image_adv_{}".format(i)))

            #Save image with adversarial image but with no prediction boxes
            save_image(image_adv[0], (root_dir +"images_adv/image_adv_{}".format(i)))

            i+=1
            print("\n")
            print("-"*60)
            print("\n")

if __name__ == "__main__":
    #logging.basicConfig(level=logging.DEBUG)
    print("\n")
    main(root_dir ="./data/")


"""
 #train to not be perceptible
                    image_adv = attack.apply_patch(x=image)
                    hash_adv = imagehash.colorhash(Image.fromarray(image_adv[0].astype(np.uint8)), binbits=4)
                    hash_diff = hash_adv - hash
                    colour_gradients = np.random.uniform(low=-hash_diff, high=hash_diff, size=patch_adv.shape)
                    patch_adv = patch_adv + colour_gradients
                    
                    #update patch
                    attack._patch = patch_adv
"""