import cv2
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import sys
import json
import os

from json import JSONEncoder
from torch import Tensor
from PIL import Image
from matplotlib.ticker import (MultipleLocator, AutoLocator, AutoMinorLocator)

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from differential_color_functions import rgb2lab_diff, ciede2000_diff
from rrap_constants import *

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
                return obj.tolist()
        if isinstance(obj, Tensor):
                return obj.detach().numpy().tolist()
        return JSONEncoder.default(self, obj)

def append_to_training_progress_file(string):
        f = open(TRAINING_PROGRESS_FILE_PATH, "a")
        f.write("\n" + string)
        f.close()

def extract_predictions(predictions_):
        # Get the predicted class
        predictions_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(predictions_["labels"])]
        append_to_training_progress_file(f"predicted classes: {str(predictions_class)}")

        # Get the predicted bounding boxes
        predictions_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(predictions_["boxes"])]

        # Get the predicted prediction score
        predictions_score = list(predictions_["scores"])
        append_to_training_progress_file(F"predicted score: {str(predictions_score)}")

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
        plt.close()

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

def get_image_as_tensor(image_path, image_size, need_grad):
        image_tensor = TRANSFORM(Image.open(image_path).resize(image_size))
        return image_tensor.clone().detach().requires_grad_(need_grad)
        
def get_rgb_diff(image_tensor):
        image_tensor = torch.stack([image_tensor], dim=0)
        return rgb2lab_diff(image_tensor,DEVICE) 

def calculate_perceptibility_gradients_between_images(og_image, loss_tracker):
        patched_image_tensor = get_image_as_tensor(PATCHED_IMAGE_PATH, image_size=og_image.get_image_size(), need_grad=True)
        patched_image_rgb_diff = get_rgb_diff(patched_image_tensor)
        d_map=ciede2000_diff(og_image.get_image_rbg_diff(), patched_image_rgb_diff, DEVICE).unsqueeze(1)
        perceptibility_dis=torch.norm(d_map.view(1,-1),dim=1)
        perceptibility_loss = perceptibility_dis.sum()
        loss_tracker.update_perceptibility_loss(perceptibility_loss.item())
        perceptibility_loss.backward(retain_graph=True)
        perceptibility_grad = patched_image_tensor.grad.cpu().numpy().copy()
        return perceptibility_grad

def get_perceptibility_gradients_of_patch(og_image, patched_image, patch_shape, patch_location, loss_tracker):
        save_image(patched_image, PATCHED_IMAGE_PATH)
        patch_perceptibility_gradients = calculate_perceptibility_gradients_between_images(og_image, loss_tracker)
        patch_perceptibility_gradients = np.reshape(patch_perceptibility_gradients, patched_image.shape)
        return patch_perceptibility_gradients[patch_location[0]:patch_location[0]+patch_shape[0],
                                              patch_location[1]:patch_location[1]+patch_shape[1],
                                              :]
        
def get_previous_training_data(training_data_path):
        try:
                f = open(training_data_path)
                training_data = json.load(f)
                f.close()
                return training_data
        except FileNotFoundError:
                return None

def record_attack_training_data(attack, step_number):
        training_data = {}
        training_data["step_number"] = str(step_number)
        training_data["detection_learning_rate"] = attack.get_detection_learning_rate()
        training_data["perceptibility_learning_rate"] = attack.get_perceptibility_learning_rate()
        loss_tracker = attack.get_loss_tracker()
        training_data["loss_data"] = {"perceptibility_loss": loss_tracker.get_rolling_perceptibility_loss(), 
                                      "detection_loss": loss_tracker.get_rolling_detection_loss()}
        training_data["patch_np_array"] = attack.get_patch()
        training_data["old_patch_detection_update"] = np.array(attack.get_old_patch_detection_update())
        training_data["old_patch_perceptibility_update"] = np.array(attack.get_old_patch_perceptibility_update())
        f = open(attack.get_training_data_path(),'w')
        json.dump(training_data, f, cls=NumpyArrayEncoder)
        f.close()

def get_previous_steps(training_data_path):
        try:
                f = open(training_data_path)
                training_data = json.load(f)
                f.close()
                return int(training_data["step_number"])
        except FileNotFoundError:
                return 0

def cal_custom_patch_location(prediction_centre_points, patch_shape):
    #Here the coordinates are store (y,x) as somewhere in Robust Dpatch they are treated as (y,x) 
    return ((int(prediction_centre_points[1] - (patch_shape[0]/2)),
             int(prediction_centre_points[0] - (patch_shape[1]/2))))  

def get_custom_patch_shape_location(image):

        prediction_box_centre_points = image.calculate_centre_point_of_prediction_boxes()
        prediction_box_size = image.calculate_size_of_prediction_boxes()

        #in the format (height, width, nb_channels) to Dpatch Requirements
        patch_shape = (int(1/5 * prediction_box_size[1]), int(1/5 * prediction_box_size[0]), 3)
        patch_location = cal_custom_patch_location(prediction_box_centre_points, patch_shape)

        return patch_shape, patch_location

def plot_data(rolling_loss_history, current_loss_history, lr_history, image_num, loss_type):
        # create figure and axis objects with subplots()
        fig,ax = plt.subplots()

        # set x-axis label
        ax.set_xlabel("Steps",fontsize=10)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        #ax.xaxis.set_major_formatter(str("{x:.0f}"))
        ax.xaxis.set_minor_locator(MultipleLocator(10))

        # make a plot
        ax.plot(rolling_loss_history, '-r', label=f'Currrent {loss_type} Loss')
        ax.plot(current_loss_history, '-g', label=f'Rolling {loss_type} Loss')
        ax.legend(loc='upper left')
        # set y-axis label
        ax.set_ylabel(f"{loss_type} Loss", fontsize=10)
        ax.yaxis.set_major_locator(AutoLocator())
        #ax.yaxis.set_major_formatter(f'{x:.0f}"')
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        # twin object for two different y-axis on the sample plot
        ax2=ax.twinx()
        # make a plot with different y-axis using second axis object
        ax2.plot(lr_history, '-b', label=f'{loss_type} Lr')
        ax2.legend(loc='upper right')
        # set y-axis label
        ax2.set_ylabel(f"{loss_type} Lr", fontsize=10)
        ax.yaxis.set_major_locator(AutoLocator())
        #ax.yaxis.set_major_formatter('{x:.0f}')
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        # save the plot as a file
        plt.title(f"{loss_type} Data Over Step Numbers", fontsize=12)
        plt.savefig(f"{PLOTS_DIRECTORY}{loss_type}_loss_data_image_{image_num}.png", format='png', dpi=100, bbox_inches='tight')
        plt.close()