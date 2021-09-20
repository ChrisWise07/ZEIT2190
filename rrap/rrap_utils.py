import cv2
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import sys
import json
from json import JSONEncoder
from torch import Tensor

from PIL import Image


sys.path.append("/mnt/c/Users/Chris Wise/Documents/Programming/ZEIT2190/rrap/")
from differential_color_functions import rgb2lab_diff, ciede2000_diff
from rrap_constants import COCO_INSTANCE_CATEGORY_NAMES, TRANSFORM, DEVICE, PATCHED_IMAGE_PATH, TRAINING_PROGRESS_FILE_PATH

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
                return obj.tolist()
        if isinstance(obj, Tensor):
                return obj.detach().numpy().tolist()
        return JSONEncoder.default(self, obj)

class Loss_Tracker:
        _total_perceptibility_loss: float = 0.0
        _num_of_perceptibility_loss_calculations: int = 0
        _current_perceptibility_loss: float = 0.0
        _total_detection_loss: float = 0.0
        _num_of_detection_loss_calculations: int = 0
        _current_detection_loss: float = 0.0

        def update_perceptibility_loss(self, loss):
                self._num_of_perceptibility_loss_calculations += 1
                self._total_perceptibility_loss += loss
                self._current_perceptibility_loss = loss

        def print_perceptibility_loss(self):
                append_to_training_progress_file(f"Current perceptibility loss: {self._current_perceptibility_loss:>7f}")
                append_to_training_progress_file(f"Running average perceptibility loss: {(self._total_perceptibility_loss/self._num_of_perceptibility_loss_calculations):7f} \n")

        def update_detection_loss(self, loss):
                self._num_of_detection_loss_calculations += 1
                self._total_detection_loss += loss
                self._current_detection_loss = loss
        
        def print_detection_loss(self):
                append_to_training_progress_file(f"Current detection loss: {self._current_detection_loss:>7f}")
                append_to_training_progress_file(f"Running average detection loss: {self._total_detection_loss/self._num_of_detection_loss_calculations:7f}\n")

        def set_loss_tracker_data(self, loss_data):
                self._total_perceptibility_loss = loss_data["total_perceptibility_loss"]
                self._num_of_perceptibility_loss_calculations = loss_data["num_of_perceptibility_loss_calculations"]
                self._current_perceptibility_loss = loss_data["current_perceptibility_loss"]
                self._total_detection_loss = loss_data["total_detection_loss"]
                self._num_of_detection_loss_calculations = loss_data["num_of_detection_loss_calculations"]
                self._current_detection_loss = loss_data["current_detection_loss"]

        def get_total_perceptibility_loss(self): return self._total_perceptibility_loss
        def get_num_of_perceptibility_loss_calculations(self): return self._num_of_perceptibility_loss_calculations
        def get_current_perceptibility_loss(self): return self._current_perceptibility_loss
        def get_total_detection_loss(self): return self._total_detection_loss
        def get_num_of_detection_loss_calculations(self): return self._num_of_detection_loss_calculations
        def get_current_detection_loss(self): return self._current_detection_loss

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

def get_image_as_tensor(image_path, image_size):
        image_tensor = TRANSFORM(Image.open(image_path).resize(image_size))
        return torch.tensor(image_tensor, requires_grad=True)

def get_rgb_diff(image_tensor):
        image_tensor = torch.stack([image_tensor], dim=0)
        return rgb2lab_diff(image_tensor,DEVICE) 

def calculate_perceptibility_gradients_between_images(og_image, loss_tracker):
        patched_image_tensor = get_image_as_tensor(PATCHED_IMAGE_PATH, image_size=og_image.get_image_size())
        patched_image_rgb_diff = get_rgb_diff(patched_image_tensor)
        d_map=ciede2000_diff(og_image.get_image_rbg_diff(), patched_image_rgb_diff, DEVICE).unsqueeze(1)
        perceptibility_dis=torch.norm(d_map.view(1,-1),dim=1)
        perceptibility_loss = perceptibility_dis.sum()
        loss_tracker.update_perceptibility_loss(perceptibility_loss)
        perceptibility_loss.backward(retain_graph=True)
        perceptibility_grad = patched_image_tensor.grad.cpu().numpy().copy()
        patched_image_tensor.grad.zero_()
        return perceptibility_grad

def get_perceptibility_gradients_of_patch(og_image, patched_image, patch_shape, patch_location, loss_tracker):
        save_image(patched_image, PATCHED_IMAGE_PATH)
        patch_perceptibility_gradients = calculate_perceptibility_gradients_between_images(og_image, loss_tracker)
        patch_perceptibility_gradients = np.reshape(patch_perceptibility_gradients, patched_image.shape)
        return patch_perceptibility_gradients[patch_location[0]:patch_location[0]+patch_shape[0],
                                              patch_location[1]:patch_location[1]+patch_shape[1],
                                              :]
        
def set_attack_data(attack, training_data_path):
        f = open(training_data_path)
        training_data = json.load(f)
        f.close()
        attack.set_detection_learning_rate(training_data["detection_learning_rate"])
        attack.set_perceptibility_learning_rate(training_data["perceptibility_learning_rate"])
        attack.get_loss_tracker().set_loss_tracker_data(training_data["loss_data"])
        attack.set_patch(np.array(training_data["patch_np_array"]))
        attack.set_old_patch_detection_update(np.array(training_data["old_patch_detection_update"]))
        attack.set_old_patch_perceptibility_update(np.array(training_data["old_patch_perceptibility_update"]))

def record_attack_training_data(attack, training_data_path):
        training_data = {}
        training_data["detection_learning_rate"] = attack.get_detection_learning_rate()
        training_data["perceptibility_learning_rate"] = attack.get_perceptibility_learning_rate()
        loss_tracker = attack.get_loss_tracker()
        loss_data = {"total_perceptibility_loss": loss_tracker.get_total_perceptibility_loss(),
                        "num_of_perceptibility_loss_calculations": loss_tracker.get_num_of_perceptibility_loss_calculations(),
                        "current_perceptibility_loss": loss_tracker.get_current_perceptibility_loss(), 
                        "total_detection_loss": loss_tracker.get_total_detection_loss(),
                        "num_of_detection_loss_calculations": loss_tracker.get_num_of_detection_loss_calculations(),
                        "current_detection_loss": loss_tracker.get_current_detection_loss()}
        training_data["loss_data"] = loss_data
        training_data["patch_np_array"] = attack.get_patch()
        training_data["old_patch_detection_update"] = np.array(attack.get_old_patch_detection_update())
        training_data["old_patch_perceptibility_update"] = np.array(attack.get_old_patch_perceptibility_update())
        f = open(training_data_path,'w')
        json.dump(training_data, open(training_data_path,'w'), cls=NumpyArrayEncoder)
        f.close()