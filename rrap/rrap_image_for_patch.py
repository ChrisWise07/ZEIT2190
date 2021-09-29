import numpy as np
import cv2
import sys
import os

from torch.functional import Tensor
from typing import Tuple

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from rrap_utils import get_rgb_diff, plot_predictions, get_image_as_tensor, append_to_training_progress_file
from rrap_constants import INITIAL_PREDICTIONS_DIRECTORY, FRCNN

class Image_For_Patch:
    _path = str
    _image_num = int
    _image_as_np_array = np.ndarray
    _image_size = Tuple[int,int]
    _image_rbg_diff = Tensor
    _predictions_boxes = []
    _image_tensor = Tensor

    def __init__(self, path, image_num):
        self._path = path
        self._image_num = image_num
        self._image_as_np_array = self._open_image_as_np_array()
        self._image_size = (self._image_as_np_array.shape[2:0:-1])
        self._image_tensor = get_image_as_tensor(path, self._image_size, need_grad=False)
        self._image_rbg_diff = get_rgb_diff(self._image_tensor)
        self._predictions_boxes = self._generate_predictions_for_image()
    
    def _open_image_as_np_array(self):
        image = cv2.imread(self._path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.stack([image], axis=0).astype(np.float32)
        return image

    def get_path(self):
        return self._path

    def get_image_as_np_array(self):
        return self._image_as_np_array

    def get_image_size(self):
        return self._image_size

    def get_image_rbg_diff(self):
        return self._image_rbg_diff

    def _generate_predictions_for_image(self):
        append_to_training_progress_file(f"\n--- Initial Predictions for image {self._image_num} ---")
        predictions_boxes = plot_predictions(object_detector = FRCNN, image = self._image_as_np_array, 
                                             path = (f"{INITIAL_PREDICTIONS_DIRECTORY}image_{self._image_num}"))
        return predictions_boxes

    def get_predictions_boxes(self):
        return self._predictions_boxes

    def calculate_centre_point_of_prediction_boxes(self):
        top_left_bbox = self._predictions_boxes[0]
        bottom_right_bbox = self._predictions_boxes[1]
        return (int((top_left_bbox[0] + bottom_right_bbox[0]) / 2),
                int((top_left_bbox[1] +  bottom_right_bbox[1]) / 2)) 
    
    def calculate_size_of_prediction_boxes(self):
        top_left_bbox = self._predictions_boxes[0]
        bottom_right_bbox = self._predictions_boxes[1]
        return (bottom_right_bbox[0] - top_left_bbox[0], bottom_right_bbox[1] - top_left_bbox[1])


    def get_image_num(self):
        return self._image_num