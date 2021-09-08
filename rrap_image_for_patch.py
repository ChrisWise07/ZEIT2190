import numpy as np
import cv2
import sys


from torch.functional import Tensor
from typing import Tuple

ROOT_DIRECTORY = "/mnt/c/Users/Chris Wise/Documents/Programming/ZEIT2190/rrap/"
DATA_DIRECTORY = ROOT_DIRECTORY + "data/"
sys.path.append(ROOT_DIRECTORY)
from rrap_utils import get_rgb_diff, plot_predictions

class Image_For_Patch:
    _path = str
    _image_num = int
    _image_as_np_array = np.ndarray
    _image_size = Tuple[int,int]
    _image_rbg_diff = Tensor
    _predictions_boxes = []
    _centre_point_of_prediction_boxes = []

    def __init__(self, path, image_num, object_detector) -> None:
        self._path = path
        self._image_num = image_num
        self._image_as_np_array = self._open_image_as_np_array()
        self._image_size = (self._image_as_np_array.shape[2:0:-1])
        self._image_rbg_diff = get_rgb_diff(image_path = self._path, image_size = self._image_size)
        self._predictions_boxes = self._generate_predictions_for_image(object_detector)
        self._centre_point_of_prediction_boxes = self._calculate_centre_point_of_prediction_boxes()
    
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

    def _generate_predictions_for_image(self, object_detector):
        print("\n--- Initial Predictions for image {} ---".format(self._image_num))
        predictions_boxes = plot_predictions(object_detector = object_detector,
                                             image = self._image_as_np_array, 
                                             path = (DATA_DIRECTORY + "initial_predictions/image_{}".format(self._image_num)))
        return predictions_boxes

    def get_predictions_boxes(self):
        return self._predictions_boxes

    def _calculate_centre_point_of_prediction_boxes(self):
        top_left_bbox = self._predictions_boxes[0]
        bottom_right_bbox = self._predictions_boxes[1]
        return (int((top_left_bbox[0] + bottom_right_bbox[0]) / 2),
                int((top_left_bbox[1] +  bottom_right_bbox[1]) / 2)) 

    def get_centre_point_of_prediction_boxes(self):
        return self._centre_point_of_prediction_boxes

    def get_image_num(self):
        return self._image_num