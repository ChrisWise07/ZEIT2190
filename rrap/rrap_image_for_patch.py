import numpy as np
import cv2
import sys
import os

from dataclasses import InitVar, dataclass, field
from torch.functional import Tensor
from typing import List, Tuple
from PIL import Image

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from rrap_utils import get_rgb_diff, file_handler
from rrap_constants import DATA_DIRECTORY, INITIAL_PREDICTIONS_DIRECTORY, IMAGES_DIRECTORY, TRAINING_PROGRESS_DIRECTORY, COCO_INSTANCE_CATEGORY_NAMES, TRANSFORM

@dataclass(repr=False, eq=False)
class Image_For_Patch:
    name: str
    file_type: InitVar[str] 
    object_detector: InitVar[None]
    image_as_np_array: np.ndarray = field(init=False)
    cropped_image_as_np_array: np.ndarray = field(init=False)
    patch_section_rbg_diff: Tensor = field(init=False)
    patch_size: Tuple[int, int, int] = field(init=False)
    patch_location: Tuple[int, int] = field(init=False)
    prediction_box: List[int] = field(init=False)

    def __post_init__(self, file_type, object_detector):
        self.image_as_np_array = self._open_image_as_np_array(file_type)
        self.predictions_box = self.generate_predictions_for_image(object_detector, self.image_as_np_array, path = f"{INITIAL_PREDICTIONS_DIRECTORY}{self.name}.{file_type}")
        crop_box = [self.predictions_box[0][0], self.predictions_box[0][1], self.predictions_box[1][0], self.predictions_box[1][1]]
        self.cropped_image_as_np_array = self._open_image_as_np_array(file_type, crop_box)

        #Customise patch location to centre of prediction box and patch to ratio of prediction box
        self.patch_size, self.patch_location = self.cal_custom_patch_shape_and_location()
        patch_rectangle_box = [self.patch_location[1], self.patch_location[0], self.patch_location[1] + self.patch_size[1], self.patch_location[0]+ self.patch_size[0]]
        patch_section_tensor = TRANSFORM(Image.fromarray(np.uint8(self.cropped_image_as_np_array[0])).crop(patch_rectangle_box)).detach()
        self.patch_section_rgb_diff = get_rgb_diff(patch_section_tensor)
  
        
    
    def _open_image_as_np_array(self, file_type, crop_box = None):
        image = np.asarray(Image.open(f"{IMAGES_DIRECTORY}{self.name}.{file_type}").crop(crop_box))
        image = np.stack([image], axis=0).astype(np.float32)
        return image
    
    def generate_predictions_for_image(self, object_detector, image, path):
        self.append_to_training_progress_file(f"\n--- Initial Predictions for {self.name} ---")
        predictions_boxes = self.plot_predictions(object_detector, image, path) 
                                            
        return predictions_boxes

    def plot_predictions(self, object_detector, image, path):     
        predictions_class, predictions_boxes = self.generate_predictions(object_detector, image)

        # Plot predictions
        self.plot_image_with_boxes(img=image[0].copy(), 
                              boxes=predictions_boxes, 
                              pred_cls=predictions_class, 
                              path = path)

        return predictions_boxes[0]

    def generate_predictions(self, object_detector, image):
        #generate predictions
        predictions = object_detector.predict(x=image)

        # Process predictions   
        return self.extract_predictions(predictions[0])

    def extract_predictions(self, predictions_,):
        # Get the predicted class
        predictions_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(predictions_["labels"])]

        # Get the predicted bounding boxes
        predictions_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(predictions_["boxes"])]

        # Get the predicted prediction score
        predictions_score = list(predictions_["scores"])
        
        self.append_to_training_progress_file(f"predicted classes: {str(predictions_class)} \n predicted score: {str(predictions_score)}")

        # Get a list of index with score greater than threshold
        threshold = 0.5
        predictions_t = [predictions_score.index(x) for x in predictions_score if x > threshold][-1]

        predictions_boxes = predictions_boxes[: predictions_t + 1]
        predictions_class = predictions_class[: predictions_t + 1]

        return predictions_class, predictions_boxes
    
    def plot_image_with_boxes(self, img, boxes, pred_cls, path):
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

        Image.fromarray(np.uint8(img)).save(path)

    def cal_custom_patch_shape_and_location(self):
        height, width = self.cropped_image_as_np_array.shape[1], self.cropped_image_as_np_array.shape[2]
        prediction_box_centre_points = (int(width/2), int(height/2)) 

        #in the format (height, width, nb_channels) to meet Dpatch Requirements
        patch_shape = (int(1/5 * height), int(1/5 * width), 3)
        patch_location = self.cal_custom_patch_location(prediction_box_centre_points, patch_shape)

        return patch_shape, patch_location

    def cal_custom_patch_location(self, prediction_centre_points, patch_shape): 
        return ((int(prediction_centre_points[1] - (patch_shape[0]/2)),
                int(prediction_centre_points[0] - (patch_shape[1]/2)))) 

    def append_to_training_progress_file(self, string):
        path = f"{TRAINING_PROGRESS_DIRECTORY}{self.name}_training.txt"
        file_handler(path = path, mode = "a", func= lambda f: f.write("\n" + string))