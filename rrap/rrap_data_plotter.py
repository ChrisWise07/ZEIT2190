import sys
import os 

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from rrap_utils import plot_data

class Data_Plotter:
    _rolling_perceptibility_loss_history: list
    _rolling_detection_loss_history: list
    _current_perceptibility_loss_history: list
    _current_detection_loss_history: list
    _detection_lr_history: list
    _perceptibility_lr_history: list
    _image_num: int

    def __init__(self, num):
        self._image_num = num
        self._rolling_perceptibility_loss_history = []
        self._rolling_detection_loss_history = []
        self._current_perceptibility_loss_history = []
        self._current_detection_loss_history = []
        self. _detection_lr_history = []
        self._perceptibility_lr_history = []

    def save_training_data(self, attack):
        loss_tracker = attack.get_loss_tracker()

        self._rolling_perceptibility_loss_history.append(loss_tracker.get_rolling_perceptibility_loss())
        self._rolling_detection_loss_history.append(loss_tracker.get_rolling_perceptibility_loss())

        self._current_perceptibility_loss_history.append(loss_tracker.get_current_perceptibility_loss())
        self._current_detection_loss_history.append(loss_tracker.get_current_perceptibility_loss())

        self._detection_lr_history.append(attack.get_detection_learning_rate())
        self._perceptibility_lr_history.append(attack.get_perceptibility_learning_rate())

    def plot_training_data(self):
        plot_data(self._rolling_detection_loss_history, self._current_detection_loss_history, 
                  self._detection_lr_history, self._image_num, 'Detection')
        plot_data(self._rolling_perceptibility_loss_history, self._current_perceptibility_loss_history, 
                  self._perceptibility_lr_history, self._image_num, 'Perceptibility')