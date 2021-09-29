import os 

#not using constants or utils files here as doing so leads to circular imports
os.path.dirname(os.path.realpath(__file__))
TRAINING_PROGRESS_FILE_PATH = f"{os.path.dirname(os.path.realpath(__file__))}/data/training_progress_file.txt"

def append_to_training_progress_file(string):
                f = open(TRAINING_PROGRESS_FILE_PATH, "a")
                f.write("\n" + string)
                f.close()

class Loss_Tracker:
        _rolling_perceptibility_loss: float = 0.0
        _rolling_detection_loss: float = 0.0
        _current_perceptibility_loss: float = 0.0
        _current_detection_loss: float = 0.0

        def update_perceptibility_loss(self, loss):
                self._current_perceptibility_loss = loss
                self._rolling_perceptibility_loss = (self._rolling_perceptibility_loss * 0.99) + (loss * 0.01)

        def print_perceptibility_loss(self):
                append_to_training_progress_file(f"Current perceptibility loss: {self._current_perceptibility_loss:7f} \n")
                append_to_training_progress_file(f"Exponential rolling average perceptibility loss: {self._rolling_perceptibility_loss:7f} \n")

        def update_detection_loss(self, loss):
                self._current_detection_loss = loss
                self._rolling_detection_loss = (self._rolling_detection_loss * 0.99) + (loss * 0.01)
        
        def print_detection_loss(self):
                append_to_training_progress_file(f"Current detection loss: {self._current_detection_loss:7f} \n")
                append_to_training_progress_file(f"Exponential rolling average detection loss: {self._rolling_detection_loss:7f}\n")

        def set_loss_tracker_data(self, loss_data):
                self._rolling_perceptibility_loss = loss_data["perceptibility_loss"]
                self._rolling_detection_loss = loss_data["detection_loss"]

        def get_rolling_perceptibility_loss(self): return self._rolling_perceptibility_loss

        def get_rolling_detection_loss(self): return self._rolling_detection_loss

        def get_current_perceptibility_loss(self): return self._current_perceptibility_loss

        def get_current_detection_loss(self): return self._current_detection_loss