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
                        print(f"Current perceptibility loss: {self._current_perceptibility_loss:>7f}")
                        print(f"Running average perceptibility loss: {(self._total_perceptibility_loss/self._num_of_perceptibility_loss_calculations):7f} \n")

        def update_detection_loss(self, loss):
                self._num_of_detection_loss_calculations += 1
                self._total_detection_loss += loss
                self._current_detection_loss = loss
        
        def print_detection_loss(self):
                        print(f"Current detection loss: {self._current_detection_loss:>7f}")
                        print(f"Running average detection loss: {self._total_detection_loss/self._num_of_detection_loss_calculations:7f}\n")