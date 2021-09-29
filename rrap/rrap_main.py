import numpy as np
import os
import sys

from art.attacks.evasion import RobustDPatch

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from rrap_utils import *
from rrap_image_for_patch import Image_For_Patch
from rrap_constants import *
from rrap_data_plotter import Data_Plotter

def generate_adversarial_patch(attack):
    image = attack.get_image_to_patch()
    image_num = image.get_image_num()
    data_plotter = Data_Plotter(image_num)
    image_copies = np.vstack([image.get_image_as_np_array()]*1)
    previous_num_steps = get_previous_steps(attack.get_training_data_path())

    append_to_training_progress_file(f"\n\n--- Generating adversarial patch for image {image_num} ---")
    for step in range(previous_num_steps, previous_num_steps + 1):
        str = f"\n\n--- Step Number: {step} | Detection Lr: {attack.get_detection_learning_rate()}, Perception Lr: {attack.get_perceptibility_learning_rate()} ---"
        append_to_training_progress_file(str)
        
        #train adv patch to trick object detector and not to be perceptibile
        attack.generate(x=image_copies, print_nth_num=1, y=None)                    

        #Save adv patch and training data every step
        record_attack_training_data(attack, step + 1)
        data_plotter.save_training_data(attack)   

        
        #Decay learning rate every 5 steps
        if ((step + 1) % 5 == 0):
            attack.decay_detection_learning_rate()
            attack.decay_perceptibility_learning_rate()

    data_plotter.plot_training_data()

def main():
    with os.scandir(IMAGES_DIRECTORY) as entries:
        i = 0

        for entry in entries:
            append_to_training_progress_file("\n" + "-"*60)

            #Create image object and, print and save predictions
            image = Image_For_Patch(entry.path, image_num = i)
            training_data_path = (f"{TRAINING_DATA_DIRECTORY}training_data_image_{image.get_image_num()}.txt")

            #Customise patch location to centre of prediction box and patch to ratio of prediction box
            patch_shape, patch_location = get_custom_patch_shape_location(image)

            # Create attack
            attack = RobustDPatch(estimator=FRCNN, max_iter=1, batch_size=1, verbose=False, rotation_weights=(1,0,0,0), 
                                  brightness_range= (1.0,1.0), decay_rate = 0.95, detection_momentum = 0.9, perceptibility_momentum = 0.9,
                                  image_to_patch = image, training_data_path = training_data_path, patch_shape = patch_shape, 
                                  patch_location=patch_location, training_data = get_previous_training_data(training_data_path))

            #create adversarial patch and apply it to orignal image
            generate_adversarial_patch(attack)
            image_adv = attack.apply_patch(x=image.get_image_as_np_array())

            #Print predictions for image with adv patch and save image with adv patch and prediction boxes
            append_to_training_progress_file(f"\n\n--- Final predictions for image with adversarial patch {i} ---")
            plot_predictions(object_detector = FRCNN, image = image_adv, path = (f"{ADVERSARIAL_PREDICTIONS_DIRECTORY}image_adv_{i}"))

            #Save image with adv patch but with no prediction boxes
            save_image(image_adv[0], (f"{ADVERSARIAL_IMAGES_DIRECTORY}image_adv_{i}"))

            i+=1
            append_to_training_progress_file("-"*60 + "\n\n")

if __name__ == "__main__":
    main()