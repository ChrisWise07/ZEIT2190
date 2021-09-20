import numpy as np
import os
import sys
from datetime import datetime

from art.attacks.evasion import RobustDPatch
from art.estimators.object_detection import PyTorchFasterRCNN

sys.path.append("/mnt/c/Users/Chris Wise/Documents/Programming/ZEIT2190/rrap/")
from rrap_utils import save_image, plot_predictions, set_attack_data, record_attack_training_data, append_to_training_progress_file
from rrap_image_for_patch import Image_For_Patch
from rrap_constants import IMAGES_DIRECTORY, PATCHES_DIRECTORY, ADVERSARIAL_PREDICTIONS_DIRECTORY, \
                           ADVERSARIAL_IMAGES_DIRECTORY, TRAINING_DATA_DIRECTORY

#Create ART object detector
FRCNN = PyTorchFasterRCNN(
    clip_values=(0, 255), attack_losses=["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
)

def get_custom_patch_location(prediction_centre_points, patch_shape):
    #Here the coordinates are store (y,x) as somewhere in Robust Dpatch they are treated as (y,x) 
    return ((int(prediction_centre_points[1] - (patch_shape[1]/2)),
             int(prediction_centre_points[0] - (patch_shape[0]/2))))   

def generate_adversarial_patch(attack, image, training_data_path):
    append_to_training_progress_file(f"\n\n--- Generating adversarial patch for image {image.get_image_num()} ---")
    image_copies = np.vstack([image.get_image_as_np_array()]*1)

    for step in range(3):
        #train adv patch to trick object detector and not to be perceptible
        patch_adv = attack.generate(x=image_copies, og_image=image, print_nth_num=2, y=None)                        
        image_adv = attack.apply_patch(x=image.get_image_as_np_array()) 

        #Save adv patch and training data every step
        save_image(patch_adv, (PATCHES_DIRECTORY + "patch_{}".format(image.get_image_num())))
        record_attack_training_data(attack, training_data_path)
        
        #Decay learning rate every 5 steps
        if ((step + 1) % 5 == 0):
            attack.decay_detection_learning_rate()
            attack.decay_perceptibility_learning_rate()

    return image_adv

def main():
    startTime = datetime.now()
    append_to_training_progress_file(str(startTime))

    # Create attack
    attack = RobustDPatch(estimator=FRCNN, max_iter=2, batch_size=1, verbose=False, 
                          rotation_weights=(1,0,0,0), brightness_range= (1.0,1.0), decay_rate = 0.95, 
                          detection_learning_rate=0.1, perceptibility_learning_rate=0.1, 
                          detection_momentum = 0.9, perceptibility_momentum = 0.9)

    with os.scandir(IMAGES_DIRECTORY) as entries:
        i = 0

        for entry in entries:
            append_to_training_progress_file("\n" + "-"*60)

            #Create image object and, print and save predictions
            image = Image_For_Patch(entry.path, image_num = i, object_detector=FRCNN)
            attack_training_data_path = (TRAINING_DATA_DIRECTORY + f"training_data_image_{image.get_image_num()}.txt")
            if os.path.exists(attack_training_data_path):
                set_attack_data(attack, attack_training_data_path)
            else:
                record_attack_training_data(attack, attack_training_data_path)
            
            # Customise patch location to centre of prediction box
            prediction_centre_points = image.get_centre_point_of_prediction_boxes()
            patch_shape = attack.get_patch_shape()
            attack.set_patch_location(get_custom_patch_location(prediction_centre_points, patch_shape))

            #create adversarial patch and apply it to orignal image
            image_adv = generate_adversarial_patch(attack, image, attack_training_data_path)
    
            #Print predictions for image with adv patch and save image with adv patch and prediction boxes
            append_to_training_progress_file(f"\n\n--- Final predictions for image with adversarial patch {i} ---")
            plot_predictions(object_detector = FRCNN, image = image_adv, path = (ADVERSARIAL_PREDICTIONS_DIRECTORY + f"image_adv_{i}"))

            #Save image with adv patch but with no prediction boxes
            save_image(image_adv[0], (ADVERSARIAL_IMAGES_DIRECTORY + f"image_adv_{i}"))

            i+=1
            append_to_training_progress_file("-"*60)
            append_to_training_progress_file("\n\n")

    append_to_training_progress_file(f"Total time taken: {datetime.now() - startTime}")

if __name__ == "__main__":
    main()
