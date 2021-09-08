import numpy as np
import os
import sys

from art.estimators.object_detection import PyTorchFasterRCNN
from art.attacks.evasion import RobustDPatch

ROOT_DIRECTORY = "/mnt/c/Users/Chris Wise/Documents/Programming/ZEIT2190/rrap/"
sys.path.append(ROOT_DIRECTORY)
from rrap_utils import save_image, generate_predictions, plot_predictions
from rrap_image_for_patch import Image_For_Patch

DATA_DIRECTORY = ROOT_DIRECTORY + "data/"
IMAGES_DIRECTORY = DATA_DIRECTORY + "images/"
PATCHES_DIRECTORY = DATA_DIRECTORY + "patches_adv/"
ADVERSARIAL_PREDICTIONS_DIRECTORY = DATA_DIRECTORY + "predictions_adv/"
ADVERSARIAL_IMAGES_DIRECTORY =  DATA_DIRECTORY + "images_adv/"

#Create ART object detector
FRCNN = PyTorchFasterRCNN(
    clip_values=(0, 255), attack_losses=["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
)

def get_custom_patch_location(prediction_centre_points, patch_shape):
    #Here the coordinates are store (y,x) as somewhere in Roboust Dpatch they are treated as (y,x) 
    return ((int(prediction_centre_points[1] - (patch_shape[1]/2)),
             int(prediction_centre_points[0] - (patch_shape[0]/2))))   

def generate_adversarial_patch(attack, image):
    print("\n\n--- Generating adversarial patch for image {} ---".format(image.get_image_num()))
    image_copies = np.vstack([image.get_image_as_np_array()]*1)

    #training loop: actually number of iterations = j*max_iter
    for j in range(2):
        #train adv patch to trick object detector and not to be perceptible
        patch_adv = attack.generate(x=image_copies, og_image=image, y=None)
                                    
        print("\n--- Periodic prediction {}---".format(j))                         
        image_adv = attack.apply_patch(x=image.get_image_as_np_array())
        generate_predictions(object_detector=FRCNN, image=image_adv)  

    return patch_adv, image_adv

def main():

    # Create attack
    attack = RobustDPatch(estimator=FRCNN, max_iter=5, batch_size=1, verbose=False, 
                          rotation_weights=(1,1,1,1), brightness_range= (0.1,1.0))

    with os.scandir(IMAGES_DIRECTORY) as entries:
        i = 0

        for entry in entries:
            print("-"*60)

            #Create image object and, print and save predictions
            image = Image_For_Patch(entry.path, image_num = i, object_detector=FRCNN)
            
            # Customise patch location to centre of prediction box
            prediction_centre_points = image.get_centre_point_of_prediction_boxes()
            patch_shape = attack.get_patch_shape()
            attack.set_patch_location(get_custom_patch_location(prediction_centre_points, patch_shape))

            #create adversarial patch
            patch_adv, image_adv = generate_adversarial_patch(attack, image)

            #Save adv patch
            save_image(patch_adv, (PATCHES_DIRECTORY + "patch_{}".format(i)))
    
            #Print predictions for image with adv patch and save image with adv patch and prediction boxes
            print("\n\n--- Final predictions for image with adversarial patch {} ---".format(i))
            plot_predictions(object_detector = FRCNN, image = image_adv, 
                             path = (ADVERSARIAL_PREDICTIONS_DIRECTORY + "/image_adv_{}".format(i)))

            #Save image with adv patch but with no prediction boxes
            save_image(image_adv[0], (ADVERSARIAL_IMAGES_DIRECTORY + "image_adv_{}".format(i)))

            i+=1
            print("-"*60)
            print("\n\n")

if __name__ == "__main__":
    print("\n")
    main()