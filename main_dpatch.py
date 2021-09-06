import cv2
import numpy as np
import os
import sys

from art.estimators.object_detection import PyTorchFasterRCNN
from art.attacks.evasion import RobustDPatch

sys.path.append("/mnt/c/Users/Chris Wise/Documents/Programming/ZEIT2190/dpatch/")
from dpatch_utils import save_image, generate_predictions, plot_predictions, get_rgb_diff

def main(root_dir):

    #Create ART object detector
    frcnn = PyTorchFasterRCNN(
        clip_values=(0, 255), attack_losses=["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
    )

    # Create attack
    attack = RobustDPatch(estimator=frcnn, max_iter=4, sample_size=1, learning_rate=5.0, batch_size=1, 
                          verbose=False, rotation_weights=(1,1,1,1), brightness_range= (0.1,1.0))

    with os.scandir(root_dir + "images") as entries:
        i = 0

        for entry in entries:
            print("-"*60)

            #Read image
            image = cv2.imread(entry.path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.stack([image], axis=0).astype(np.float32)
            image_copies = np.vstack([image]*4).astype(np.float32)

            image_size = (image.shape[2],image.shape[1])
            image_rbg_diff = get_rgb_diff(file_location_name = (root_dir + "images/" + entry.name), image_size = image_size)

            #Print predictions for og image and save og image with prediction boxes
            print("\n--- Initial Predictions for image {} ---".format(i))
            predictions_boxes = plot_predictions(object_detector = frcnn,
                                                  image = image, 
                                                  file_location_name = (root_dir + "predictions/image_{}".format(i)))

            #Get centre of image and patch
            top_left_bbox = predictions_boxes[0]
            bottom_right_bbox = predictions_boxes[1]
            centre_point = (int((top_left_bbox[0] + bottom_right_bbox[0]) / 2),
                            int((top_left_bbox[1] +  bottom_right_bbox[1]) / 2)) 
            patch_dimensions = (40,40)
            #Here the coordinates are store (y,x) as somewhere in Roboust Dpatch they are treated as (y,x) 
            top_left_coordinates = ((int(centre_point[1] - (patch_dimensions[1]/2)),
                                  int(centre_point[0] - (patch_dimensions[0]/2))))
            # Customise patch location
            attack.set_patch_location(top_left_coordinates)

            print("\n\n--- Training adversarial patch for image {} ---".format(i))

            #training loop: actually number of iterations = j*max_iter
            for j in range(4):
                #train adv patch to trick object detector and not to be perceptible
                patch_adv = attack.generate(x=image_copies,
                                            root_dir=root_dir, 
                                            og_image_rbg_diff = image_rbg_diff, 
                                            og_image = image,
                                            og_image_size = image_size,
                                            y=None)
                                            
                print("\n--- Periodic prediction {}---".format(j))                         
                image_adv = attack.apply_patch(x=image)
                generate_predictions(object_detector=frcnn, image=image_adv)

            #Save adv patch
            save_image(patch_adv, (root_dir + "/patches_adv/patch_{}".format(i)))
    
            #Print predictions for image with adv patch and save image with adv patch and prediction boxes
            print("\n\n--- Final predictions for image with adversarial patch {} ---".format(i))
            plot_predictions(object_detector = frcnn, image = image_adv, 
                                       file_location_name = (root_dir + "predictions_adv/image_adv_{}".format(i)))

            #Save image with adv patch but with no prediction boxes
            save_image(image_adv[0], (root_dir +"images_adv/image_adv_{}".format(i)))

            i+=1
            print("-"*60)
            print("\n\n")

if __name__ == "__main__":
    print("\n")
    main(root_dir ="./data/")
