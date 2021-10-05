import numpy as np
import os
import sys
import threading
from multiprocessing import Pool
import multiprocessing as mp

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from rrap_custom_dpatch_robust import RobustDPatch
from rrap_custom_pytorch_faster_rcnn import PyTorchFasterRCNN
from rrap_utils import *
from rrap_image_for_patch import Image_For_Patch
from rrap_constants import *
from rrap_data_plotter import Data_Plotter

def generate_adversarial_patch(attack, image, step_num):
    image_name = image.name
    data_plotter = Data_Plotter()
    image_copies = np.vstack([image.image_as_np_array]*1)
    previous_num_steps = get_previous_steps(attack.get_training_data_path())

    image.append_to_training_progress_file(f"\n\n--- Generating adversarial patch for {image_name} ---")

    for step in range(previous_num_steps, previous_num_steps + step_num):
        image.append_to_training_progress_file(f"\n\n--- Step Number: {step} ---")
        
        #train adv patch to trick object detector and not to be perceptibile
        attack.generate(x=image_copies, print_nth_num=1, y=None)                    

        #Save adv patch and training data every step
        record_attack_training_data(attack, step + 1)
        data_plotter.save_training_data(attack.get_loss_tracker(), attack.get_perceptibility_learning_rate(), attack.get_detection_learning_rate())   

        
        #Decay learning rate every 5 steps
        if ((step + 1) % 1 == 0):
            attack.decay_detection_learning_rate()
            #attack.decay_perceptibility_learning_rate()

    data_plotter.plot_training_data(image_name)

def generate_rrap_for_image(image_name):
    image_name, file_type = image_name.split(".")

    #Create ART object detector for each process
    frcnn = PyTorchFasterRCNN(
        clip_values=(0, 255), attack_losses=["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
    )

    #Create image object and, print and save predictions
    image = Image_For_Patch(name = image_name, object_detector=frcnn, file_type=file_type)
    training_data_path = (f"{TRAINING_DATA_DIRECTORY}training_data_for_{image_name}.txt")

    # Create attack
    attack = RobustDPatch(estimator=frcnn, max_iter=1, batch_size=1, verbose=False, rotation_weights=(1,0,0,0), 
                        brightness_range= (1.0,1.0), decay_rate = 0.95, detection_momentum = 0.9, perceptibility_momentum = 0.9,
                        image_to_patch = image, training_data_path = training_data_path, perceptibility_learning_rate = 10.0, 
                        detection_learning_rate = 5.0, training_data = get_previous_training_data(training_data_path))

    #create adversarial patch and apply it to orignal image
    generate_adversarial_patch(attack, image, step_num = 1)
    adv_patch = attack.get_patch()
    image_adv = attack.apply_patch(x=image.image_as_np_array)

    #Print predictions for image with adv patch and save image with adv patch and prediction boxes
    image.append_to_training_progress_file(f"\n\n--- Final predictions for {image_name} with adversarial patch ---")
    image.plot_predictions(object_detector = frcnn, image = image_adv, path = f"{ADVERSARIAL_PREDICTIONS_DIRECTORY}adv_{image_name}")

    #Save adv pathc and, image with adv patch but with no prediction boxes
    save_image(image_adv[0], (f"{ADVERSARIAL_IMAGES_DIRECTORY}adv_{image_name}"))
    save_image(adv_patch, (f"{PATCHES_DIRECTORY}patch_for_{image_name}"))

def main():
    """
    with os.scandir(IMAGES_DIRECTORY) as entries:
        [threading.Thread(target = generate_rrap_for_image, args=(entry.name,)).start() for entry in entries]
    """

    """
    with os.scandir(IMAGES_DIRECTORY) as entries:
        [generate_rrap_for_image(entry.name) for entry in entries] 

    """
    mp.set_start_method('spawn')

    with os.scandir(IMAGES_DIRECTORY) as entries:
        with Pool(mp.cpu_count()) as p:
            p.map(generate_rrap_for_image, [entry.name for entry in entries])

if __name__ == "__main__":
    main()