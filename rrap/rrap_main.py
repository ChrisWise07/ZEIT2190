import numpy as np
import os
import sys
from PIL import Image

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from rrap_custom_dpatch_robust import RobustDPatch
from rrap_utils import *
from rrap_image_for_patch import Image_For_Patch
from rrap_constants import *
from rrap_data_plotter import Data_Plotter

def generate_adversarial_patch(attack, image, step_num):
    image_name = image.name
    data_plotter = Data_Plotter()
    image_copies = np.vstack([image.cropped_image_as_np_array]*1)
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
            attack.decay_perceptibility_learning_rate()

    data_plotter.plot_training_data(image_name)

def generate_rrap_for_image(image_name):
    image_name, file_type = image_name.split(".")

    image = Image_For_Patch(name = image_name, object_detector=FRCNN, file_type=file_type)
    training_data_path = (f"{TRAINING_DATA_DIRECTORY}training_data_for_{image_name}.txt")

    attack = RobustDPatch(estimator=FRCNN, max_iter=1, batch_size=1, verbose=False, rotation_weights=(1,0,0,0), 
                        brightness_range= (1.0,1.0), decay_rate = 0.95, detection_momentum = 0.9, perceptibility_momentum = 0.9,
                        image_to_patch = image, training_data_path = training_data_path, perceptibility_learning_rate = 100.0, 
                        detection_learning_rate = 5.0, training_data = get_previous_training_data(training_data_path))

    generate_adversarial_patch(attack, image, step_num = 1)
    adv_patch = attack.get_patch()
    Image.fromarray(np.uint8(adv_patch)).save(f"{PATCHES_DIRECTORY}patch_for_{image_name}.{file_type}")

    cropped_adv_image = Image.fromarray(np.uint8(attack.apply_patch(x=image.cropped_image_as_np_array)[0]))
    image_adv = Image.open(f"{IMAGES_DIRECTORY}{image_name}.{file_type}").copy()
    image_adv.paste(cropped_adv_image, box=(int(image.predictions_box[0][0]), int(image.predictions_box[0][1])))
    image_adv.save(f"{ADVERSARIAL_IMAGES_DIRECTORY}adv_{image_name}.{file_type}")
    image_adv_as_np_array = np.stack([np.asarray(image_adv)], axis=0).astype(np.float32)

    image.append_to_training_progress_file(f"\n\n--- Final predictions for {image_name} with adversarial patch ---")
    image.plot_predictions(object_detector = FRCNN, image = image_adv_as_np_array, path = f"{ADVERSARIAL_PREDICTIONS_DIRECTORY}adv_{image_name}.{file_type}")

def main():
    with os.scandir(IMAGES_DIRECTORY) as entries:
        [generate_rrap_for_image(entry.name) for entry in entries] 

if __name__ == "__main__":
    main()