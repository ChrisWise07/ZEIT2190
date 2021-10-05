import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import json
import os

from json import JSONEncoder
from torch import Tensor
from PIL import Image
from matplotlib.ticker import (MultipleLocator, AutoLocator, AutoMinorLocator)

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from differential_color_functions import rgb2lab_diff, ciede2000_diff
from rrap_constants import *

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
                return obj.tolist()
        if isinstance(obj, Tensor):
                return obj.detach().numpy().tolist()
        return JSONEncoder.default(self, obj)

def save_image(img, path):
        plt.axis("off")
        plt.imshow(img.astype(np.uint64), interpolation="nearest")
        plt.savefig(path, bbox_inches="tight", pad_inches = 0)
        plt.close()

def get_device():
        if not torch.cuda.is_available():
                return torch.device("cpu")
        else:
                cuda_idx = torch.cuda.current_device()
                return torch.device(f"cuda:{cuda_idx}")

def get_image_as_tensor(image_path, image_size, need_grad):
        image_tensor = TRANSFORM(Image.open(image_path).resize(image_size))
        return image_tensor.clone().detach().requires_grad_(need_grad)
        
def get_rgb_diff(image_tensor):
        image_tensor = torch.stack([image_tensor], dim=0)
        return rgb2lab_diff(image_tensor,get_device()) 

def calculate_perceptibility_gradients_between_images(og_image, loss_tracker):
        patched_image_tensor = get_image_as_tensor(PATCHED_IMAGE_PATH, image_size=og_image.get_image_size(), need_grad=True)
        patched_image_rgb_diff = get_rgb_diff(patched_image_tensor)
        d_map=ciede2000_diff(og_image.get_image_rbg_diff(), patched_image_rgb_diff, get_device()).unsqueeze(1)
        perceptibility_dis=torch.norm(d_map.view(1,-1),dim=1)
        perceptibility_loss = perceptibility_dis.sum()
        loss_tracker.update_perceptibility_loss(perceptibility_loss.item())
        perceptibility_loss.backward(retain_graph=True)
        perceptibility_grad = patched_image_tensor.grad.cpu().numpy().copy()
        return perceptibility_grad

def get_perceptibility_gradients_of_patch(og_image, patched_image, patch_shape, patch_location, loss_tracker):
        save_image(patched_image, PATCHED_IMAGE_PATH)
        patch_perceptibility_gradients = calculate_perceptibility_gradients_between_images(og_image, loss_tracker)
        patch_perceptibility_gradients = np.reshape(patch_perceptibility_gradients, patched_image.shape)
        return patch_perceptibility_gradients[patch_location[0]:patch_location[0]+patch_shape[0],
                                              patch_location[1]:patch_location[1]+patch_shape[1],
                                              :]
def file_handler(path, mode, func):
        try:
                f = open(path, mode)
                value = func(f)
                f.close()
                return value
        except FileNotFoundError:
                return 0

def get_previous_training_data(training_data_path):
        return file_handler(path = training_data_path, mode = "r", func = lambda f: json.load(f))

def get_previous_steps(training_data_path):
        return file_handler(path = training_data_path, mode = "r", func = lambda f: int(json.load(f)["step_number"]))

def record_attack_training_data(attack, step_number):
        training_data = {}
        training_data["step_number"] = str(step_number)
        training_data["detection_learning_rate"] = attack.get_detection_learning_rate()
        training_data["perceptibility_learning_rate"] = attack.get_perceptibility_learning_rate()
        loss_tracker = attack.get_loss_tracker()
        training_data["loss_data"] = {"perceptibility_loss": loss_tracker.rolling_perceptibility_loss, 
                                      "detection_loss": loss_tracker.rolling_detection_loss}
        training_data["patch_np_array"] = attack.get_patch()
        training_data["old_patch_detection_update"] = np.array(attack.get_old_patch_detection_update())
        training_data["old_patch_perceptibility_update"] = np.array(attack.get_old_patch_perceptibility_update())
        file_handler(path = attack.get_training_data_path(), mode = "w", func = lambda f: json.dump(training_data, f, cls=NumpyArrayEncoder))


def plot_data(rolling_loss_history, current_loss_history, lr_history, image_name, loss_type):
        # create figure and axis objects with subplots()
        fig,ax = plt.subplots()

        # set x-axis label
        ax.set_xlabel("Steps",fontsize=10)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(10))

        # make a plot
        ax.plot(rolling_loss_history, '-r', label=f'Currrent {loss_type} Loss')
        ax.plot(current_loss_history, '-g', label=f'Rolling {loss_type} Loss')
        ax.legend(loc='upper left')
        # set y-axis label
        ax.set_ylabel(f"{loss_type} Loss", fontsize=10)
        ax.yaxis.set_major_locator(AutoLocator())
        #ax.yaxis.set_major_formatter(f'{x:.0f}"')
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        # twin object for two different y-axis on the sample plot
        ax2=ax.twinx()
        # make a plot with different y-axis using second axis object
        ax2.plot(lr_history, '-b', label=f'{loss_type} Lr')
        ax2.legend(loc='upper right')
        # set y-axis label
        ax2.set_ylabel(f"{loss_type} Lr", fontsize=10)
        ax.yaxis.set_major_locator(AutoLocator())
        #ax.yaxis.set_major_formatter('{x:.0f}')
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        # save the plot as a file
        plt.title(f"{loss_type} Data Over Step Numbers", fontsize=12)
        plt.savefig(f"{PLOTS_DIRECTORY}{loss_type}_loss_data_{image_name}.png", bbox_inches='tight')
        plt.close()