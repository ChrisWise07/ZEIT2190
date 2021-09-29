#!/bin/bash
rm ./data/training_progress_file.txt
rm -rfd ./data/attack_training_data
mkdir ./data/attack_training_data
python3 -W ignore ./rrap_main.py