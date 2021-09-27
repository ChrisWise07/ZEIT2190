#!/bin/bash
cp -ru './rrap' '/mnt/c/Users/Chris Wise/OneDrive/Documents/Uni/UNSW/2021/Sem 2/ZEIT2190/Code'
rm ./rrap/data/training_progress_file.txt
rm -rfd ./rrap/data/attack_training_data
mkdir ./rrap/data/attack_training_data
python3 -W ignore ./rrap/rrap_main.py