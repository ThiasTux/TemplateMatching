import glob
import os
from os.path import join

import numpy as np

DOWNSAMPLING_RATE = 20


def create_labelling_project_files(input_path):
    project_output_path = input_path.replace("raw", "project")

    ## Downsample gyroscope data for labelling tool software
    torso_file = [file for file in glob.glob(input_path + "/torso*.LOG") if os.stat(file).st_size != 0][0]
    torso_data = np.loadtxt(torso_file)
    torso_time = torso_data[::DOWNSAMPLING_RATE, 1]
    torso_seconds = torso_time // 1000
    torso_microseconds = (torso_time % 1000) * 1000
    torso_axes = torso_data[::DOWNSAMPLING_RATE, 6]
    torso_output_file = join(project_output_path, "torso_{}.txt".format(int(500 / DOWNSAMPLING_RATE)))
    np.savetxt(torso_output_file,
               np.stack((torso_seconds, torso_microseconds, torso_axes), axis=1), fmt="%d")
    uarm_file = [file for file in glob.glob(input_path + "/uarm*.LOG") if os.stat(file).st_size != 0][0]
    uarm_data = np.loadtxt(uarm_file)
    uarm_time = uarm_data[::DOWNSAMPLING_RATE, 1]
    uarm_seconds = uarm_time // 1000
    uarm_microseconds = (uarm_time % 1000) * 1000
    uarm_axes = uarm_data[::DOWNSAMPLING_RATE, 6]
    uarm_output_file = join(project_output_path, "uarm_{}.txt".format(int(500 / DOWNSAMPLING_RATE)))
    np.savetxt(uarm_output_file,
               np.stack((uarm_seconds, uarm_microseconds, uarm_axes), axis=1), fmt="%d")
    larm_file = [file for file in glob.glob(input_path + "/larm*.LOG") if os.stat(file).st_size != 0][0]
    larm_data = np.loadtxt(larm_file)
    larm_time = larm_data[::DOWNSAMPLING_RATE, 1]
    larm_seconds = larm_time // 1000
    larm_microseconds = (larm_time % 1000) * 1000
    larm_axes = larm_data[::DOWNSAMPLING_RATE, 6]
    larm_output_file = join(project_output_path, "larm_{}.txt".format(int(500 / DOWNSAMPLING_RATE)))
    np.savetxt(larm_output_file,
               np.stack((larm_seconds, larm_microseconds, larm_axes), axis=1), fmt="%d")
    hand_file = [file for file in glob.glob(input_path + "/hand*.LOG") if os.stat(file).st_size != 0][0]
    hand_data = np.loadtxt(hand_file)
    hand_time = hand_data[::DOWNSAMPLING_RATE, 1]
    hand_seconds = hand_time // 1000
    hand_microseconds = (hand_time % 1000) * 1000
    hand_axes = hand_data[::DOWNSAMPLING_RATE, 6]
    hand_output_file = join(project_output_path, "hand_{}.txt".format(int(500 / DOWNSAMPLING_RATE)))
    np.savetxt(hand_output_file,
               np.stack((hand_seconds, hand_microseconds, hand_axes), axis=1), fmt="%d")

    ## Process video


def create_merge_file(input_path):
    torso_file = [file for file in glob.glob(input_path + "/torso*.LOG") if os.stat(file).st_size != 0][0]
    uarm_file = [file for file in glob.glob(input_path + "/uarm*.LOG") if os.stat(file).st_size != 0][0]
    larm_file = [file for file in glob.glob(input_path + "/larm*.LOG") if os.stat(file).st_size != 0][0]
    hand_file = [file for file in glob.glob(input_path + "/hand*.LOG") if os.stat(file).st_size != 0][0]
