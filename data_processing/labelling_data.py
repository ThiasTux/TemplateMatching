import glob
import os
from os.path import join
from subprocess import call
from xml.dom import minidom
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement

import matlab.engine
import numpy as np

DOWNSAMPLING_RATE = 1


def create_merge_file(input_path, user):
    input_path = join(input_path, "raw", user)
    files = list()
    files.append([file for file in glob.glob(input_path + "/torso*.txt") if os.stat(file).st_size != 0][0])
    files.append([file for file in glob.glob(input_path + "/uarm*.txt") if os.stat(file).st_size != 0][0])
    files.append([file for file in glob.glob(input_path + "/larm*.txt") if os.stat(file).st_size != 0][0])
    files.append([file for file in glob.glob(input_path + "/hand*.txt") if os.stat(file).st_size != 0][0])
    data = list()
    for f in files:
        data.append(np.loadtxt(f))
    print("Data loaded!")
    eng = matlab.engine.start_matlab()
    torso_data = data[0][:100, 1:]
    uarm_data = data[1][:100, 1:]
    larm_data = data[2][:100, 1:]
    hand_data = data[3][:100, 1:]
    output_data = eng.dtcMerge(matlab.double([5]), matlab.logical([False]),
                               matlab.double(torso_data.tolist()), matlab.int64([2]), matlab.double(uarm_data.tolist()),
                               matlab.int64([2]),
                               matlab.double(larm_data.tolist()), matlab.int64([2]), matlab.double(hand_data.tolist()),
                               matlab.int64([2]),
                               nargout=1)
    eng.quit()
    output_data = np.array(output_data)
    notnan_data = output_data[~np.isnan(output_data).any(axis=1)]
    np.savetxt(join(input_path.replace("raw", "processed"), "merged_data_cleaned.txt"), notnan_data,
               fmt="%.4f " * notnan_data.shape[1])
    print("Done: {}".format(output_data.shape))


def process_data_3dmodel(input_path):
    ## Run dtcMerge from MATLAB first

    merged_data = np.loadtxt(join(input_path, "raw", "merged_data.txt"))

    print(merged_data.shape)

    notnan_data = merged_data[~np.isnan(merged_data).any(axis=1)]

    np.savetxt(join(input_path, "raw", "merged_data_cleaned.txt"), notnan_data, fmt="%.4f " * notnan_data.shape[1])


def prepare_annotation(input_path):
    # merge_encode_videos(input_path)
    downsample_extract_annotation_data(input_path)
    create_labelling_project_files(input_path)


def merge_encode_videos(input_path):
    devnull = open(os.devnull, 'w')
    raw_folder = "raw"
    camera_folders = ["c_cam", "rx_cam", "sx_cam"]
    for cam_folder in camera_folders:
        files_path = join(input_path, raw_folder, cam_folder)
        files = [file for file in glob.glob(files_path + "/*.mp4") if os.stat(file).st_size != 0]
        files = sorted(files)
        with open(join(files_path, "video_list.txt"), "w") as video_list_file:
            for f in files:
                video_list_file.write("file \'{}\'\n".format(f))
        ffmpeg_command1 = "ffmpeg -f concat -safe 0 -i {} -c:v h264_nvenc -pix_fmt yuv420p -r 30 -an {}".format(
            join(files_path, "video_list.txt"), join(files_path, "tmp_output.mp4"))
        call(ffmpeg_command1, shell=True, stdout=devnull)
        ffmpeg_command2 = "ffmpeg -i {} -vcodec libxvid -vtag XVID -pix_fmt yuv420p -an {}".format(
            join(files_path, "tmp_output.mp4"), join(files_path, "output.avi"))
        call(ffmpeg_command2, shell=True, stdout=devnull)
    print("Video encoding ended!")


def downsample_extract_annotation_data(input_path):
    users = ["user1", "user2", "user3", "user4"]
    files = list()
    for user in users:
        files.append(
            [file for file in glob.glob(join(input_path, "raw", user) + "/hand*.txt") if os.stat(file).st_size != 0][0])
    files = sorted(files)
    for i, file in enumerate(files):
        data = np.loadtxt(file)
        data_seconds = data[:, 1] // 1000
        data_microseconds = (data[:, 1] % 1000) * 1000
        freq = 500
        cut_off = 0
        # data_axes = fd.butter_lowpass_filter(data[:, 1], cut_off, freq)[::DOWNSAMPLING_RATE]
        data_axes = data[:, 3]
        new_data = np.empty((len(data_axes), 3))
        new_data[:, 0] = data_seconds[::DOWNSAMPLING_RATE]
        new_data[:, 1] = data_microseconds[::DOWNSAMPLING_RATE]
        new_data[:, 2] = data_axes
        output_file = input_path + "/project/user{}.txt".format(i + 1)
        # output_file = file.replace("raw", "project").replace("user{}".format(i + 1), "")
        np.savetxt(output_file, new_data, fmt="%d")


def create_labelling_project_files(input_path):
    xml_project = Element('Project')

    xml_video_config = SubElement(xml_project, 'VideoConfig')
    xml_master_video = SubElement(xml_video_config, 'MasterVideo')
    master_video_path = join(input_path, "raw", "c_cam", "output.avi")
    xml_master_video.text = master_video_path

    xml_additional_videos = SubElement(xml_project, 'AdditionalVideos')
    xml_sx_add_video = SubElement(xml_additional_videos, 'Video')
    xml_sx_video_path = SubElement(xml_sx_add_video, 'FilePath')
    sx_video_path = join(input_path, "raw", "sx_cam", "output.avi")
    xml_sx_video_path.text = sx_video_path

    xml_rx_add_video = SubElement(xml_additional_videos, 'Video')
    xml_rx_video_path = SubElement(xml_rx_add_video, 'FilePath')
    rx_video_path = join(input_path, "raw", "rx_cam", "output.avi")
    xml_rx_video_path.text = rx_video_path

    xml_data_config = SubElement(xml_project, 'DataConfig')
    data_paths = [file for file in glob.glob(input_path + "project/*.txt") if os.stat(file).st_size != 0]
    data_paths = sorted(data_paths)
    data_parser_path = join(input_path, "acc_long.properties")
    for data in data_paths:
        xml_data_file = SubElement(xml_data_config, 'DataFile')
        xml_filepath = SubElement(xml_data_file, 'FilePath')
        xml_filepath.text = data
        xml_parser_class = SubElement(xml_data_file, 'ParserClass')
        xml_parser_class.text = "io.ColumnFileParser"
        xml_parser_config = SubElement(xml_data_file, 'ParserConfigFile')
        xml_parser_config.text = data_parser_path
        # if acc_offset != 0:
        #     xml_offset = SubElement(xml_data_file, 'Offset')
        #     xml_offset.text = acc_offset

    xml_label_config = SubElement(xml_project, 'LabelTrackConfig')

    xml_label_track = SubElement(xml_label_config, 'LabelTrack')
    xml_label_file = SubElement(xml_label_track, 'LabelTrackFile')
    labels_path = join(input_path, "project", "project_track1.xml")
    xml_label_file.text = labels_path

    xml_label_track = SubElement(xml_label_config, 'LabelTrack')
    xml_label_file = SubElement(xml_label_track, 'LabelTrackFile')
    labels_path = join(input_path, "project", "project_track2.xml")
    xml_label_file.text = labels_path

    xml_label_track = SubElement(xml_label_config, 'LabelTrack')
    xml_label_file = SubElement(xml_label_track, 'LabelTrackFile')
    labels_path = join(input_path, "project", "project_track3.xml")
    xml_label_file.text = labels_path

    xml_label_track = SubElement(xml_label_config, 'LabelTrack')
    xml_label_file = SubElement(xml_label_track, 'LabelTrackFile')
    labels_path = join(input_path, "project", "project_track4.xml")
    xml_label_file.text = labels_path

    output_path = join(input_path, "project", "project.xml")

    with open(output_path, 'w') as file:
        file.write(prettify(xml_project))


def prettify(elem):
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def create_processed_data(input_path):
    users = ["user2", "user3", "user4"]
    for user in users:
        create_merge_file(input_path, user)
