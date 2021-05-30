import argparse
import time

import cv2

from coarse_stab import generate_stabilized_video, get_frame_warp, read_video, stabilize
from fine_stab import fine_stab
from utils import ffmpeg_video, mkdir_if_not_exist, plot_vertex_motion


def get_parser():
    parser = argparse.ArgumentParser(description='EE229')

    parser.add_argument('--input_video', default='./data/Regular/0.avi', type=str, help='input directory')
    parser.add_argument('--output_dir', default='/GPFS/data/haoningwu/EE229/output/', type=str)
    parser.add_argument('--patch_size', default=16, type=int, help='block of size in patch')
    parser.add_argument('--propagation_radius', default=300, type=int, help='motion propogation radius')
    parser.add_argument('--border', default=20, type=int, help='')

    return parser


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    start_time = time.time()

    input_video = args.input_video
    output_dir = args.output_dir
    border = args.border
    patch_size = args.patch_size
    propagation_radius = args.propagation_radius
    x_motion_vector_path = output_dir + 'x_motion_vector_path/'
    new_x_motion_vector_path = output_dir + 'new_x_motion_vector_path/'
    motion_save_path = output_dir + 'path/'
    fine_stab_path = output_dir + 'fine_stab/'
    
    mkdir_if_not_exist(output_dir)
    mkdir_if_not_exist(x_motion_vector_path)
    mkdir_if_not_exist(new_x_motion_vector_path)
    mkdir_if_not_exist(motion_save_path)
    mkdir_if_not_exist(fine_stab_path)

    video = cv2.VideoCapture(input_video)
    # propogate motion vectors and generate vertex motion paths
    print("read video...")
    x_motion_patches, y_motion_patches, x_paths, y_paths = read_video(video, patch_size)

    # stabilize the vertex profiles
    print("stabilize...")
    opt_x_paths, opt_y_paths = stabilize(x_paths, y_paths)

    # visualize optimized paths
    print("plot vertex motion...")
    plot_vertex_motion(opt_x_paths, opt_y_paths, motion_save_path)

    # get updated mesh warps
    print("get frame warp...")
    x_motion_patches, y_motion_patches, new_x_motion_patches, new_y_motion_patches = get_frame_warp(x_motion_patches, y_motion_patches, x_paths, y_paths, opt_x_paths, opt_y_paths)

    # apply updated mesh warps & save the result
    print("generate stabilized video...")
    generate_stabilized_video(video, x_motion_patches, y_motion_patches, new_x_motion_patches, new_y_motion_patches, x_motion_vector_path, new_x_motion_vector_path, patch_size, border, output_dir)
    print('Time elapsed: ', str(time.time() - start_time))
    # ffmpeg_video(output_dir)
    
    # fine_stab_video = 'coarse_stab.avi'
    # fine_stab(fine_stab_video, fine_stab_path)
