import glob
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def mkdir_if_not_exist(input_dir):
    flag = os.path.exists(input_dir)
    if not flag:
        os.makedirs(input_dir)
        print("New Directory:", input_dir)


def l2_dst(a, b):
    a, b = a.flatten(), b.flatten()
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def ffmpeg_video(img_dir, output_dir = './', video_name = 'coarse_stab.avi'):
    cmd = "ffmpeg -r 25 -i " + img_dir + "/%5d.png -pix_fmt yuv420p -b 20M " + output_dir + video_name
    os.system(cmd)


def init_dict(cols, rows):
    dict = {}
    for i in range(rows):
        for j in range(cols):
            dict[i, j] = [[0]]
    return dict


def gaussian_kernel(t, r, window_size):
    if np.abs(r-t) > window_size:
        return 0
    else:
        return np.exp((-9 * (r-t)**2) / window_size**2)


def gaussian_window(time_window_size, spatial_window_size):
    window = np.zeros((time_window_size, time_window_size))

    for i in range(window.shape[0]):
        for j in range(-spatial_window_size//2, spatial_window_size//2 + 1):
            if i+j < 0 or i+j >= window.shape[1] or j == 0:
                continue
            window[i, i+j] = gaussian_kernel(i, i+j, spatial_window_size)

    return window


def timer(func):
    def func_wrapper(*args, **kwargs):

        time_start = time.time()
        result = func(*args, **kwargs)
        time_end = time.time()
        time_spend = time_end - time_start
        print('{0} cost time {1} s'.format(func.__name__, time_spend))
        return result
    return func_wrapper


def plot_vertex_motion(x_paths, opt_x_paths, save_path):
    """
    Input:
    x_paths: original patch vertex motion
    opt_x_paths: optimized patch vertex motion
    """
    # plot some vertex paths
    for i in range(x_paths.shape[0]):
        for j in range(0, x_paths.shape[1], 10):
            plt.plot(x_paths[i, j, :])
            plt.plot(opt_x_paths[i, j, :])
            plt.savefig(save_path + str(i) + '_' + str(j) + '.png')
            plt.clf()


def save_motion_vectors(x_motion_patch, y_motion_patch, PATCH_SIZE, x_motion_vector_path, frame_num, frame, r=5):
    for i in range(x_motion_patch.shape[0]):
        for j in range(x_motion_patch.shape[1]):
            x, y = j * PATCH_SIZE, i * PATCH_SIZE
            theta = np.arctan2(y_motion_patch[i, j], x_motion_patch[i, j])
            cv2.line(frame, (x, y), (int(x + r * np.cos(theta)),int(y + r * np.sin(theta))), color=(0, 0, 255), thickness=1)

    cv2.imwrite(x_motion_vector_path + str(frame_num).zfill(5)+'.png', frame)

def batch_resize(input_dir, output_dir, width=640, height=360):
    
    def convert_img_name(image, output_dir, width, height):
        img = Image.open(image)
        try:
            new_img = img.resize((width,height), Image.BICUBIC)   
            new_img.save(os.path.join(output_dir, os.path.basename(image)))
        except Exception as e:
            print(e)

    for image in glob.glob(input_dir + "/*.png"):
        convert_img_name(image, output_dir, width, height)
