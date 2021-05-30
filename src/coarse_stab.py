import cv2
import numpy as np

from propagation import propagate, vertex_motion_path, warp_frame
from optimizer import online_optimize_path
from utils import save_motion_vectors, timer


@timer
def read_video(video, patch_size):

    # parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=400, qualityLevel=0.01, minDistance=7, blockSize=7)

    # parameters for lucas kanade optical flow
    flow_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

    # Take first frame
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    _, prev_frame = video.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # motion patches in x-direction and y-direction
    x_motion_patches, y_motion_patches = [], []
    
    # path parameters
    x_paths = np.zeros((prev_frame.shape[0] // patch_size, prev_frame.shape[1] // patch_size, 1))
    y_paths = np.zeros((prev_frame.shape[0] // patch_size, prev_frame.shape[1] // patch_size, 1))

    for frame_num in range(1, frame_count):
        # processing frames
        flag, curr_frame = video.read()

        if not flag:
            break
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # find corners in it
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

        # calculate optical flow
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **flow_params)

        # Select good points
        curr_pts, prev_pts = curr_pts[status==1], prev_pts[status==1]

        # estimate motion mesh for old_frame
        x_motion_patch, y_motion_patch = propagate(prev_pts, curr_pts, curr_frame)

        try:
            x_motion_patches = np.concatenate((x_motion_patches, np.expand_dims(x_motion_patch, axis=2)), axis=2)
            y_motion_patches = np.concatenate((y_motion_patches, np.expand_dims(y_motion_patch, axis=2)), axis=2)
        except:
            x_motion_patches = np.expand_dims(x_motion_patch, axis=2)
            y_motion_patches = np.expand_dims(y_motion_patch, axis=2)

        # generate vertex profiles
        x_paths, y_paths = vertex_motion_path(x_paths, y_paths, x_motion_patch, y_motion_patch)

        # updates frames
        prev_frame = curr_frame.copy()
        prev_gray = curr_gray.copy()

    return [x_motion_patches, y_motion_patches, x_paths, y_paths]

@timer
def stabilize(x_paths, y_paths):
    """
    Input:
    x_paths: motion vector accumulation on patch vertices in x-direction
    y_paths: motion vector accumulation on patch vertices in y-direction
    
    Output:
    opt_x_paths, opt_y_paths: optimized paths in x-direction and y-direction
    """

    opt_x_paths = online_optimize_path(x_paths)
    opt_y_paths = online_optimize_path(y_paths)

    return [opt_x_paths, opt_y_paths]

@timer
def get_frame_warp(x_motion_patches, y_motion_patches, x_paths, y_paths, opt_x_paths, opt_y_paths):
    """
    Input
    x_motion_patches: motion vectors on patch vertices in x-direction
    y_motion_patches: motion vectors on patch vertices in y-direction
    x_paths: motion vector accumulation on patch vertices in x-direction
    y_paths: motion vector accumulation on patch vertices in y-direction    
    opt_x_paths: optimized motion vector accumulation in x-direction
    opt_y_paths: optimized motion vector accumulation in x-direction
    
    Output:
    Updated motion patches for each frame with which that needs to be warped
    """

    x_motion_patches = np.concatenate((x_motion_patches, np.expand_dims(x_motion_patches[:, :, -1], axis=2)), axis=2)
    y_motion_patches = np.concatenate((y_motion_patches, np.expand_dims(y_motion_patches[:, :, -1], axis=2)), axis=2)
    new_x_motion_patches = opt_x_paths - x_paths
    new_y_motion_patches = opt_y_paths - y_paths

    return x_motion_patches, y_motion_patches, new_x_motion_patches, new_y_motion_patches


@timer
def generate_stabilized_video(video, x_motion_patches, y_motion_patches, new_x_motion_patches, new_y_motion_patches, x_motion_vector_path, new_x_motion_vector_path, PATCH_SIZE, border, output_dir):
    """
    Input:
    video: cv2.VideoCapture object of the given video
    x_motion_patches: motion vectors on mesh vertices in x-direction
    y_motion_patches: motion vectors on mesh vertices in y-direction
    new_x_motion_patches: updated motion vectors on mesh vertices in x-direction to be warped with
    new_y_motion_patches: updated motion vectors on mesh vertices in y-direction to be warped with
    """

    # get video properties
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_num in range(frame_count):
        # reconstruct from frames
        flag, frame = video.read()
        if not flag:
            break
        x_motion_patch = x_motion_patches[:, :, frame_num]
        y_motion_patch = y_motion_patches[:, :, frame_num]
        new_x_motion_patch = new_x_motion_patches[:, :, frame_num]
        new_y_motion_patch = new_y_motion_patches[:, :, frame_num]

        # warping
        new_frame = warp_frame(frame, new_x_motion_patch, new_y_motion_patch)
        new_frame = new_frame[border:-border, border:-border, :]
        new_frame = cv2.resize(new_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(output_dir + str(frame_num).zfill(5) + '.png', new_frame)
        save_motion_vectors(x_motion_patch, y_motion_patch, PATCH_SIZE, x_motion_vector_path, frame_num, frame, r=5)
        save_motion_vectors(new_x_motion_patch, new_y_motion_patch, PATCH_SIZE, new_x_motion_vector_path, frame_num, new_frame, r=5)

    video.release()
