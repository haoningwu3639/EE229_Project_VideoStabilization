import cv2
import numpy as np
from scipy.signal import medfilt

from utils import init_dict, l2_dst

def keypoint_transform(H, keypoint):
    """
    Input:
    H: homography matrix of dimension (3*3)
    keypoint: the (x, y) point to be transformed

    Output:
    keypoint_trans: Transformed point keypoint_trans = H * (keypoint, 1)
    """

    keypoint = np.append(keypoint, 1)
    a, b, c = np.dot(H, keypoint)

    keypoint_trans = np.array([[a/c, b/c]]).flatten()

    return keypoint_trans


def propagate(input_points, output_points, input_frame, PATCH_SIZE=16, PROP_R=300):
    """
    Input:
    intput_points: points in input_frame which are matched feature points with output_frame
    output_points: points in input_frame which are matched feature points with intput_frame
    input_frame
    H: the homography between input and output points

    Output: 
    x_motion_patch, y_motion_patch: Motion patch in x-direction and y-direction for input_frame
    """

    cols, rows = input_frame.shape[1] // PATCH_SIZE, input_frame.shape[0] // PATCH_SIZE

    x_motion = init_dict(cols, rows)
    y_motion = init_dict(cols, rows)
    temp_x_motion = init_dict(cols, rows)
    temp_y_motion = init_dict(cols, rows)

    # pre-warping with global homography
    H, _ = np.array(cv2.findHomography(input_points, output_points, cv2.RANSAC))
    for i in range(rows):
        for j in range(cols):
            point = np.array([[PATCH_SIZE * j, PATCH_SIZE * i]])

            point_trans = keypoint_transform(H, point)

            x_motion[i, j] = point.flatten()[0] - point_trans[0]
            y_motion[i, j] = point.flatten()[1] - point_trans[1]

    # distribute feature motion vectors
    for i in range(rows):
        for j in range(cols):
            vertex = np.array([[PATCH_SIZE * j, PATCH_SIZE * i]])
            for in_point, out_point in zip(input_points, output_points):
                # velocity = point - feature point in current frame

                distance = l2_dst(in_point, vertex)
                if distance < PROP_R:
                    point_trans = keypoint_transform(H, in_point)

                    temp_x_motion[i, j] = [out_point[0] - point_trans[0]]
                    temp_y_motion[i, j] = [out_point[1] - point_trans[1]]

    # Apply one Median Filter on obtained motion for each vertex
    x_motion_patch = np.zeros((rows, cols), dtype=float)
    y_motion_patch = np.zeros((rows, cols), dtype=float)

    for key in x_motion.keys():

        temp_x_motion[key].sort()
        temp_y_motion[key].sort()
        x_motion_patch[key] = x_motion[key] + temp_x_motion[key][len(temp_x_motion[key]) // 2]
        y_motion_patch[key] = y_motion[key] + temp_y_motion[key][len(temp_y_motion[key]) // 2]

    # Apply the other Median Filter over the motion patch for outliers
    x_motion_patch = medfilt(x_motion_patch, kernel_size=[3, 3])
    y_motion_patch = medfilt(y_motion_patch, kernel_size=[3, 3])

    return x_motion_patch, y_motion_patch


def vertex_motion_path(x_path, y_path, x_motion_patch, y_motion_patch):
    """
    Input:
    x_path: motion path along x_direction
    y_path: motion path along y_direction
    x_motion_patch: obtained motion patch along x_direction
    y_motion_patch: obtained motion patch along y_direction
    
    Output:
    x_paths, y_paths: Updated x_paths, y_paths with new x_motion_patch, y_motion_patch added to the last x_paths, y_paths
    """

    x_path_new = x_path[:, :, -1] + x_motion_patch
    y_path_new = y_path[:, :, -1] + y_motion_patch
    x_paths = np.concatenate((x_path, np.expand_dims(x_path_new, axis=2)), axis=2)
    y_paths = np.concatenate((y_path, np.expand_dims(y_path_new, axis=2)), axis=2)

    return x_paths, y_paths


def warp_frame(frame, x_motion_patch, y_motion_patch, PATCH_SIZE=16):
    """
    Input:
    frame is the current frame
    x_motion_patch: the motion_patch to be warped on frame along x-direction
    y_motion_patch: the motion patch to be warped on frame along y-direction
    
    Output:
    new_frame: a warped frame according to given motion patches x_motion_patch, y_motion_patch
    """

    map_x = np.zeros((frame.shape[0], frame.shape[1]), np.float32)
    map_y = np.zeros((frame.shape[0], frame.shape[1]), np.float32)

    for i in range(x_motion_patch.shape[0] - 1):
        for j in range(x_motion_patch.shape[1] - 1):

            x, y = int(j * PATCH_SIZE), int(i * PATCH_SIZE)
            x_next, y_next = int((j+1) * PATCH_SIZE), int((i+1) * PATCH_SIZE)

            src = np.array(
                [[x, y], [x, y_next], [x_next, y], [x_next, y_next]]
                )

            dst = np.array(
                [[x + x_motion_patch[i, j], y + y_motion_patch[i, j]],
                 [x + x_motion_patch[i+1, j], y_next + y_motion_patch[i+1, j]],
                 [x_next + x_motion_patch[i, j+1], y + y_motion_patch[i, j+1]],
                 [x_next + x_motion_patch[i+1, j+1], y_next + y_motion_patch[i+1, j+1]]]
                 )

            H, _ = cv2.findHomography(src, dst, cv2.RANSAC)

            for k in range(y, y_next):
                for l in range(x, x_next):

                    x_res, y_res, w_res = np.dot(H, np.append(np.array([[l, k]]), 1))
                    if w_res != 0:
                        x_res, y_res = x_res / (w_res*1.0), y_res / (w_res*1.0)
                    else:
                        x_res, y_res = l, k
                    map_x[k, l] = x_res
                    map_y[k, l] = y_res

    # repeat motion vectors for remaining frame in x-direction
    for j in range(PATCH_SIZE*x_motion_patch.shape[1], map_x.shape[1]):
        map_x[:, j] = map_x[:, PATCH_SIZE * x_motion_patch.shape[0] - 1]
        map_y[:, j] = map_y[:, PATCH_SIZE * x_motion_patch.shape[0] - 1]

    # repeat motion vectors for remaining frame in y-direction
    for i in range(PATCH_SIZE*x_motion_patch.shape[0], map_x.shape[0]):
        map_x[i, :] = map_x[PATCH_SIZE * x_motion_patch.shape[0] - 1, :]
        map_y[i, :] = map_y[PATCH_SIZE * x_motion_patch.shape[0] - 1, :]

    # deforms patch
    new_frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return new_frame
