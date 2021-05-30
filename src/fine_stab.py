
import cv2
import numpy as np
from superpoint import SuperPointWrapper

def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # Define filter
    f = np.ones(window_size) / window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # Convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Unpadding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed


def smooth(trajectory, radius=50):
    smooth_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(3):
        smooth_trajectory[:, i] = movingAverage(trajectory[:, i], radius)

    return smooth_trajectory


def alternative_pts(prev_gray, curr_gray):
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7)

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    idx = np.where(status == 1)[0]

    prev_pts, curr_pts = prev_pts[idx], curr_pts[idx]
    
    return prev_pts, curr_pts

def fixBorder(frame):
    s = frame.shape
    # Scale the image 4% without moving the center
    T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))

    return frame


def fine_stab(input_video, output_dir, weights_path = '../pretrained_model/superpoint_v1.pth', cuda = True):
    # Read input video
    video = cv2.VideoCapture(input_video)

    # Get frame count and fps
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get width and height of video stream
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Read the first frame
    _, prev = video.read()

    # Convert frame to grayscale
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Pre-define transformation-store array
    transforms = np.zeros((n_frames-1, 3), np.float32)
    
    SPNet = SuperPointWrapper(weights_path=weights_path, cuda=cuda)

    for i in range(n_frames - 2):
        # SuperPoint KeyPoints Detector

        prev_gray_float32 = prev_gray.astype('float32')
        
        prev_pts, _, _ = SPNet.run(prev_gray_float32)
        
        prev_pts = prev_pts.astype('float32').T
        prev_pts = np.array([prev_pts[:, :2]])
        prev_pts = np.transpose(prev_pts, (1, 0, 2))

        if prev_pts.shape[0] <= 10:
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7)

        # Read next frame
        flag, curr = video.read()
        if not flag:
            break

        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, winSize=(15, 15), maxLevel=2)

        # Filter only valid points
        idx = np.where(status == 1)[0]

        prev_pts, curr_pts = prev_pts[idx], curr_pts[idx]

        # If SuperPoint doesn't work well, we need alternative plan
        if prev_pts.shape[0] <= 10:
            prev_pts, curr_pts = alternative_pts(prev_gray, curr_gray)
            
        # Find transformation matrix
        m = cv2.estimateAffine2D(prev_pts, curr_pts)[0]
        # m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]

        # Estimate translation
        dx, dy, da = m[0, 2], m[1, 2], np.arctan2(m[1, 0], m[0, 0])

        # Store transformation
        transforms[i] = [dx, dy, da]

        # Move to next frame
        prev_gray = curr_gray

        # Compute trajectory using cumulative sum of transformations
        trajectory = np.cumsum(transforms, axis=0)

    trajectory = np.cumsum(transforms, axis=0)

    smooth_trajectory = smooth(trajectory)
    # Calculate difference in smoothed_trajectory and trajectory
    difference = smooth_trajectory - trajectory

    # Calculate newer transformation array
    transforms_smooth = transforms + difference

    # Reset stream to first frame
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Write n_frames-1 transformed frames
    for i in range(n_frames - 2):
        # Read next frame
        success, frame = video.read()
        if not success:
            break
        # Extract transformations from the new transformation array
        dx, dy, da = transforms_smooth[i]

        # Reconstruct transformation matrix accordingly to new values
        m = np.array([[np.cos(da), -np.sin(da), dx], [np.sin(da), np.cos(da), dy]])

        # Apply affine warping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (width, height))

        # Fix border artifacts
        frame_stabilized = fixBorder(frame_stabilized)

        # Write the frame to the file
        frame_out = cv2.vconcat([frame, frame_stabilized])

        cv2.imwrite(output_dir + str(i).zfill(5) + ".png", frame_out)
