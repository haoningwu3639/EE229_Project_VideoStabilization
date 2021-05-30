import numpy as np
from cvxpy import *
from tqdm import tqdm

from utils import gaussian_kernel, gaussian_window

def cvx_optimize_path(trajectory, window_size=6, lambda_t=1):
    """
    Input:
    trajectory: original camera trajectory
    window_size

    Output:
    smooth_trajectory: an optimized gaussian smooth camera trajectory 
    """

    height, width, time = trajectory.shape
    smooth_trajectory = np.empty_like(trajectory)

    for i in range(height):
        for j in range(width):
            Prob = Variable(time)
            objective = 0
            for t in range(time):

                # optimized path distance loss
                path_distance = (Prob[t] - trajectory[i, j, t])**2

                # Smoothness Loss
                for d in range(window_size):
                    if t-d < 0:
                        break
                    gauss = gaussian_kernel(t, t-d, window_size)
                    gauss_weight = gauss * (Prob[t] - Prob[t-d])**2
                    if d == 0:
                        smoothness = gauss_weight
                    else:
                        smoothness += gauss_weight

                    objective += path_distance + lambda_t * smoothness

            prob = Problem(Minimize(objective))
            prob.solve()
            smooth_trajectory[i, j, :] = np.asarray(Prob.value).reshape(-1)

    return smooth_trajectory


def offline_optimize_path(trajectory, iterations=50, window_size=6, lambda_t=1):
    """
    Input:
    trajectory: original camera trajectory
    interation: default = 50
    window_size: default = 6
    lambda_t: default = 1

    Output:
    smooth_trajectory: an optimized gaussian smooth camera trajectory 
    """

    height, width, time = trajectory.shape
    smooth_trajectory = np.empty_like(trajectory)

    window = gaussian_window(time, window_size)
    gamma = 1 + lambda_t * np.dot(window, np.ones((trajectory.shape[2],)))

    for i in range(height):
        for j in range(width):
            track = np.array(trajectory[i, j, :])
            for _ in range(iterations):
                track = np.divide(
                    trajectory[i, j, :] + lambda_t * np.dot(window, track), gamma)
            smooth_trajectory[i, j, :] = np.array(track)

    return smooth_trajectory


def online_optimize_path(trajectory, buffer_size=100, iterations=50, window_size=6, beta=1, lambda_t=1):
    """
    Input:
    trajectory: original camera trajectory
    buffer_size: default = 100
    iterations: default = 50
    window_size: default = 32
    beta: default = 1
    lambda_t: default = 1

    Output:
    smooth_trajectory: an optimized gaussian smooth camera trajectory 
    """
    
    height, width, time = trajectory.shape
    smooth_trajectory = np.empty_like(trajectory)

    window = gaussian_window(buffer_size, window_size)

    bar = tqdm(height * width)
    for i in range(height):
        for j in range(width):
            res = []
            d = None
            # online optimization
            for t in range(1, time+1):
                if t < buffer_size + 1:
                    track = np.array(trajectory[i, j, :t])
                    if not d is None:
                        for _ in range(iterations):
                            alpha = trajectory[i, j, :t] + lambda_t * np.dot(window[:t, :t], track)
                            alpha[:-1] = alpha[:-1] + beta * d
                            gamma = 1 + lambda_t * np.dot(window[:t, :t], np.ones((t,)))
                            gamma[:-1] = gamma[:-1] + beta
                            track = np.divide(alpha, gamma)
                else:
                    track = np.array(trajectory[i, j, t-buffer_size: t])
                    for _ in range(iterations):
                        alpha = trajectory[i, j, t-buffer_size: t] + lambda_t * np.dot(window, track)
                        alpha[:-1] = alpha[:-1] + beta * d[1:]
                        gamma = 1 + lambda_t * np.dot(window, np.ones((buffer_size,)))
                        gamma[:-1] = gamma[:-1] + beta
                        track = np.divide(alpha, gamma)
                d = np.asarray(track)
                res.append(track[-1])
            smooth_trajectory[i, j, :] = np.array(res)
            bar.update(1)
    bar.close()
    return smooth_trajectory