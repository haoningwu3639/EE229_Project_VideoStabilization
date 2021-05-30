from vidstab import VidStab
import matplotlib.pyplot as plt

def main():
    # Using defaults

    stabilizer = VidStab()
    stabilizer.stabilize(input_path='../data/Regular/5.avi', output_path='vidstab.avi')

    # Using a specific keypoint detector
    # stabilizer = VidStab(kp_method = 'ORB')
    # stabilizer.stabilize(input_path = 'input_video.mov', output_path = 'stable_video.avi')

    # Using a specific keypoint detector and customizing keypoint parameters
    # stabilizer = VidStab(kp_method = 'FAST', threshold = 42, nonmaxSuppression = False)
    # stabilizer.stabilize(input_path = 'input_video.mov', output_path = 'stable_video.avi')

    stabilizer.plot_trajectory()
    plt.savefig('./vidstab_trajectory.png')
    plt.clf()

    stabilizer.plot_transforms()
    plt.savefig('./vidstab_transforms.png')
    plt.clf()

if __name__ == '__main__':
    main()