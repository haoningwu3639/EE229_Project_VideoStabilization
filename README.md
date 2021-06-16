
# EE229 Project - Video Stabilization
Group assginment of EE229 Digital Image Processing, a video stabilization algorithm based on sparse optical flow which combines the advantage of DIP theory and neural network performance.

# Authors:
| Name | ID | E-mail |
| :-----: | :----: | :----: |
| Haoning Wu* | 518030910285 | whn15698781666@sjtu.edu.cn |
| Longrun Zhi | 518030910320 | zlongrun@sjtu.edu.cn |
| Yifei Li | 518030901306 | liyifei919518@sjtu.edu.cn |

# Requirements
Code tested on following environments, other version should also work:
* python 3.6.3
* cv2
* numpy
* matplotlib
* pytorch
* vidstab
* SuperPoint pretrained model downloaded at [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork)

# Project Structure
* [`bash`](bash): Some bash scripts, not important.
* [`images`](images): Some figures in the paper, not important, just referring to the paper is enough.
* [`data`](data): Place the dataset downloaded from [NUS dataset](http://liushuaicheng.org/SIGGRAPH2013/database.html) here.
* [`pretrained_model`](pretrained_model): Place the pretrained superpoint model here.
* [`src`](src): Source code
* [`src/vidstab_test.py`](src/vidstab_test.py): the script used to utilize python vidstab to stabilize videos.
* `batch_rename.py`, `utils.py`, `metrics.py` are tool scripts.
* `coarse_stab.py`, `fine_stab.py`, `optimizer.py` `propagation.py`, superpoint are several key scripts of the algorithm.
* [`src/main.py`](src/main.py): Main function of the algorithm, simply run it with
    * python --input_video ... --output_dir ...

# Results 
 Contact us for more video results.

# References
 If you gonna learn more about Video Stabilization, we recommend this repository for you: [awesome-video-stabilization](https://github.com/yaochih/awesome-video-stabilization)

# Miscs
 Since we still have a lot of work to do, we have no time to maintain this project now. We hope that we can further improve this repository when we are free in the future. If you have any questions, please contact us by email. Thanks.
