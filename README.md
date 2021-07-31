# **DS-SLAM-HERO**

This project is DS-SLAM system based on the HERO platform, which mainly gets rid of the dependence on GPU.

The Heterogeneous Extensible Robot Open(HERO) platform is designed for robotic research, which brings in the flexible computational capacities by CPU and FPGA. DS-SLAM is a complete robust semantic SLAM system, which could reduce the influence of dynamic objects on pose estimation, such as walking people and other moving robots. Meanwhile, DS-SLAM could also provide semantic presentation of the octo-tree map．In this Open Source project, we provide one example to run DS_SLAM_HERO in the TUM dataset with RGB-D sensors.

As described in **An FPGA Based Energy Efficient DS-SLAM Accelerator for Mobile Robots in Dynamic Environment **Yakun Wu, Li Luo, Shujuan Yin, Mengqi Yu, Fei Qiao, Hongzhi Huang, Xuesong Shi, Qi Wei, Xinjun Liu, Published in Applied Sciences.

DS-SLAM-HERO is developed by the joint research project of iVip Lab @ EE, THU (https://ivip-tsinghua.github.io/iViP-Homepage/).

If you have any questions or use DS_SLAM_HERO for commercial purposes, please contact: qiaofei@tsinghua.edu.cn

# 1. License
 
DS-SLAM-HERO allows personal and research use only. For a commercial license please contact: qiaofei@tsinghua.edu.cn

If you use DS-SLAM in an academic work, please cite their publications as below:

Wu Y, Luo L, Yin S, et al. An FPGA Based Energy Efficient DS-SLAM Accelerator for Mobile Robots in Dynamic Environment[J]. Applied Sciences, 2021, 11(4):1828.

# 2. Prerequisites

We have tested the library in Ubuntu 14.04 and 16.04, but it should be easy to compile in other platforms. The FPGA part is developed by embracing OpenCL programming. The experiment is performed on a computer with HERO.

### DS-SLAM Prereguisites

DS-SLAM-HERO is based on the DS-SLAM. In order to run DS_SLAM_HERO, you have to install environment needed by DS_SLAM.

### OpenCL-SegNet

DS-SLAM-HERO includs an OpenCL-based FPGA accelerator which can run the SegNet model very efficiently. This project is tested with Intel Arrial 10 FPGA and Intel OpenCL SDK v17.1.

### ROS

We provide one example to process the TUM dataset as RGB-D image. A version Hydro or newer is needed. You should create a ROS catkin workspace(in our case, catkin_ws).

### OctoMap and RVIZ

We provide semantic presentation of the octo-tree map by OctoMap. RViz display plugins for visualizing octomap messages. We suggest that the installation path of octomap_mapping and octomap_rviz_plugins to be catkin_ws/src. Add #define COLOR_OCTOMAP_SERVER into the OctomapServer.h at the folder of  octomap_mapping/octomap_server/include/octomap_server Download and install instructions can be found at: https://github.com/OctoMap/octomap_mapping and https://github.com/OctoMap/octomap_rviz_plugins.    

# 3. Building DS_SLAM library and the example

We provide a script DS_SLAM_BUILD.sh to build the third party libraries and DS-SLAM. Please make sure you have installed all previous required dependencies. Execute:

```c++
cd DS-SLAM
chmod +x DS_SLAM_BUILD.sh
./DS_SLAM_BUILD.sh
```

# 4. TUM example

1. Add the path including Examples/ROS/ORB_SLAM2_PointMap_SegNetM to the ROS_PACKAGE_PATH environment variable. Open .bashrc file and add at the end the following line. Replace PATH by the folder where you cloned ORB_SLAM2:

```
export  ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:PATH/DS-SLAM/Examples/ROS/ORB_SLAM2_PointMap_SegNetM
```

2. Download a sequence from http://vision.in.tum.de/data/datasets/rgbd-dataset/download and unzip it. We suggest you download rgbd_dataset_freiburg3_walking_xyz.

3. We provide DS_SLAM_HERO.launch script to run TUM example. Change PATH_TO_SEQUENCE and  PATH_TO_SEQUENCE/associate.txt in the DS_SLAM_TUM.launch to the sequence directory that you download before, then  execute the following command in a new terminal. Execute:

```
cd DS-SLAM
roslaunch DS_SLAM_HERO.launch 
```

#  5. Something about Folders

The function of folder in the catkin_ws/src/ORB_SLAM2_PointMap_SegNetM/Examples/ROS/ORB_SLAM2_PointMap_SegNetM.

1. segmentation: the section of segmentation including source code, header file and dynamic link library created by Cmake.
2. launch: used for showing octomap.
3. prototxts and tools: containing some parameters associated with caffe net relating to the semantic segmentation thread of DS_SLAM. There is the folder of models(provided at https://pan.baidu.com/s/1gkI7nAvijF5Fjj0DTR0rEg extract code: fpa3), please download and place the folder in the same path with the folder of prototxts and tools.





