

# Self-supervised 6-DoF Robot Grasping by Demonstration via Augmented Reality Teleoperation System

This repository contains the contrastive learning code and resources for the research paper titled "Self-supervised 6-DoF Robot Grasping by Demonstration via Augmented Reality Teleoperation System", presented at ICRA2024. [Paper Link](https://arxiv.org/abs/2404.03067)

<!-- ### Authors:
- Xiwen Dengxiong
- Xueting Wang
- Shi Bai
- Yunbo Zhang -->

### Authors:
- [Xiwen Dengxiong](https://sherwindengxiong.github.io/), Xueting Wang, [Yunbo Zhang](https://www.willyunbozhang.com/): Rochester Institute of Technology, Rochester, NY, USA
- Shi Bai: Figure AI, Sunnyvale, CA, USA



### <span style="color:blue">Recent Progress: (based on ROS1)</span>
- Contrastive learning code （demo code）
- Interaction interation and control [Unity_Project](https://github.com/SherwinDengxiong/test_zed)
- Demonstration with the robot [ROS_Package](https://github.com/SherwinDengxiong/xarm_moveit)

### <span style="color:blue">Recent Progress: (based on ROS2)</span>
- Multi-modal interaction with LLM (In Progress)
- Working on additional features

### Resources:
- YouTube Video: [Link](https://www.youtube.com/watch?v=mcrLj-tX90s&t=1s)
- Poster: [Link](https://drive.google.com/file/d/1uwrhE1fvfgeEWirSU_vHL4GJyJXrQex3/view?usp=sharing)
- CDIME Lab: [Link](https://www.youtube.com/@cdimelabs6965)
### Abstract:
Most existing 6-DoF robot grasping solutions depend on strong supervision on grasp pose to ensure satisfactory performance, which could be laborious and impractical when the robot works in some restricted area. To this end, we propose a self-supervised 6-DoF grasp pose detection framework via an Augmented Reality (AR) teleoperation system that can efficiently learn human demonstrations and provide 6-DoF grasp poses without grasp pose annotations. Specifically, the system collects the human demonstration from the AR environment and contrastively learns the grasping strategy from the demonstration. For the real-world experiment, the proposed system leads to satisfactory grasping abilities and learning to grasp unknown objects within three demonstrations.
![View Poster](poster.pdf)


### Installation:
<!-- [Include installation instructions here if applicable] -->

This project is supported by Unity 20.03 or higher and ROS1 Noetic

Unity Packages:
1. URDF Importer version 0.4.0 [Link](https://github.com/Unity-Technologies/URDF-Importer)
2. ROS TCP Connector version 0.5.0 [Link](https://github.com/Unity-Technologies/ROS-TCP-Connector)

ROS1 Packages:
1. ROS TCP Endpoint version 0.5.0 [Link](https://github.com/Unity-Technologies/ROS-TCP-Endpoint)
2. Xarm6 ros1 package from Ufactory [Link](https://github.com/xArm-Developer/xarm_ros)

For further question about the package, please refer to the documentation from [Unity Robotics Hub](https://github.com/Unity-Technologies/Unity-Robotics-Hub) and [Ufactory](https://github.com/xArm-Developer/xarm_ros)

### Usage:
<!-- [Include usage instructions here if applicable] -->

### Citation:
<!-- [If you want users to cite your paper, include citation information here] -->
@article{dengxiong2024self,
  title={Self-supervised 6-DoF Robot Grasping by Demonstration via Augmented Reality Teleoperation System},
  author={Dengxiong, Xiwen and Wang, Xueting and Bai, Shi and Zhang, Yunbo},
   booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2024}
}
### License:
<!-- [Include license information here] -->

### Acknowledgments:
<!-- [If there are any acknowledgments you want to make, include them here] -->
Special thanks to the [Unity Robotics Hub Team] ((https://github.com/Unity-Technologies/Unity-Robotics-Hub)) and [Ufactory Team] for their useful packages and tutorials.
### Contributing:
<!-- [Include guidelines for contributing if applicable] -->

### Issues:
<!-- [If there are any known issues, mention them here] -->

### Notes:
<!-- [Include any additional notes or disclaimers here] -->


### Contact:
- Xiwen Dengxiong: sd6384@rit.edu
