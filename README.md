git clone --recurse-submodules https://github.com/kernel-AI-robotics/RealSense.git
cd RealSense
colcon build --cmake-args -DUSE_LIFECYCLE_NODE=OFF
source install/setup.bash
