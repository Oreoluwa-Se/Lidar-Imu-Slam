FROM osrf/ros:noetic-desktop
ENV DEBINA_FRONTEND=noninteractive
SHELL ["/bin/bash","-c"]

# dependencies for git, python, c++, logging, eigen3, yaml, pcl
RUN apt-get update --fix-missing && \
    apt-get install -y git && \
    apt-get install -y build-essential \
    cmake \
    terminator \
    nano \
    vim \
    python3-rosdep \
    gcc \
    tree \ 
    libgoogle-glog-dev \ 
    libeigen3-dev \ 
    libyaml-cpp-dev \
    libpcl-dev \
    libtbb-dev \
    ros-noetic-rviz ros-noetic-pcl-ros ros-noetic-eigen-conversions \
    libatlas-base-dev libsuitesparse-dev libglew-dev wget\
    ros-noetic-gazebo-ros-pkgs

# # source and build ros 1
RUN mkdir -p /env_ws/bag && \
    mkdir -p /env_ws/src

RUN apt-get -y dist-upgrade && \ 
    source /opt/ros/noetic/setup.bash && \  
    cd /env_ws && catkin_make && \
    source devel/setup.bash

# source and add to .bashrc file
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN echo "source /env_ws/devel/setup.bash" >> ~/.bashrc
RUN echo "source /usr/share/gazebo/setup.bash" >> ~/.bashrc

# install ceres-solver 2.0
RUN mkdir -p /third_party
WORKDIR /third_party
RUN wget https://github.com/ceres-solver/ceres-solver/archive/refs/tags/2.0.0.tar.gz
RUN tar zxf 2.0.0.tar.gz
RUN cd ceres-solver-2.0.0
RUN mkdir build && cd build
RUN ls
RUN cmake -DCMAKE_BUILD_TYPE=Release ./ceres-solver-2.0.0 && make -j2 && make install

# install robin map
RUN git clone https://github.com/Tessil/robin-map.git 
WORKDIR /third_party/robin-map/tests
RUN  mkdir build
RUN cd build && cmake .. && cmake --build . 

# copy rosbag into env_ws


COPY ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod 755 /ros_entrypoint.sh
WORKDIR '/'
ENTRYPOINT [ "/ros_entrypoint.sh" ]
CMD ["bash"]