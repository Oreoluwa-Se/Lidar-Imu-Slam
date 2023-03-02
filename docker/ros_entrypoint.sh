#!/bin/bash
set -e

. "opt/ros/noetic/setup.bash"
. "/env_ws/devel/setup.bash"
. "/usr/share/gazebo/setup.sh"


exec "$@"