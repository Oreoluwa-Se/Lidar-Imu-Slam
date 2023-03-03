#ifndef CALCULATION_HELPERS_HPP
#define CALCULATION_HELPERS_HPP

#include "sensor_msgs/point_cloud2_iterator.h"
#include "sensor_msgs/PointCloud2.h"
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <algorithm>
#include "types.hpp"
#include <vector>
#include <cmath>

namespace
{
    using SE3d = Sophus::SE3d;
    using SO3d = Sophus::SO3d;

}

namespace utils
{
    // Pointcloud processing.
    std::vector<double> get_time_stamps(const sensor_msgs::PointCloud2::ConstPtr &msg,
                                        const sensor_msgs::PointField &field = {});

    std::vector<double> normalize_timestamps(const std::vector<double> &timestamps);

    utils::Vec3dVector pointcloud2eigen(const utils::PointCloudXYZI &msg);

    // SOPHUS SUBRUNNERS
    utils::vector<6> delta_pose(const SE3d &first, const SE3d &last);

    // calculate scan angle velocity given a frame rate.
    double calc_scan_ang_vel(int frame_rate);

    // create skew matrix.
    utils::Mat3d skew_matrix(const utils::Vec3d &vec);

    // convert vector to matrix.
    SE3d vector6d_to_mat4d(const utils::vector<6> &x);

    // use a Transform matrix to transform a set of points.
    void transform_points(const SE3d &T, utils::Vec3dVector &points);

    // calculate motion between two poses.
    utils::Vec3Tuple get_motion(const SE3d &start_pose, const SE3d &end_pose, double dt);

    // ADHOC
    inline double square(double x) { return x * x; }

}
#endif