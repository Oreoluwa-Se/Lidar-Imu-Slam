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

    utils::Vec3d rotation_matrix_to_euler_angles(const utils::Mat3d &rot);

    // ADHOC
    inline double square(double x)
    {
        return x * x;
    }

    // Helper for calculating voxel
    utils::Voxel get_vox_index(const utils::Vec3d &point, double vox_size);

    inline void print_vector(const std::vector<utils::PointNormal, Eigen::aligned_allocator<utils::PointNormal>> &vec)
    {
        std::cout << "Eigen::Vector3d Vector: " << std::endl;
        std::cout << "------------------------" << std::endl;
        for (int i = 0; i < vec.size(); ++i)
        {
            std::cout << vec[i].x << ", " << vec[i].y << ", " << vec[i].z << std::endl;
        }
        std::cout << "------------------------" << std::endl;
    }

    inline void print_vector(const std::vector<utils::Vec3d> &vec)
    {
        std::cout << "Eigen::Vector3d Vector: " << std::endl;
        std::cout << "------------------------" << std::endl;
        for (int i = 0; i < vec.size(); ++i)
        {
            std::cout << vec[i][0] << ", " << vec[i][1] << ", " << vec[i][2] << std::endl;
        }
        std::cout << "------------------------" << std::endl;
    }

}
#endif