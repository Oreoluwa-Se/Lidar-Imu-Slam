#ifndef ODOM_EKF_HPP
#define ODOM_EKF_HPP

#include "limu/sensors/sync_frame.hpp"
#include <sensor_msgs/Imu.h>
#include <Eigen/SparseCore>
#include <Eigen/Cholesky>
#include "common.hpp"
#include "states.hpp"
#include "helper.hpp"

namespace odometry
{
    struct Pose6D
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        Pose6D(
            const double offset_time, const Eigen::Vector3d &acc,
            const Eigen::Vector3d &gyr, const Eigen::Vector3d &vel,
            const Eigen::Vector3d &pos, const Eigen::Matrix3d &rot)
            : offset_time(offset_time), acc(acc), gyr(gyr), vel(vel),
              pos(pos), rot(rot){};

        double offset_time;
        Eigen::Vector3d acc;
        Eigen::Vector3d gyr;
        Eigen::Vector3d vel;
        Eigen::Vector3d pos;
        Eigen::Matrix3d rot;
    };

    struct PARAMETERS
    {
        typedef std::shared_ptr<PARAMETERS> Ptr;
        int lidar_pose_trail;
        double noise_scale;
        // Imu_noise parameters
        double init_pos_noise;
        double init_vel_noise;
        double init_ori_noise;
        double init_bga_noise;
        double init_baa_noise;
        double init_bat_noise;
        double acc_process_noise;
        double gyro_process_noise;
        double acc_process_noise_rev;
        double gyro_process_noise_rev;
    };

};
#endif