#ifndef ODOM_EKF_HPP
#define ODOM_EKF_HPP

#include "limu/sensors/sync_frame.hpp"
#include "limu/sensors/imu/frame.hpp"
#include "limu/sensors/lidar/icp.hpp"
#include <sensor_msgs/Imu.h>
#include <Eigen/SparseCore>
#include <Eigen/Cholesky>
#include "common.hpp"
#include "states.hpp"
#include "helper.hpp"

namespace odometry
{
    template <typename T>
    struct dyn_share_modified
    {
        bool valid, converge;
        T M_noise;
        Eigen::Matrix<T, Eigen::Dynamic, 1> z;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> h_x;
        Eigen::Matrix<T, 6, 1> z_IMU;
        Eigen::Matrix<T, 6, 1> R_IMU;
    };

    class EKF
    {
    public:
        using LidarInfo = frame::Lidar::ProcessingInfo::Ptr;
        typedef std::shared_ptr<EKF> Ptr;
        EKF(const LidarInfo &icp_params, PARAMETERS::Ptr &parameters)
            : icp_ptr(std::make_shared<lidar::KissICP>(icp_params)),
              est_extrinsic(true), use_imu_as_input(false),
              laser_point_cov(0.1), init_map(false),
              inp_state(StateInput(parameters)),
              out_state(StateOutput(parameters)){};

        void h_model_input(dyn_share_modified<double> &data, frame::LidarImuInit::Ptr &meas);
        void h_model_output(dyn_share_modified<double> &data, frame::LidarImuInit::Ptr &meas);
        void h_model_IMU_output(dyn_share_modified<double> &data, const frame::LidarImuInit::Ptr &meas);
        bool update_h_model_input(frame::LidarImuInit::Ptr &meas);
        bool update_h_model_output(frame::LidarImuInit::Ptr &meas);
        bool update_h_model_IMU_output(frame::LidarImuInit::Ptr &meas);
        utils::Vec3dVector points_body_to_world(const utils::Vec3dVector &points);

    public:
        // attributes
        lidar::KissICP::Ptr icp_ptr;

        bool est_extrinsic;
        bool use_imu_as_input;
        Eigen::Matrix3d Lidar_R_wrt_IMU = Eigen::Matrix3d::Zero();
        Eigen::Vector3d Lidar_T_wrt_IMU = Eigen::Vector3d::Zero();

    private:
        StateInput inp_state;
        StateOutput out_state;
        double laser_point_cov;
        utils::Vec3dVector curr_downsampled_frame;

        // process noise covariance
        bool init_map;
    };
};
#endif