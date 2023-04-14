#ifndef ODOM_EKF_HPP
#define ODOM_EKF_HPP

#include "limu/sensors/sync_frame.hpp"
#include "limu/sensors/imu/frame.hpp"
#include "limu/sensors/lidar/icp.hpp"
#include "limu/sensors/sensor_init.hpp"
#include <Eigen/SparseCore>
#include <Eigen/Cholesky>
#include "common.hpp"
#include "states.hpp"
#include "helper.hpp"
#include "geometry_msgs/Vector3.h"

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

        bool satu_check[6];
    };

    class EKF
    {
    public:
        using LidarInfo = frame::Lidar::ProcessingInfo::Ptr;
        typedef std::unique_ptr<EKF> Ptr;

        EKF(const LidarInfo &icp_params, PARAMETERS::Ptr &parameters)
            : icp_ptr(std::make_shared<lidar::KissICP>(icp_params)),
              est_extrinsic(true), use_imu_as_input(true), // testing when we use imu as input
              laser_point_cov(0.1), init_map(false),
              inp_state(StateInput(parameters)),
              out_state(StateOutput(parameters)),
              Lidar_R_wrt_IMU(Eigen::Matrix3d::Zero()),
              Lidar_T_wrt_IMU(Eigen::Vector3d::Zero()),
              ang_vel_read(Eigen::Vector3d::Zero()),
              acc_vel_read(Eigen::Vector3d::Zero()),
              satu_acc(parameters->satu_acc),
              satu_gyro(parameters->satu_gyro),
              sensor_init(std::make_shared<SensorInit>()),
              data_accum_finished(false), frame_num(0)
        {
            Jaco_rot = Eigen::MatrixXd::Zero(30000, 3);
        }

        void set_extrinsics(const Eigen::Matrix3d &_Lidar_R_wrt_IMU, const Eigen::Vector3d &_Lidar_T_wrt_IMU)
        {
            Lidar_R_wrt_IMU = _Lidar_R_wrt_IMU;
            Lidar_T_wrt_IMU = _Lidar_T_wrt_IMU;
        }

        void update_imu_reading(const geometry_msgs::Vector3 &imu_gyro, const geometry_msgs::Vector3 &imu_acc, double sf = 1)
        {
            ang_vel_read << imu_gyro.x, imu_gyro.y, imu_gyro.z;
            acc_vel_read << imu_acc.x, imu_acc.y, imu_acc.z;
            acc_vel_read *= sf;
        }

        utils::Vec3dVector pre_initialization(frame::LidarImuInit::Ptr &meas, bool init = false);
        void initialize_map(frame::LidarImuInit::Ptr &meas);

        void h_model_input(dyn_share_modified<double> &data, frame::LidarImuInit::Ptr &meas);
        void h_model_output(dyn_share_modified<double> &data, frame::LidarImuInit::Ptr &meas);
        void h_model_IMU_output(dyn_share_modified<double> &data, const frame::LidarImuInit::Ptr &meas);
        bool update_h_model_lidar_output(frame::LidarImuInit::Ptr &meas, bool input);
        bool update_h_model_IMU_output(frame::LidarImuInit::Ptr &meas);
        bool initialize_sensors(double odom_freq, double timestamp, bool &enabled,
                                double &timediff_imu_wrt_lidar, const double &move_start_time);
        utils::Vec3dVector points_body_to_world(const utils::Vec3dVector &points);
        double position_norm();
        bool map_def()
        {
            return init_map;
        }
        std::tuple<Eigen::Vector3d, Eigen::Quaterniond> update_map();
        Sophus::SE3d get_lidar_pos();

    public:
        // attributes
        lidar::KissICP::Ptr icp_ptr;

        bool est_extrinsic, use_imu_as_input;
        Eigen::Matrix3d Lidar_R_wrt_IMU;
        Eigen::Vector3d Lidar_T_wrt_IMU;
        StateInput inp_state;
        StateOutput out_state;
        Eigen::Vector3d ang_vel_read, acc_vel_read;
        Eigen::Vector3d zero = Eigen::Vector3d::Zero();
        SensorInit::Ptr sensor_init;

        int frame_num;

    private:
        template <typename StateType>
        void h_model_lidar(StateType &state, dyn_share_modified<double> &data, frame::LidarImuInit::Ptr &meas);

        template <typename State>
        void calculate_h_lidar(dyn_share_modified<double> &data, int idx, const State &m, const Eigen::Vector3d &src_point);

        template <int H_Dim>
        bool update_h_model(frame::LidarImuInit::Ptr &meas);

        double laser_point_cov, satu_acc, satu_gyro;
        utils::Vec3dVector original_points, curr_downsampled_frame;
        bool init_map, data_accum_finished;
        Eigen::MatrixXd Jaco_rot;
    };
};
#endif