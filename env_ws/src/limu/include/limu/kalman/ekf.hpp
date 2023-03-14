#ifndef ODOM_EKF_HPP
#define ODOM_EKF_HPP

#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include "limu/sensors/sync_frame.hpp"
#include <sensor_msgs/Imu.h>
#include <Eigen/SparseCore>
#include <Eigen/Cholesky>
#include "common.hpp"
#include <array>

namespace kalman
{
    /*EKF state structure
        pose = [x, y, z, w, x, y, z]

        state x:
        [
            0-2: x, y, z [imu-world extrinic]
            3-5: v_x, v_y, v_z [velocity]
            6-9: w, x, y, z [imu-world rotation]
            10-12: bga0, bga1, bga2
            13-15: baa0, baa1, baa2
            16-18: bgt0, bgt1, bgt2
            19-21: grav_x, grav_y, grav_z
            22-24: x, y, z [imu-lidar translation]
            25-28: w, x, y, z [imu-lidar rotation]
            29: imutoLidartimeshift
        ]
    */

    constexpr int POS = 0;
    constexpr int VEL = 3;
    constexpr int ORI = 6;
    constexpr int BGA = 10;
    constexpr int BAA = 13;
    constexpr int BAT = 16;
    constexpr int GRAV = 19;
    constexpr int POS_IMU_LIDAR = 22;
    constexpr int ROT_IMU_LIDAR = 25;
    constexpr int SFT = 29;
    constexpr int LIDAR = 30;
    constexpr int INNER_DIM = LIDAR;
    constexpr int POSE_DIM = 7;
    constexpr int LIDAR_POINT_DIM = 3;

    // BGA = bias gyro additive
    // BAA = bias acc additive
    // BAT = bias acc transform

    constexpr int INNER_VAR_COUNT = 9;
    constexpr std::array<int, INNER_VAR_COUNT> STATE_PARTS = {POS, VEL, ORI, BGA, BAA, BAT, POS_IMU_LIDAR, ROT_IMU_LIDAR, SFT};
    const std::array<std::string, INNER_VAR_COUNT> STATE_PART_NAMES = {"POS", "VEL", "ORI", "BGA", "BAA", "BAT", "POS_IMU_LIDAR", "ROT_IMU_LIDAR", "SFT"};
    constexpr std::array<int, INNER_VAR_COUNT> STATE_PART_SIZES = {3, 3, 4, 3, 3, 3, 3, 4, 1};

    constexpr int Q_ACC = 0;
    constexpr int Q_GYRO = 3;
    constexpr int Q_BGA_DRIFT = 6;
    constexpr int Q_BAA_DRIFT = 9;
    constexpr int Q_DIM = 12;

    struct EKF_PARAMETERS
    {
        typedef std::shared_ptr<EKF_PARAMETERS> Ptr;
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
        // imu time noise
        double init_lidar_imu_time_noise;

        // Trail uncertainty
        double init_pos_trail_noise;
        double init_ori_trail_noise;

        // unceritanity parameter for zero velocity update
        double visualZuptR; // the smaller the stronger the update
    };

    struct Pose6D
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        Pose6D(const double offset_time, const Eigen::Vector3d &acc,
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

    // Odometry implementation -Iterative Error state quaternion version
    class EKF
    {

    public:
        typedef std::unique_ptr<EKF> Ptr;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        struct Tracker
        {
            typedef std::unique_ptr<Tracker> Ptr;
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            inline void populate_imu_pose(double offset_time)
            {
                imu_pose.emplace_back(
                    offset_time, acc_s_last, ang_vel_last,
                    vel, pos, rot);
            }

            sensor_msgs::Imu::ConstPtr last_imu;
            Eigen::Vector3d acc_s_last;
            Eigen::Vector3d ang_vel_last;
            Eigen::Vector3d mean_acc;
            Eigen::Vector3d mean_gyr;
            Eigen::Vector3d vel;
            Eigen::Vector3d pos;
            Eigen::Matrix3d rot;
            std::vector<Pose6D> imu_pose;
        };

        explicit EKF(EKF_PARAMETERS::Ptr parameters);

        void initialize_imu_global_orientation(
            const Eigen::Vector3d &xa, const Eigen::Vector3d &calc_grav);
        void predict(double t, const Eigen::Vector3d &xg, const Eigen::Vector3d &xa,
                     const Eigen::Vector3d &calc_grav, const Eigen::Vector3d &trans_lidar_imu,
                     const Eigen::Matrix3d &rot_lidar_imu);

        void normalize_quaternions(bool only_current = false);
        void zero_vel_update(
            Eigen::VectorXd &state, Eigen::MatrixXd &K_g, Eigen::MatrixXd &P_cov,
            Eigen::MatrixXd &HP, Eigen::LDLT<Eigen::MatrixXd> &inv_s,
            Eigen::MatrixXd &H, Eigen::MatrixXd &R, double r);

        void update_and_propagate();
        Eigen::Vector3d position() const;
        Eigen::Vector3d velocity() const;
        Eigen::Vector4d orientation() const;
        Eigen::Vector3d gravity_check() const;
        double get_current_time() const;
        double speed() const;

    public:
        void motion_compensation_with_imu(frame::LidarImuInit::Ptr &meas);
        // motion compensation tracker
        Tracker::Ptr mc_tracker;

    private:
        Eigen::Matrix4d calculate_S(const Eigen::Vector3d &xg, const Eigen::VectorXd &m, const double dt);

        std::tuple<Eigen::Vector3d, Eigen::Vector4d> propagate_state(
            Eigen::VectorXd &state, Eigen::Matrix4d &A, const Eigen::Matrix3d &R,
            const Eigen::Matrix3d &rot_lidar_imu, const Eigen::Vector3d &trans_lidar_imu,
            const double dt, const Eigen::Vector3d &calc_grav, const Eigen::Vector3d &xa,
            const double acc_process_noise_rev, const double gyro_process_noise);

        void initialize_state_jacobians(
            Eigen::Matrix<double, INNER_DIM, INNER_DIM> &Fx, Eigen::Matrix<double, INNER_DIM, Q_DIM> &Fw, const Eigen::Vector3d &T_ab, const Eigen::Vector4d &prev_quat,
            const Eigen::Matrix4d &A, const Eigen::Matrix3d &R, std::unique_ptr<Eigen::Matrix3d[]> dR,
            const Eigen::Vector3d &xa, const double dt);

        void initialize_process_covariance(
            Eigen::MatrixXd &Pc, double init_pos_noise, double init_vel_noise,
            double init_bga_noise, double init_baa_noise, double init_bat_noise,
            double init_lidar_imu_time_noise, double init_pos_trail_noise,
            double init_ori_trail_noise, double noise_scale);

        void update_visual_pose_aug();
        void update_undo_augmentation();
        void maintain_positive_semi_definite();

        EKF_PARAMETERS::Ptr params;

        // kalman filter parameters
        Eigen::Vector3d grav;
        Eigen::VectorXd m;                     // state dimensions
        Eigen::MatrixXd P;                     // state covariance
        Eigen::Matrix<double, Q_DIM, Q_DIM> Q; // process noise covariance

        // large matrices storage
        Eigen::Matrix<double, INNER_DIM, INNER_DIM> dydx; // used in predict step
        Eigen::Matrix<double, INNER_DIM, Q_DIM> dydq;     // for noise covariance

        // used in the update steps
        Eigen::MatrixXd H;     //  (n * stateDim, n = 1,2,3,4)
        Eigen::MatrixXd HP;    // (n * stateDim)
        Eigen::MatrixXd K;     // (stateDim * n)
        Eigen::MatrixXd tmp_1; // (stateDim * stateDim).
        Eigen::MatrixXd tmp_0; // (stateDim * stateDim).
        Eigen::MatrixXd S;
        Eigen::LDLT<Eigen::MatrixXd> invS; //
        Eigen::MatrixXd R;                 // (n * n)

        // matrixes used for covariance propagation
        Eigen::Matrix<double, INNER_DIM, INNER_DIM> dydx_m;
        Eigen::Matrix<double, INNER_DIM, Q_DIM> dydq_m;
        Eigen::MatrixXd P_m;
        Eigen::MatrixXd tmp_1_m; // (stateDim * stateDim).
        Eigen::MatrixXd tmp_0_m; // (stateDim * stateDim).

        // augmentation matrices -> Mostly for handling things when stationary
        Eigen::SparseMatrix<double> visAugH;
        Eigen::SparseMatrix<double> visAugQ;
        Eigen::SparseMatrix<double> visUnaugmentA;
        std::vector<Eigen::SparseMatrix<double>> visAugA;

        // trackers and helpers
        const int lidar_pose_count;
        const int state_dim;
        std::vector<double> augment_times;
        double time, ZUPTtime, ZRUPTtime, initZUPTtime;
        double noise_scale;
        double last_lidar_end_time;
        bool was_stationary;
        double prev_sampleT;
        double first_sampleT;
        bool orientation_initialized;
        bool first_sample;
        int augment_count;
    };
};
#endif