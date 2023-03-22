#ifndef STATES_HPP
#define STATES_HPP

#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include "common.hpp"
#include "helper.hpp"

namespace odometry
{
    /*EKF state structure
            pose = [x, y, z, w, x, y, z]

            state x:
            [
                0-2: x, y, z [imu-world extrinic]
                3-5: v_x, v_y, v_z [imu-world velocity]
                6-9: w, x, y, z [imu-world rotation]
                10-12: bga0, bga1, bga2 [gyro bias]
                13-15: baa0, baa1, baa2 [acc bias]
                16-18: bgt0, bgt1, bgt2 [multiplicative bias]
                19-21: grav_x, grav_y, grav_z [gravity]
                22-24: x, y, z [imu-lidar translation]
                25-28: w, x, y, z [imu-lidar rotation]
                29-32: w, x, y, z [imu gyro pred]
                33-35: x, y, z [imu acc]
            ]
        */

    // Note A_B means from B->A
    constexpr int POS = 0;
    constexpr int VEL = 3;
    constexpr int ORI = 6;
    constexpr int BGA = 10;
    constexpr int BAA = 13;
    constexpr int BAT = 16;
    constexpr int GRAV = 19;
    constexpr int POS_LIDAR_IMU = 22;
    constexpr int ORI_LIDAR_IMU = 25;
    constexpr int IMU_ACC = 29;
    constexpr int IMU_GYRO = 33;
    constexpr int LIDAR = 36;

    constexpr int INNER_DIM = LIDAR;
    constexpr int POSE_DIM = 7;
    constexpr int LIDAR_POINT_DIM = 3;

    constexpr int STATE_IN_DIM = INNER_DIM - 6;
    // BGA = bias gyro additive
    // BAA = bias acc additive
    // BAT = bias acc transform

    constexpr int INNER_VAR_COUNT = 11;
    constexpr std::array<int, INNER_VAR_COUNT> STATE_PARTS = {POS, VEL, ORI, BGA, BAA, BAT, GRAV, POS_LIDAR_IMU, ORI_LIDAR_IMU, IMU_ACC, IMU_GYRO};
    const std::array<std::string, INNER_VAR_COUNT> STATE_PART_NAMES = {"POS", "VEL", "ORI", "BGA", "BAA", "BAT", "GRAV", "POS_LIDAR_IMU", "ORI_LIDAR_IMU", "IMU_ACC", "IMU_GYRO"};
    constexpr std::array<int, INNER_VAR_COUNT> STATE_PART_SIZES = {3, 3, 4, 3, 3, 3, 3, 3, 4, 3, 3};

    constexpr int Q_ACC = 0;
    constexpr int Q_GYRO = 3;
    constexpr int Q_BGA_DRIFT = 6;
    constexpr int Q_BAA_DRIFT = 9;
    constexpr int Q_DIM = 12;

    struct BaseState
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        BaseState()
            : pos(Eigen::Vector3d::Zero()),         // estimated position at imu
              vel(Eigen::Vector3d::Zero()),         // estimated velocity at imu (world frame)
              gyro_bias(Eigen::Vector3d::Zero()),   // gyroscope bias
              acc_bias(Eigen::Vector3d::Zero()),    // acclerator bias
              mult_bias(Eigen::Vector3d::Zero()),   // multiplicative bias
              grav(Eigen::Vector3d::Zero()),        // estimated gravity acceleration
              offset_T_L_I(Eigen::Vector3d::Zero()) // translation from lidar frame L to imu frame I
        {
            (rot << 1, 0, 0, 0).finished();          // estimated rotation at imu
            (offset_R_L_I << 1, 0, 0, 0).finished(); // rotation from lidar frame L to imu frame I
            prev_quat = rot;                         // tracks previous quaternion
        }

        Eigen::Matrix4d quat_4x4_rot(const Eigen::Vector3d &xg, const double dt, bool b_input = true)
        {
            Eigen::Vector3d w;
            {
                if (b_input)
                    w = xg - gyro_bias;
                else
                    w = xg;
            }

            Eigen::Matrix4d S;
            (S << 0, -w[0], -w[1], -w[2],
             w[0], 0, -w[2], w[1],
             w[1], w[2], 0, -w[0],
             w[2], -w[1], w[0], 0)
                .finished();

            S *= dt / 2;
            Eigen::Matrix4d A = S.exp();
            return A;
        }

        void process_covariance(Eigen::MatrixXd &dydx, Eigen::Matrix3d &R, const Eigen::Matrix4d &A, double dt)
        {
            // derivatives of velocity wrt acceleration noise
            dydq.block(VEL, Q_ACC, 3, 3) = R.transpose() * dt;

            // quaternion derivatices w.r.t gyroscope noise
            Eigen::Matrix4d dS0, dS1, dS2;
            (dS0 << 0, dt / 2, 0, 0, -dt / 2, 0, 0, 0, 0, 0, 0, dt / 2, 0, 0, -dt / 2, 0).finished();
            (dS1 << 0, 0, dt / 2, 0, 0, 0, 0, -dt / 2, -dt / 2, 0, 0, 0, 0, dt / 2, 0, 0).finished();
            (dS2 << 0, 0, 0, dt / 2, 0, 0, dt / 2, 0, 0, -dt / 2, 0, 0, -dt / 2, 0, 0, 0).finished();
            dydq.block(ORI, Q_GYRO, 4, 1) = A * dS0 * prev_quat;
            dydq.block(ORI, Q_GYRO + 1, 4, 1) = A * dS1 * prev_quat;
            dydq.block(ORI, Q_GYRO + 2, 4, 1) = A * dS2 * prev_quat;
            dydq.block(BGA, Q_BGA_DRIFT, 3, 3).setIdentity(3, 3);
            dydq.block(BAA, Q_BAA_DRIFT, 3, 3).setIdentity(3, 3);

            // velocity and orientation wrt sensors
            dydq.block(VEL, Q_GYRO, 3, 3) = dydx.block(VEL, ORI, 3, 4) * dydq.block(ORI, Q_GYRO, 4, 3);
        }

        Eigen::Vector3d pos;          // estimated imu position wrt global
        Eigen::Vector3d vel;          // estimated imu velocity wrt global
        Eigen::Vector4d rot;          // estimated imu rotation wrt global
        Eigen::Vector3d gyro_bias;    // gyroscope bias
        Eigen::Vector3d acc_bias;     // acclerator bias
        Eigen::Vector3d mult_bias;    // multiplicative bias
        Eigen::Vector3d grav;         // estimated gravity acceleration
        Eigen::Vector3d offset_T_L_I; // rotation from lidar frame L to imu frame I
        Eigen::Vector4d offset_R_L_I; // translation from lidar frame L to imu frame I
        Eigen::MatrixXd dydx;         // used in predict step
        Eigen::MatrixXd dydq;         // for noise covariance

        Eigen::Vector4d prev_quat; // should be updated after state propagation
    };

    // EXTENSIONS
    struct StateInput : public BaseState
    {
        StateInput() : BaseState()
        {
            dydx = Eigen::MatrixXd::Zero(STATE_IN_DIM, STATE_IN_DIM);
            dydq = Eigen::MatrixXd::Zero(STATE_IN_DIM, Q_DIM);
        }

        Eigen::Matrix<double, STATE_IN_DIM, 1> get_f(const Eigen::Vector3d &xg, const Eigen::Vector3d &xa)
        {
            Eigen::Matrix<double, STATE_IN_DIM, 1> m = Eigen::Matrix<double, STATE_IN_DIM, 1>::Zero();

            // position
            m.segment(POS, 3) = vel;

            // velocity
            Eigen::Matrix3d R = utils::quat2rmat(rot);
            Eigen::Vector3d Txab = m.segment(BAT, 3).asDiagonal() * xa - m.segment(BAA, 3);
            m.segment(VEL, 3) += (R.transpose() * Txab + grav);

            // orientation -> going to be an issue here
            // Eigen::Matrix4d A = quat_4x4_rot(xg, 1.0); // apply this update later
            m.segment(ORI, 4) = rot;

            return m;
        }

        void dydx_dydq(const Eigen::Vector3d &xg, const Eigen::Vector3d &xa, double dt)
        {
            dydx.setZero();
            dydq.setZero();

            dydx.block(POS, POS, 3, 3).setIdentity(3, 3);
            dydx.block(VEL, VEL, 3, 3).setIdentity(3, 3);
            dydx.block(POS, VEL, 3, 3).setIdentity(3, 3) *= dt;
            dydx.block(BGA, BGA, 3, 3).setIdentity(3, 3);
            dydx.block(BAA, BAA, 3, 3).setIdentity(3, 3);
            dydx.block(BAT, BAT, 3, 3).setIdentity(3, 3);
            dydx.block(GRAV, GRAV, 3, 3).setIdentity(3, 3);
            dydx.block(POS, GRAV, 3, 3).setIdentity(3, 3) *= dt;
            dydx.block(POS_LIDAR_IMU, POS_LIDAR_IMU, 3, 3).setIdentity(3, 3);
            dydx.block(ORI_LIDAR_IMU, ORI_LIDAR_IMU, 4, 4).setIdentity(4, 4);

            // -dt cause using global errors
            Eigen::Matrix4d A = quat_4x4_rot(xg, -dt, true);
            Eigen::Matrix3d dR[4];
            Eigen::Matrix3d R = utils::extract_rot_dr(A * rot, dR);
            Eigen::Vector3d T_ab = mult_bias.asDiagonal() * xa - acc_bias;

            // derivatives of velocity w.r.t quaternion
            for (int i = 0; i < 4; i++)
                dydx.block(VEL, ORI + i, 3, 1) = dR[i].transpose() * T_ab * dt;

            dydx.block(VEL, ORI, 3, 4) = dydx.block(VEL, ORI, 3, 4) * A;

            // derivative of quaternion wrt self
            dydx.block(ORI, ORI, 4, 4) = A;

            // // derivatives of velocity wrt acceleration noise
            // dydq.block(VEL, Q_ACC, 3, 3) = R.transpose() * dt;

            // // quaternion derivatices w.r.t gyroscope noise
            // Eigen::Matrix4d dS0, dS1, dS2;
            // (dS0 << 0, dt / 2, 0, 0, -dt / 2, 0, 0, 0, 0, 0, 0, dt / 2, 0, 0, -dt / 2, 0).finished();
            // (dS1 << 0, 0, dt / 2, 0, 0, 0, 0, -dt / 2, -dt / 2, 0, 0, 0, 0, dt / 2, 0, 0).finished();
            // (dS2 << 0, 0, 0, dt / 2, 0, 0, dt / 2, 0, 0, -dt / 2, 0, 0, -dt / 2, 0, 0, 0).finished();
            // dydq.block(ORI, Q_GYRO, 4, 1) = A * dS0 * prev_quat;
            // dydq.block(ORI, Q_GYRO + 1, 4, 1) = A * dS1 * prev_quat;
            // dydq.block(ORI, Q_GYRO + 2, 4, 1) = A * dS2 * prev_quat;
            // dydq.block(BGA, Q_BGA_DRIFT, 3, 3).setIdentity(3, 3);
            // dydq.block(BAA, Q_BAA_DRIFT, 3, 3).setIdentity(3, 3);

            // // velocity and orientation wrt sensors
            // dydq.block(VEL, Q_GYRO, 3, 3) = dydx.block(VEL, ORI, 3, 4) * dydq.block(ORI, Q_GYRO, 4, 3);
            process_covariance(dydx, R, A, dt);
            // derivative of velocity w.r.t gyro bias
            dydx.block(VEL, BGA, 3, 3) = -dydq.block(VEL, Q_GYRO, 3, 3);

            // quaternion derivative wrt gyro bias
            dydx.block(ORI, BGA, 4, 3) = -dydq.block(ORI, Q_GYRO, 4, 3);

            // derivatives of the velocity w.r.t the acc. bias
            dydx.block(VEL, BAA, 3, 3) = -R.transpose() * dt;

            // derivatives of the velocity w.r.t the acc. transformation
            dydx.block(VEL, BAT, 3, 3) = R.transpose() * xa.asDiagonal() * dt;
        }

        // void h_model_input() {}
    };

    struct StateOutput : public BaseState
    {
        StateOutput()
            : BaseState(),
              imu_acc(Eigen::Vector3d::Zero()),
              imu_gyro(Eigen::Vector3d::Zero())
        {
            dydx = Eigen::MatrixXd::Zero(INNER_DIM, INNER_DIM);
            dydq = Eigen::MatrixXd::Zero(INNER_DIM, Q_DIM);
        }

        Eigen::Matrix<double, INNER_DIM, 1> get_f(const Eigen::Vector3d &xg, const Eigen::Vector3d &xa)
        {
            Eigen::Matrix<double, INNER_DIM, 1> m = Eigen::Matrix<double, INNER_DIM, 1>::Zero();

            // position
            m.segment(POS, 3) = vel;

            // velocity - using estimated as outputs
            Eigen::Matrix3d R = utils::quat2rmat(rot);
            m.segment(VEL, 3) += (R.transpose() * imu_acc + grav);

            // orientation
            m.segment(ORI, 4) = imu_gyro;

            return m;
        }

        void dydx_dydq(const Eigen::Vector3d &xg, const Eigen::Vector3d &xa, double dt)
        {
            dydx.setZero();
            dydq.setZero();

            dydx.block(POS, POS, 3, 3).setIdentity(3, 3);
            dydx.block(VEL, VEL, 3, 3).setIdentity(3, 3);
            dydx.block(POS, VEL, 3, 3).setIdentity(3, 3) *= dt;
            dydx.block(BGA, BGA, 3, 3).setIdentity(3, 3);
            dydx.block(BAA, BAA, 3, 3).setIdentity(3, 3);
            dydx.block(BAT, BAT, 3, 3).setIdentity(3, 3);
            dydx.block(GRAV, GRAV, 3, 3).setIdentity(3, 3);
            dydx.block(POS, GRAV, 3, 3).setIdentity(3, 3) *= dt;
            dydx.block(POS_LIDAR_IMU, POS_LIDAR_IMU, 3, 3).setIdentity(3, 3);
            dydx.block(ORI_LIDAR_IMU, ORI_LIDAR_IMU, 4, 4).setIdentity(4, 4);

            // -dt cause using global errors
            Eigen::Matrix4d A = quat_4x4_rot(xg, -dt, false);
            Eigen::Matrix3d dR[4];
            Eigen::Matrix3d R = utils::extract_rot_dr(A * rot, dR);
            Eigen::Vector3d T_ab = xa;

            // derivatives of velocity w.r.t quaternion
            for (int i = 0; i < 4; i++)
                dydx.block(VEL, ORI + i, 3, 1) = dR[i].transpose() * T_ab * dt;

            dydx.block(VEL, ORI, 3, 4) = dydx.block(VEL, ORI, 3, 4) * A;

            // derivative of quaternion wrt self
            dydx.block(ORI, ORI, 4, 4) = A;

            // derivatives of velocity wrt acceleration noise
            dydq.block(VEL, Q_ACC, 3, 3) = R.transpose() * dt;

            // quaternion derivatices w.r.t gyroscope noise
            Eigen::Matrix4d dS0, dS1, dS2;
            (dS0 << 0, dt / 2, 0, 0, -dt / 2, 0, 0, 0, 0, 0, 0, dt / 2, 0, 0, -dt / 2, 0).finished();
            (dS1 << 0, 0, dt / 2, 0, 0, 0, 0, -dt / 2, -dt / 2, 0, 0, 0, 0, dt / 2, 0, 0).finished();
            (dS2 << 0, 0, 0, dt / 2, 0, 0, dt / 2, 0, 0, -dt / 2, 0, 0, -dt / 2, 0, 0, 0).finished();
            dydq.block(ORI, Q_GYRO, 4, 1) = A * dS0 * prev_quat;
            dydq.block(ORI, Q_GYRO + 1, 4, 1) = A * dS1 * prev_quat;
            dydq.block(ORI, Q_GYRO + 2, 4, 1) = A * dS2 * prev_quat;
            dydq.block(BGA, Q_BGA_DRIFT, 3, 3).setIdentity(3, 3);
            dydq.block(BAA, Q_BAA_DRIFT, 3, 3).setIdentity(3, 3);

            // velocity and orientation wrt sensors
            dydq.block(VEL, Q_GYRO, 3, 3) = dydx.block(VEL, ORI, 3, 4) * dydq.block(ORI, Q_GYRO, 4, 3);

            // derivative of velocity w.r.t predicted acc
            dydx.block(VEL, IMU_ACC, 3, 3) = R.transpose() * dt;

            // quaternion derivative wrt predicted gyro
            Eigen::Matrix3d rot_mat;
            {
                Eigen::Vector4d quat;
                quat[0] = A(3, 2);
                quat[1] = A(2, 3);
                quat[2] = A(1, 3);
                quat[3] = A(2, 1);

                rot_mat = utils::quat2rmat(quat);
            }
            dydx.block(ORI, IMU_GYRO, 3, 3) = rot_mat * utils::quat2rmat(prev_quat);
        }

        Eigen::Vector3d imu_acc;  // estimated imu acceleration
        Eigen::Vector3d imu_gyro; // estimated imu gyro
    };
}
#endif