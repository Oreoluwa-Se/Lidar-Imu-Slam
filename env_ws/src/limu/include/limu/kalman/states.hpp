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
    constexpr int IMU_GYRO = 32;
    constexpr int LIDAR = 35;

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

    constexpr int h_matrix_col = 14;

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

        // stuff
        double satu_acc;
        double satu_gyro;
    };

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

        explicit BaseState(const PARAMETERS::Ptr &parameters)
            : pos(Eigen::Vector3d::Zero()),          // estimated position at imu
              vel(Eigen::Vector3d::Zero()),          // estimated velocity at imu (world frame)
              gyro_bias(Eigen::Vector3d::Zero()),    // gyroscope bias
              acc_bias(Eigen::Vector3d::Zero()),     // acclerator bias
              mult_bias(Eigen::Vector3d::Zero()),    // multiplicative bias
              grav(Eigen::Vector3d::Zero()),         // estimated gravity acceleration
              offset_T_L_I(Eigen::Vector3d::Zero()), // translation from lidar frame L to imu frame I
              params(parameters)
        {
            (rot << 1, 0, 0, 0).finished();          // estimated rotation at imu
            (offset_R_L_I << 1, 0, 0, 0).finished(); // rotation from lidar frame L to imu frame I
            prev_quat = rot;                         // tracks previous quaternion
            process_covariance_setup();
        }

        void initialize_orientation(const Eigen::Vector3d &xa)
        {
            Eigen::Vector3d f_grav(0, 0, gravity);
            Eigen::Quaterniond orient = Eigen::Quaterniond::FromTwoVectors(f_grav, xa);
            Eigen::Vector4d q;
            q << orient.w(), orient.x(), orient.y(), orient.z();
            rot = q;
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

        Eigen::Matrix3d L_I_rot_m()
        {
            return utils::quat2rmat(offset_R_L_I);
        }

        Eigen::Matrix3d rot_m()
        {
            return utils::quat2rmat(rot);
        }

        void initialize_state_covariance()
        {
            P.setIdentity();

            // imu - global extrinsic
            P.block(POS, POS, 3, 3) *= utils::square(params->init_pos_noise);
            P.block(VEL, VEL, 3, 3) *= utils::square(params->init_vel_noise);
            P.block(ORI, ORI, 4, 4) *= utils::square(params->init_ori_noise);

            // imu bias initialization
            P.block(BGA, BGA, 3, 3) *= utils::square(params->init_bga_noise);
            P.block(BAA, BAA, 3, 3) *= utils::square(params->init_baa_noise);
            P.block(BAT, BAT, 3, 3) *= utils::square(params->init_bat_noise);
            P.block(GRAV, GRAV, 3, 3) *= utils::square(params->init_bat_noise);

            // imu -lidar extrinsic
            P.block(POS_LIDAR_IMU, POS_LIDAR_IMU, 3, 3) *= utils::square(params->init_pos_noise);
            P.block(ORI_LIDAR_IMU, ORI_LIDAR_IMU, 4, 4) *= utils::square(params->init_ori_noise);
        }

        void dydq_setup(Eigen::MatrixXd &dydx, Eigen::Matrix3d &R, const Eigen::Matrix4d &A, double dt)
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

        void process_covariance_setup()
        {
            // covariance noise Q setup
            Q.block(Q_ACC, Q_ACC, 3, 3).setIdentity(3, 3) *= utils::square(params->acc_process_noise);
            Q.block(Q_GYRO, Q_GYRO, 3, 3).setIdentity(3, 3) *= utils::square(params->gyro_process_noise);
            Q *= params->noise_scale;
        }

        void dydx_dydq_base(const Eigen::Vector3d &xg, const Eigen::Vector3d &xa, double dt, bool b_input = true)
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
            Eigen::Matrix4d A = quat_4x4_rot(xg, -dt, b_input);
            Eigen::Matrix3d dR[4];
            Eigen::Matrix3d R = utils::extract_rot_dr(A * rot, dR);
            Eigen::Vector3d T_ab = b_input ? mult_bias.asDiagonal() * xa - acc_bias : xa;

            // derivatives of velocity w.r.t quaternion
            for (int i = 0; i < 4; i++)
                dydx.block(VEL, ORI + i, 3, 1) = dR[i].transpose() * T_ab * dt;

            dydx.block(VEL, ORI, 3, 4) = dydx.block(VEL, ORI, 3, 4) * A;

            // derivative of quaternion wrt self
            dydx.block(ORI, ORI, 4, 4) = A;

            dydq_setup(dydx, R, A, dt);

            if (b_input)
            {
                // derivative of velocity w.r.t gyro bias
                dydx.block(VEL, BGA, 3, 3) = -dydq.block(VEL, Q_GYRO, 3, 3);

                // quaternion derivative wrt gyro bias
                dydx.block(ORI, BGA, 4, 3) = -dydq.block(ORI, Q_GYRO, 4, 3);

                // derivatives of the velocity w.r.t the acc. bias
                dydx.block(VEL, BAA, 3, 3) = -R.transpose() * dt;

                // derivatives of the velocity w.r.t the acc. transformation
                dydx.block(VEL, BAT, 3, 3) = R.transpose() * xa.asDiagonal() * dt;
            }
            else
            {
                // derivative of velocity w.r.t predicted acc
                dydx.block(VEL, IMU_ACC, 3, 3) = R.transpose() * dt;

                // orientation derivative wrt predicted gyro
                dydx.block(ORI, IMU_GYRO, 3, 3) = dt * utils::quat2rmat(prev_quat);
            }
        }

        void noise_covariance_increment(double dt)
        {
            // random walk bias for gyro
            if (params->gyro_process_noise > 0.0)
            {
                const double qc = utils::square(params->gyro_process_noise);
                const double theta = params->gyro_process_noise_rev;
                Q.block(Q_BGA_DRIFT, Q_BGA_DRIFT, 3, 3).setIdentity(3, 3) *= params->noise_scale * qc;

                if (theta > 0.0)
                    Q.block(Q_BGA_DRIFT, Q_BGA_DRIFT, 3, 3) *= (1 - exp(-2 * dt * theta)) / (2 * theta);
            }

            // random walk bias for accelerometer
            if (params->acc_process_noise > 0.0)
            {
                const double qc = utils::square(params->acc_process_noise);
                const double theta = params->acc_process_noise_rev;
                Q.block(Q_BAA_DRIFT, Q_BAA_DRIFT, 3, 3).setIdentity(3, 3) *= params->noise_scale * qc;

                if (theta > 0.0)
                    Q.block(Q_BAA_DRIFT, Q_BAA_DRIFT, 3, 3) *= (1 - exp(-2 * dt * theta)) / (2 * theta);
            }
        }

        template <int Rows = Eigen::Dynamic>
        Eigen::Matrix<double, Rows, h_matrix_col> iterative_properties()
        {
            /*
             * To return state properties affected by measurement and used in the iterative step.
             * properies are: pos, rot, offset_trans_lidar_imu, offset_rot_lidar_imu
             * note the offset_trans_lidar_imu to be put in quaternion form
             */

            Eigen::Matrix<double, Rows, h_matrix_col> props;
            // imu_world parameters
            props.block(0, 0, Rows, 3) = P.block(0, POS, Rows, 3);
            props.block(0, 3, Rows, 4) = P.block(0, ORI, Rows, 4);

            // lidar imu positions
            props.block(0, 7, Rows, 7) = P.block(0, POS_LIDAR_IMU, Rows, 7);

            return props;
        }

        template <int State_DIM = Eigen::Dynamic>
        Eigen::MatrixXd col_iterative_properties(const Eigen::MatrixXd &PHT)
        {
            /*
             * Given the PHT matrix want to extract relevant information from the column
             * properies are: pos, rot, offset_trans_lidar_imu, offset_rot_lidar_imu
             * note the offset_trans_lidar_imu to be put in quaternion form
             */

            const int num_meas = PHT.cols();
            const int half_col = h_matrix_col / 2;

            Eigen::MatrixXd props = Eigen::MatrixXd::Zero(h_matrix_col, num_meas);
            // imu_world parameters
            props.block(0, 0, 3, num_meas) = PHT.block(POS, 0, 3, num_meas);
            props.block(3, 0, 4, num_meas) = PHT.block(ORI, 0, 4, num_meas);

            // lidar imu positions
            props.block(7, 0, half_col, num_meas) = PHT.block(POS_LIDAR_IMU, 0, 7, num_meas);

            return props;
        }

        Eigen::Vector4d increment_q(Eigen::Vector4d &orig, Eigen::Vector4d &dx)
        {
            Eigen::Quaterniond q1(orig);
            Eigen::Quaterniond q2(dx);

            Eigen::Quaterniond res = q2 * q1;
            res.normalize();
            return res.coeffs();
        }

        Eigen::Vector4d decrement_q(Eigen::Vector4d &orig, Eigen::Vector4d &dx)
        {
            Eigen::Quaterniond q1(orig);
            Eigen::Quaterniond q2(dx);

            Eigen::Quaterniond res = q2.inverse() * q1;
            res.normalize();
            return res.coeffs();
        }

        Eigen::Vector3d pos;                   // estimated imu position wrt global
        Eigen::Vector3d vel;                   // estimated imu velocity wrt global
        Eigen::Vector4d rot;                   // estimated imu rotation wrt global
        Eigen::Vector3d gyro_bias;             // gyroscope bias
        Eigen::Vector3d acc_bias;              // acclerator bias
        Eigen::Vector3d mult_bias;             // multiplicative bias
        Eigen::Vector3d grav;                  // estimated gravity acceleration
        Eigen::Vector3d offset_T_L_I;          // rotation from lidar frame L to imu frame I
        Eigen::Vector4d offset_R_L_I;          // translation from lidar frame L to imu frame I
        Eigen::MatrixXd dydx;                  // used in predict step
        Eigen::MatrixXd dydq;                  // for noise covariance
        Eigen::Matrix<double, Q_DIM, Q_DIM> Q; // process covariance
        Eigen::MatrixXd P;                     // state covariance
        Eigen::Vector4d prev_quat;             // should be updated after state propagation
        PARAMETERS::Ptr params;
    };

    // EXTENSIONS
    struct StateInput : public BaseState
    {
        StateInput()
            : BaseState()
        {
            dydx = Eigen::MatrixXd::Zero(STATE_IN_DIM, STATE_IN_DIM);
            dydq = Eigen::MatrixXd::Zero(STATE_IN_DIM, Q_DIM);
            P = Eigen::MatrixXd::Zero(STATE_IN_DIM, STATE_IN_DIM);
            initialize_state_covariance();
        }

        explicit StateInput(PARAMETERS::Ptr &parameters) : BaseState(parameters)
        {
            dydx = Eigen::MatrixXd::Zero(STATE_IN_DIM, STATE_IN_DIM);
            dydq = Eigen::MatrixXd::Zero(STATE_IN_DIM, Q_DIM);
            P = Eigen::MatrixXd::Zero(STATE_IN_DIM, STATE_IN_DIM);

            // same base state covariance
            initialize_state_covariance();
        }

        // constructor definitions
        StateInput(const StateInput &b) : BaseState()
        {
            this->pos = b.pos;
            this->vel = b.vel;
            this->rot = b.rot;
            this->gyro_bias = b.gyro_bias;
            this->acc_bias = b.acc_bias;
            this->mult_bias = b.mult_bias;
            this->grav = b.grav;
            this->offset_T_L_I = b.offset_T_L_I;
            this->offset_R_L_I = b.offset_R_L_I;
            this->dydx = b.dydx;
            this->dydq = b.dydq;
            this->Q = b.Q;
            this->P = b.P;
            this->prev_quat = b.prev_quat;
        }

        StateInput &operator=(const StateInput &b)
        {
            this->pos = b.pos;
            this->vel = b.vel;
            this->rot = b.rot;
            this->gyro_bias = b.gyro_bias;
            this->acc_bias = b.acc_bias;
            this->mult_bias = b.mult_bias;
            this->grav = b.grav;
            this->offset_T_L_I = b.offset_T_L_I;
            this->offset_R_L_I = b.offset_R_L_I;
            this->dydx = b.dydx;
            this->dydq = b.dydq;
            this->Q = b.Q;
            this->P = b.P;
            this->prev_quat = b.prev_quat;

            return *this;
        }

        StateInput operator+(const Eigen::Matrix<double, STATE_IN_DIM, 1> &m)
        {
            StateInput a;
            a.pos = this->pos + m.segment(POS, 3);
            a.vel = this->vel + m.segment(VEL, 3);

            Eigen::Vector4d ori_val = m.segment(ORI, 4);
            a.rot = increment_q(this->rot, ori_val);

            a.gyro_bias = this->gyro_bias + m.segment(BGA, 3);
            a.acc_bias = this->acc_bias + m.segment(BAA, 3);
            a.mult_bias = this->mult_bias;
            a.grav = this->grav + m.segment(GRAV, 3);
            a.offset_T_L_I = this->offset_T_L_I + m.segment(POS_LIDAR_IMU, 3);

            ori_val = m.segment(ORI_LIDAR_IMU, 4);
            a.offset_R_L_I = decrement_q(this->offset_R_L_I, ori_val);

            a.dydx = this->dydx;
            a.dydq = this->dydq;
            a.Q = this->Q;
            a.P = this->P;
            a.prev_quat = this->prev_quat;

            return a;
        }

        StateInput &operator+=(const Eigen::Matrix<double, STATE_IN_DIM, 1> &m)
        {
            this->pos += m.segment(POS, 3);
            this->vel += m.segment(VEL, 3);

            Eigen::Vector4d ori_val = m.segment(ORI, 4);
            this->rot = increment_q(this->rot, ori_val);

            this->gyro_bias += m.segment(BGA, 3);
            this->acc_bias += m.segment(BAA, 3);
            this->mult_bias = m.segment(BAT, 3);
            this->grav += m.segment(GRAV, 3);
            this->offset_T_L_I += m.segment(POS_LIDAR_IMU, 3);

            ori_val = m.segment(ORI_LIDAR_IMU, 4);
            this->offset_R_L_I = decrement_q(this->offset_R_L_I, ori_val);

            return *this;
        }

        Eigen::Matrix<double, STATE_IN_DIM, 1> get_f(const Eigen::Vector3d &xg, const Eigen::Vector3d &xa)
        {
            Eigen::Matrix<double, STATE_IN_DIM, 1> m = Eigen::Matrix<double, STATE_IN_DIM, 1>::Zero();

            // position
            m.segment(POS, 3) = vel;

            // velocity
            Eigen::Matrix3d R = utils::quat2rmat(rot);
            Eigen::Vector3d Txab = m.segment(BAT, 3).asDiagonal() * xa - m.segment(BAA, 3);
            m.segment(VEL, 3) = (R.transpose() * Txab + grav);

            // orientation -> going to be an issue here
            m.segment(ORI, 4) = rot;

            return m;
        }

        void dydx_dydq(const Eigen::Vector3d &xg, const Eigen::Vector3d &xa, double dt)
        {
            dydx_dydq_base(xg, xa, dt, true);
        }

        void increment(double dt, const Eigen::Vector3d &xg, const Eigen::Vector3d &xa)
        {
            const Eigen::Matrix<double, STATE_IN_DIM, 1> curr_state = get_f(xg, xa);

            // increment imu_world_pos.
            pos += curr_state.segment(VEL, 3) * dt;

            // increment xg.
            Eigen::Matrix4d A = quat_4x4_rot(xg, -dt, true);
            Eigen::Matrix3d R = utils::quat2rmat(A * curr_state.segment(ORI, 4));

            // increment imu_world_vel.
            Eigen::Vector3d T_ab = curr_state.segment(BAT, 3).asDiagonal() * xa - curr_state.segment(BAA, 3);
            vel += (R.transpose() * T_ab + curr_state.segment(GRAV, 3)) * dt;

            // increment imu_world_rot.
            prev_quat = rot;
            rot = A * curr_state.segment(ORI, 4);

            // BGA BAA mean reversion
            if (params->acc_process_noise_rev > 0.0)
                acc_bias = curr_state.segment(BAA, 3) * exp(-dt * params->acc_process_noise_rev);

            if (params->gyro_process_noise > 0.0)
                gyro_bias = curr_state.segment(BGA, 3) * exp(-dt * params->gyro_process_noise);
        }

        void predict(const Eigen::Vector3d &xg, const Eigen::Vector3d &xa, double dt, bool predict_state, bool prop_cov)
        {
            std::cout << "StatesInput File - Predict" << std::endl;
            if (predict_state)
                increment(dt, xg, xa);

            if (prop_cov)
            {
                std::cout << "StatesInput File - Propagate" << std::endl;
                noise_covariance_increment(dt);
                dydx_dydq(xg, xa, dt);
                P = dydx * P * dydx.transpose() + dydq * Q * dydq.transpose();
            }

            std::cout << "StatesInput File - Predict done" << std::endl;
        }
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
            P = Eigen::MatrixXd::Zero(INNER_DIM, INNER_DIM);
            initialize_state_covariance();
        }

        explicit StateOutput(PARAMETERS::Ptr &parameters)
            : BaseState(parameters),
              imu_acc(Eigen::Vector3d::Zero()),
              imu_gyro(Eigen::Vector3d::Zero())
        {
            dydx = Eigen::MatrixXd::Zero(INNER_DIM, INNER_DIM);
            dydq = Eigen::MatrixXd::Zero(INNER_DIM, Q_DIM);
            P = Eigen::MatrixXd::Zero(INNER_DIM, INNER_DIM);

            // initialize state convariance
            initialize_state_covariance();
            P.block(IMU_ACC, IMU_ACC, 3, 3) *= utils::square(params->init_bga_noise);
            P.block(IMU_GYRO, IMU_GYRO, 3, 3) *= utils::square(params->init_baa_noise);
        }

        // constructor definitions
        StateOutput(const StateOutput &b)
        {
            this->pos = b.pos;
            this->vel = b.vel;
            this->rot = b.rot;
            this->gyro_bias = b.gyro_bias;
            this->acc_bias = b.acc_bias;
            this->mult_bias = b.mult_bias;
            this->grav = b.grav;
            this->offset_T_L_I = b.offset_T_L_I;
            this->offset_R_L_I = b.offset_R_L_I;
            this->dydx = b.dydx;
            this->dydq = b.dydq;
            this->Q = b.Q;
            this->P = b.P;
            this->imu_acc = b.imu_acc;
            this->imu_gyro = b.imu_gyro;

            this->prev_quat = b.prev_quat;
        }

        StateOutput &operator=(const StateOutput &b)
        {
            this->pos = b.pos;
            this->vel = b.vel;
            this->rot = b.rot;
            this->gyro_bias = b.gyro_bias;
            this->acc_bias = b.acc_bias;
            this->mult_bias = b.mult_bias;
            this->grav = b.grav;
            this->offset_T_L_I = b.offset_T_L_I;
            this->offset_R_L_I = b.offset_R_L_I;
            this->dydx = b.dydx;
            this->dydq = b.dydq;
            this->Q = b.Q;
            this->P = b.P;
            this->prev_quat = b.prev_quat;
            this->imu_acc = b.imu_acc;
            this->imu_gyro = b.imu_gyro;
            return *this;
        }

        StateOutput operator+(const Eigen::Matrix<double, INNER_DIM, 1> &m)
        {
            StateOutput a;
            a.pos = this->pos + m.segment(POS, 3);
            a.vel = this->vel + m.segment(VEL, 3);

            Eigen::Vector4d ori_val = m.segment(ORI, 4);
            a.rot = increment_q(this->rot, ori_val);

            a.gyro_bias = this->gyro_bias + m.segment(BGA, 3);
            a.acc_bias = this->acc_bias + m.segment(BAA, 3);
            a.mult_bias = m.segment(BAT, 3);
            a.grav = this->grav + m.segment(GRAV, 3);
            a.offset_T_L_I = this->offset_T_L_I + m.segment(POS_LIDAR_IMU, 3);

            ori_val = m.segment(ORI_LIDAR_IMU, 4);
            a.offset_R_L_I = decrement_q(this->offset_R_L_I, ori_val);

            a.imu_acc = this->imu_acc + m.segment(IMU_ACC, 3);
            a.imu_gyro = this->imu_gyro + m.segment(IMU_GYRO, 3);

            a.dydx = this->dydx;
            a.dydq = this->dydq;
            a.Q = this->Q;
            a.P = this->P;
            a.prev_quat = this->prev_quat;

            return a;
        }

        StateOutput &operator+=(const Eigen::Matrix<double, INNER_DIM, 1> &m)
        {
            this->pos += m.segment(POS, 3);
            this->vel += m.segment(VEL, 3);

            Eigen::Vector4d ori_val = m.segment(ORI, 4);
            this->rot = increment_q(this->rot, ori_val);

            this->gyro_bias += m.segment(BGA, 3);
            this->acc_bias += m.segment(BAA, 3);
            this->mult_bias = m.segment(BAT, 3);
            this->grav += m.segment(GRAV, 3);
            this->offset_T_L_I += m.segment(POS_LIDAR_IMU, 3);

            ori_val = m.segment(ORI_LIDAR_IMU, 4);
            this->offset_R_L_I = decrement_q(this->offset_R_L_I, ori_val);

            this->imu_acc += m.segment(IMU_ACC, 3);
            this->imu_gyro += m.segment(IMU_GYRO, 3);

            return *this;
        }

        Eigen::Matrix<double, INNER_DIM, 1> get_f(const Eigen::Vector3d &xg, const Eigen::Vector3d &xa)
        {
            Eigen::Matrix<double, INNER_DIM, 1> m = Eigen::Matrix<double, INNER_DIM, 1>::Zero();

            // position
            m.segment(POS, 3) = vel;

            // velocity - using estimated as outputs
            Eigen::Matrix3d R = utils::quat2rmat(rot);
            m.segment(VEL, 3) = (R.transpose() * imu_acc + grav);

            // orientation
            m.segment(ORI, 4) = rot;

            return m;
        }

        void dydx_dydq(const Eigen::Vector3d &xg, const Eigen::Vector3d &xa, double dt)
        {
            dydx_dydq_base(xg, xa, dt, false);
        }

        void increment(double dt, const Eigen::Vector3d &xg, const Eigen::Vector3d &xa)
        {
            const Eigen::Matrix<double, INNER_DIM, 1> curr_state = get_f(xg, xa);

            // increment imu_world_pos.
            pos += curr_state.segment(VEL, 3) * dt;

            // increment xg.
            Eigen::Matrix4d A = quat_4x4_rot(xg, -dt, false);
            Eigen::Matrix3d R = utils::quat2rmat(A * curr_state.segment(ORI, 4));

            // increment imu_world_vel. -> should be bias free
            // Eigen::Vector3d T_ab = curr_state.segment(BAT, 3).asDiagonal() * xa - curr_state.segment(BAA, 3);
            // vel += (R.transpose() * T_ab + curr_state.segment(GRAV, 3)) * dt;
            vel += (R.transpose() * xa + curr_state.segment(GRAV, 3)) * dt;

            // increment imu_world_rot.
            prev_quat = rot;
            rot = A * curr_state.segment(ORI, 4);

            // BGA BAA mean reversion
            if (params->acc_process_noise_rev > 0.0)
                acc_bias = curr_state.segment(BAA, 3) * exp(-dt * params->acc_process_noise_rev);

            if (params->gyro_process_noise > 0.0)
                gyro_bias = curr_state.segment(BGA, 3) * exp(-dt * params->gyro_process_noise);

            // increase imu prediction by the bias
            imu_acc += curr_state.segment(BAA, 3) * dt;
            imu_gyro += curr_state.segment(BGA, 3) * dt;
        }

        void predict(const Eigen::Vector3d &xg, const Eigen::Vector3d &xa, double dt, bool predict_state, bool prop_cov)
        {
            // use previous prediction as input
            Eigen::Vector3d gyr = imu_gyro;
            Eigen::Vector3d acc = imu_acc;
            std::cout << "StatesOutput File - Predict" << std::endl;
            if (predict_state)
                increment(dt, gyr, acc);

            if (prop_cov)
            {
                std::cout << "StatesOutput File - Propagate" << std::endl;
                noise_covariance_increment(dt);
                dydx_dydq(gyr, acc, dt);
                P = dydx * P * dydx.transpose() + dydq * Q * dydq.transpose();
            }
        }

        Eigen::Vector3d imu_acc;  // estimated imu acceleration
        Eigen::Vector3d imu_gyro; // estimated imu gyro
    };
}
#endif