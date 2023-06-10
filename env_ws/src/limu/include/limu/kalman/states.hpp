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
    namespace NominalState
    {
        constexpr int POS = 0;
        constexpr int VEL = 3;
        constexpr int ORI = 6;
        constexpr int BGA = 10;
        constexpr int BAA = 13;
        constexpr int BAT = 16;
        constexpr int GRAV = 19;
        constexpr int POS_LIDAR_IMU = 22;
        constexpr int ORI_LIDAR_IMU = 25;
        constexpr int IMU_GYRO = 29;
        constexpr int IMU_ACC = 32;
        constexpr int LIDAR = 35;
        constexpr int h_matrix_col = 14;
    }

    namespace ErrorState
    {
        constexpr int POS = 0;
        constexpr int VEL = 3;
        constexpr int ORI = 6;
        constexpr int BGA = 9;
        constexpr int BAA = 12;
        constexpr int BAT = 15;
        constexpr int GRAV = 18;
        constexpr int POS_LIDAR_IMU = 21;
        constexpr int ORI_LIDAR_IMU = 24;
        constexpr int IMU_GYRO = 27;
        constexpr int IMU_ACC = 30;
        constexpr int LIDAR = 33;

        constexpr int Q_VEL = 0;
        constexpr int Q_ORI = 3;
        constexpr int Q_GYRO = 6;
        constexpr int Q_ACC = 9;
        constexpr int Q_BAT = 12;
        constexpr int Q_IMU_GYRO = 15;
        constexpr int Q_IMU_ACC = 18;
        constexpr int Q_DIM = 21;

        constexpr int h_matrix_col = 12;
    }

    // size of H jacobian from imu_measurement
    constexpr int h_imu_jacob_row = 6;
    constexpr int h_imu_jacob_col = 15;

    struct Parameters
    {
        typedef std::shared_ptr<Parameters> Ptr;
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

        double noise_process_BAA;
        double noise_process_BGA;

        double init_imu_acc_noise;
        double init_imu_gyro_noise;
        double imu_acc_output;
        double imu_gyro_output;
    };

    struct StateAttributes
    {
        StateAttributes() : pos(Eigen::Vector3d::Zero()),
                            vel(Eigen::Vector3d::Zero()),
                            gyro_bias(Eigen::Vector3d::Zero()),
                            acc_bias(Eigen::Vector3d::Zero()),
                            mult_bias(Eigen::Vector3d(1, 1, 1)),
                            grav(Eigen::Vector3d::Zero()),
                            offset_T_L_I(Eigen::Vector3d::Zero()),
                            imu_acc(Eigen::Vector3d::Zero()),
                            imu_gyro(Eigen::Vector3d::Zero()) {}

        explicit StateAttributes(
            const bool is_nominal)
            : pos(Eigen::Vector3d::Zero()),
              vel(Eigen::Vector3d::Zero()),
              gyro_bias(Eigen::Vector3d::Zero()),
              acc_bias(Eigen::Vector3d::Zero()),
              mult_bias(Eigen::Vector3d(1, 1, 1)),
              grav(Eigen::Vector3d::Zero()),
              offset_T_L_I(Eigen::Vector3d::Zero()),
              imu_acc(Eigen::Vector3d::Zero()),
              imu_gyro(Eigen::Vector3d::Zero())
        {
            if (is_nominal)
            {
                offset_R_L_I = Eigen::Vector4d(1, 0, 0, 0);
                rot = Eigen::Vector4d(1, 0, 0, 0);
            }
            else
            {
                offset_R_L_I = Eigen::Vector3d(0, 0, 0);
                rot = Eigen::Vector3d(0, 0, 0);
            }
        }

        void print_attributes()
        {
            std::cout << "--------- State Attributes ---------" << std::endl;
            std::cout << "Position (pos): " << pos.transpose() << std::endl;
            std::cout << "Velocity (vel): " << vel.transpose() << std::endl;
            std::cout << "Rotation (rot): " << rot.transpose() << std::endl;
            std::cout << "Gyroscope Bias (gyro_bias): " << gyro_bias.transpose() << std::endl;
            std::cout << "Accelerometer Bias (acc_bias): " << acc_bias.transpose() << std::endl;
            std::cout << "Multiplicative Bias (mult_bias): " << mult_bias.transpose() << std::endl;
            std::cout << "Gravity (grav): " << grav.transpose() << std::endl;
            std::cout << "Lidar to IMU Rotation (offset_T_L_I): " << offset_T_L_I.transpose() << std::endl;
            std::cout << "Lidar to IMU Translation (offset_R_L_I): " << offset_R_L_I.transpose() << std::endl;
            std::cout << "IMU Acceleration (imu_acc): " << imu_acc.transpose() << std::endl;
            std::cout << "IMU Gyro (imu_gyro): " << imu_gyro.transpose() << std::endl;
        }

        Eigen::Vector3d pos;          // estimated imu position wrt global
        Eigen::Vector3d vel;          // estimated imu velocity wrt global
        Eigen::VectorXd rot;          // estimated imu rotation wrt global
        Eigen::Vector3d gyro_bias;    // gyroscope bias
        Eigen::Vector3d acc_bias;     // accelerometer bias
        Eigen::Vector3d mult_bias;    // multiplicative bias
        Eigen::Vector3d grav;         // estimated gravity acceleration
        Eigen::Vector3d offset_T_L_I; // rotation from lidar frame L to imu frame I
        Eigen::VectorXd offset_R_L_I; // translation from lidar frame L to imu frame I
        Eigen::Vector3d imu_acc;      // estimated imu acceleration
        Eigen::Vector3d imu_gyro;     // estimated imu gyro
    };

    // building state
    class State
    {
    public:
        typedef std::shared_ptr<State> Ptr;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        // class construction
        State() : nominal_state(true), error_state(false), imu_as_input(false) {}

        explicit State(const std::string &loc)
            : params(std::make_shared<Parameters>()),
              nominal_state(true), error_state(false),
              imu_as_input(false)
        {
            setup_noise_constants(loc);

            if (imu_as_input)
            {
                Q = Eigen::MatrixXd::Zero(ErrorState::Q_DIM - 6, ErrorState::Q_DIM - 6);
                dydq = Eigen::MatrixXd::Zero(ErrorStateDim, ErrorState::Q_DIM - 6);
            }
            else
            {
                Q = Eigen::MatrixXd::Zero(ErrorState::Q_DIM, ErrorState::Q_DIM);
                dydq = Eigen::MatrixXd::Zero(ErrorStateDim, ErrorState::Q_DIM);
            }

            // matrices for calculations
            dydx = Eigen::MatrixXd::Zero(ErrorStateDim, ErrorStateDim);
            P = Eigen::MatrixXd::Zero(ErrorStateDim, ErrorStateDim);
            G = Eigen::MatrixXd::Identity(ErrorStateDim, ErrorStateDim);

            dydq_setup();
            initialize_state_covariance();
            true_state_jacobian_init();
        }

        State(const State &b)
        {
            this->nominal_state = b.nominal_state;
            this->error_state = b.error_state;
            this->ErrorStateDim = b.ErrorStateDim;
            this->NominalStateDim = b.NominalStateDim;

            this->dydx = b.dydx;
            this->dydq = b.dydq;
            this->Q = b.Q;
            this->P = b.P;
            this->R = b.R;
        }

        State &operator=(const State &b)
        {
            this->nominal_state = b.nominal_state;
            this->error_state = b.error_state;
            this->ErrorStateDim = b.ErrorStateDim;
            this->NominalStateDim = b.NominalStateDim;

            this->dydx = b.dydx;
            this->dydq = b.dydq;
            this->Q = b.Q;
            this->P = b.P;
            this->R = b.R;

            return *this;
        }

        State operator+(const Eigen::Matrix<double, -1, 1> &m)
        {
            // incrementing nominal_state by error_state
            State a(file_loc);

            a.nominal_state.pos = this->nominal_state.pos + m.segment(ErrorState::POS, 3);
            a.nominal_state.vel = this->nominal_state.vel + m.segment(ErrorState::VEL, 3);
            a.nominal_state.gyro_bias = this->nominal_state.gyro_bias + m.segment(ErrorState::BGA, 3);
            a.nominal_state.acc_bias = this->nominal_state.acc_bias + m.segment(ErrorState::BAA, 3);
            a.nominal_state.mult_bias = this->nominal_state.mult_bias + m.segment(ErrorState::BAT, 3);
            a.nominal_state.grav = this->nominal_state.grav + m.segment(ErrorState::GRAV, 3);
            a.nominal_state.offset_T_L_I = this->nominal_state.offset_T_L_I + m.segment(ErrorState::POS_LIDAR_IMU, 3);

            if (m.rows() == ErrorState::LIDAR || m.rows() == NominalState::LIDAR)
            {
                a.nominal_state.imu_acc = this->nominal_state.imu_acc + m.segment(ErrorState::IMU_ACC, 3);
                a.error_state.imu_acc = m.segment(ErrorState::IMU_ACC, 3);

                a.nominal_state.imu_gyro = this->nominal_state.imu_gyro + m.segment(ErrorState::IMU_GYRO, 3);
                a.error_state.imu_gyro = m.segment(ErrorState::IMU_GYRO, 3);
            }

            if (m.rows() == ErrorStateDim)
            {
                a.error_state.pos = m.segment(ErrorState::POS, 3);
                a.error_state.vel = m.segment(ErrorState::VEL, 3);

                a.nominal_state.rot = utils::dquat_left_multiply(
                    m.segment(ErrorState::ORI, 3), this->nominal_state.rot);
                a.error_state.rot = m.segment(ErrorState::ORI, 3);

                a.error_state.gyro_bias = m.segment(ErrorState::BGA, 3);
                a.error_state.acc_bias = m.segment(ErrorState::BAA, 3);
                a.error_state.mult_bias = m.segment(ErrorState::BAT, 3);
                a.error_state.grav = m.segment(ErrorState::BAT, 3);
                a.error_state.offset_T_L_I = m.segment(ErrorState::POS_LIDAR_IMU, 3);

                a.nominal_state.offset_R_L_I = utils::dquat_left_multiply(
                    m.segment(ErrorState::ORI_LIDAR_IMU, 3),
                    this->nominal_state.offset_R_L_I);
                a.error_state.offset_R_L_I = m.segment(ErrorState::ORI_LIDAR_IMU, 3);
            }
            else
            {
                a.nominal_state.rot = utils::quat_left_multiply(m.segment(ErrorState::ORI, 4), this->nominal_state.rot);
                a.nominal_state.offset_R_L_I = utils::quat_left_multiply(
                    m.segment(ErrorState::ORI_LIDAR_IMU, 4), this->nominal_state.offset_R_L_I);
                a.error_state = this->error_state;
            }

            return a;
        }

        State &operator+=(const Eigen::Matrix<double, -1, 1> &m)
        {
            // imu position, vel, ori
            nominal_state.pos += m.segment(ErrorState::POS, 3);
            nominal_state.vel += m.segment(ErrorState::VEL, 3);
            nominal_state.gyro_bias += m.segment(ErrorState::BGA, 3);
            nominal_state.acc_bias += m.segment(ErrorState::BAA, 3);
            nominal_state.mult_bias += m.segment(ErrorState::BAT, 3);
            nominal_state.grav += m.segment(ErrorState::GRAV, 3);
            nominal_state.offset_T_L_I += m.segment(ErrorState::POS_LIDAR_IMU, 3);

            if (m.rows() == ErrorState::LIDAR)
            {
                nominal_state.imu_acc += m.segment(ErrorState::IMU_ACC, 3);
                nominal_state.imu_gyro += m.segment(ErrorState::IMU_GYRO, 3);
            }

            if (m.rows() == ErrorStateDim)
            {
                nominal_state.rot = utils::dquat_left_multiply(
                    m.segment(ErrorState::ORI, 3), nominal_state.rot);

                nominal_state.offset_R_L_I = utils::dquat_left_multiply(
                    m.segment(ErrorState::ORI_LIDAR_IMU, 3),
                    nominal_state.offset_R_L_I);
            }
            else
            {
                nominal_state.rot = utils::quat_left_multiply(
                    m.segment(NominalState::ORI, 4), nominal_state.rot);

                nominal_state.offset_R_L_I = utils::quat_left_multiply(
                    m.segment(NominalState::ORI_LIDAR_IMU, 4), nominal_state.offset_R_L_I);
            }

            return *this;
        }

        /*................... Matrices and Initializations ...................*/
        void initialize_orientation(const Eigen::Vector3d &xa)
        {
            Eigen::Quaterniond orient = Eigen::Quaterniond::FromTwoVectors(-nominal_state.grav, xa);
            Eigen::Vector4d q = utils::quat2vec(orient);

            if (is_ground_vehicle)
                assert(std::abs(q[3]) <= 1e-6);

            nominal_state.rot = q;
        }

        void true_state_jacobian_init()
        {
            TSJ = Eigen::MatrixXd::Zero(NominalStateDim, ErrorStateDim);

            TSJ.block(NominalState::POS, ErrorState::POS, 3, 3).setIdentity(3, 3);
            TSJ.block(NominalState::VEL, ErrorState::VEL, 3, 3).setIdentity(3, 3);

            TSJ.block(NominalState::BGA, ErrorState::BGA, 3, 3).setIdentity(3, 3);
            TSJ.block(NominalState::BAA, ErrorState::BAA, 3, 3).setIdentity(3, 3);
            TSJ.block(NominalState::BAT, ErrorState::BAT, 3, 3).setIdentity(3, 3);

            TSJ.block(NominalState::GRAV, ErrorState::GRAV, 3, 3).setIdentity(3, 3);
            TSJ.block(NominalState::POS_LIDAR_IMU, ErrorState::POS_LIDAR_IMU, 3, 3).setIdentity(3, 3);

            if (!imu_as_input)
            {
                TSJ.block(NominalState::IMU_GYRO, ErrorState::IMU_GYRO, 3, 3).setIdentity(3, 3);
                TSJ.block(NominalState::IMU_ACC, ErrorState::IMU_ACC, 3, 3).setIdentity(3, 3);
            }

            // orientation derivations calculated when needed ->true_state_jacobian
        }

        void setup_noise_constants(const std::string &loc)
        {
            file_loc = loc;
            const YAML::Node node = YAML::LoadFile(loc);

            auto &kf_params = node["kalman_filter_params"];
            auto &state_covariance = kf_params["state_covariance"];
            auto &measurement_covariance = kf_params["measurement_covariance"];
            auto &common = node["common"];

            set_double_param(params->noise_scale, kf_params["noise_scale"]);
            params->noise_scale *= params->noise_scale;
            // imu noise parameters
            set_double_param(params->init_pos_noise, state_covariance["init_pos_noise"]);
            set_double_param(params->init_vel_noise, state_covariance["init_vel_noise"]);
            set_double_param(params->init_ori_noise, state_covariance["init_ori_noise"]);
            set_double_param(params->init_bga_noise, state_covariance["init_bga_noise"]);
            set_double_param(params->init_baa_noise, state_covariance["init_baa_noise"]);
            set_double_param(params->init_bat_noise, state_covariance["init_bat_noise"]);
            set_double_param(params->init_imu_acc_noise, state_covariance["init_imu_acc_noise"]);
            set_double_param(params->init_imu_gyro_noise, state_covariance["init_imu_gyro_noise"]);

            set_double_param(params->acc_process_noise, measurement_covariance["acc_process_noise"]);
            set_double_param(params->gyro_process_noise, measurement_covariance["gyro_process_noise"]);
            set_double_param(params->acc_process_noise_rev, measurement_covariance["acc_process_noise_rev"]);
            set_double_param(params->gyro_process_noise_rev, measurement_covariance["gyro_process_noise_rev"]);
            set_double_param(params->imu_acc_output, measurement_covariance["imu_acc_output"]);
            set_double_param(params->imu_gyro_output, measurement_covariance["imu_gyro_output"]);
            set_double_param(params->noise_process_BAA, measurement_covariance["noise_process_baa"]);
            set_double_param(params->noise_process_BGA, measurement_covariance["noise_process_bga"]);

            set_bool_param(is_ground_vehicle, kf_params["is_ground_vehicle"]);
            set_bool_param(imu_as_input, common["use_imu_as_input"]);

            if (imu_as_input)
            {
                ErrorStateDim = ErrorState::LIDAR - 6;
                NominalStateDim = NominalState::LIDAR - 6;
            }
            else
            {
                ErrorStateDim = ErrorState::LIDAR;
                NominalStateDim = NominalState::LIDAR;
            }
        }

        void dydq_setup()
        {
            dydq.setZero();

            dydq.block(ErrorState::VEL, ErrorState::Q_VEL, 3, 3).setIdentity(3, 3);
            dydq.block(ErrorState::ORI, ErrorState::Q_ORI, 3, 3).setIdentity(3, 3);
            dydq.block(ErrorState::BAA, ErrorState::Q_ACC, 3, 3).setIdentity(3, 3);

            dydq.block(ErrorState::BGA, ErrorState::Q_GYRO, 3, 3).setIdentity(3, 3);
            dydq.block(ErrorState::BAT, ErrorState::Q_BAT, 3, 3).setIdentity(3, 3);

            if (!imu_as_input)
            {
                dydq.block(ErrorState::IMU_ACC, ErrorState::Q_IMU_ACC, 3, 3).setIdentity(3, 3);
                dydq.block(ErrorState::IMU_GYRO, ErrorState::Q_IMU_GYRO, 3, 3).setIdentity(3, 3);
            }
        }

        void initialize_state_covariance()
        {
            // imu - global extrinsic
            P.block(ErrorState::POS, ErrorState::POS, 3, 3).setIdentity(3, 3) *= utils::square(params->init_pos_noise);
            P.block(ErrorState::VEL, ErrorState::VEL, 3, 3).setIdentity(3, 3) *= utils::square(params->init_vel_noise);
            P.block(ErrorState::ORI, ErrorState::ORI, 3, 3).setIdentity(3, 3);

            if (is_ground_vehicle)
                // this is because ground vehicle restricted to x_y plane movements
                P(ErrorState::ORI + 2, ErrorState::ORI + 2) = 0;

            P.block(ErrorState::ORI, ErrorState::ORI, 3, 3) *= utils::square(params->init_ori_noise);

            // imu bias initialization
            P.block(ErrorState::BGA, ErrorState::BGA, 3, 3).setIdentity(3, 3) *= utils::square(params->init_bga_noise);
            P.block(ErrorState::BAA, ErrorState::BAA, 3, 3).setIdentity(3, 3) *= utils::square(params->init_baa_noise);
            P.block(ErrorState::BAT, ErrorState::BAT, 3, 3).setIdentity(3, 3) *= utils::square(params->init_bat_noise);
            P.block(ErrorState::GRAV, ErrorState::GRAV, 3, 3).setIdentity(3, 3) *= utils::square(params->init_bat_noise);

            // relativity to lidar estimates
            P.block(ErrorState::POS_LIDAR_IMU, ErrorState::POS_LIDAR_IMU, 3, 3).setIdentity(3, 3) *= utils::square(params->init_pos_noise);
            P.block(ErrorState::ORI_LIDAR_IMU, ErrorState::ORI_LIDAR_IMU, 3, 3).setIdentity(3, 3) *= utils::square(params->init_ori_noise);

            if (!imu_as_input)
            {
                P.block(ErrorState::IMU_ACC, ErrorState::IMU_ACC, 3, 3).setIdentity(3, 3) *= utils::square(params->init_imu_acc_noise);
                P.block(ErrorState::IMU_GYRO, ErrorState::IMU_GYRO, 3, 3).setIdentity(3, 3) *= utils::square(params->init_imu_gyro_noise);
            }

            P *= params->noise_scale;
        }

        /*................... Process Jacobian ...................*/
        void dydx_setup(double dt)
        {
            dydx.setZero();

            // Position derivatives
            dydx.block(ErrorState::POS, ErrorState::POS, 3, 3).setIdentity(3, 3);
            dydx.block(ErrorState::POS, ErrorState::VEL, 3, 3).setIdentity(3, 3) *= dt;

            // Velocity derivatives
            dydx.block(ErrorState::VEL, ErrorState::VEL, 3, 3).setIdentity(3, 3);
            dydx.block(ErrorState::VEL, ErrorState::ORI, 3, 3) = R.transpose() * utils::skew_matrix(corrected_acc) * dt;

            if (imu_as_input)
            {
                dydx.block(ErrorState::VEL, ErrorState::BAA, 3, 3) = -R.transpose() * dt;
                dydx.block(ErrorState::VEL, ErrorState::BAT, 3, 3) = R.transpose() * measured_acc.asDiagonal() * dt;
            }
            else
                dydx.block(ErrorState::VEL, ErrorState::IMU_ACC, 3, 3) = -R.transpose() * dt;

            dydx.block(ErrorState::VEL, ErrorState::GRAV, 3, 3).setIdentity(3, 3) *= dt;

            // Orientation derivatives
            dydx.block(ErrorState::ORI, ErrorState::ORI, 3, 3) = A; // check nominal equation

            if (imu_as_input)
                dydx.block(ErrorState::ORI, ErrorState::BGA, 3, 3).setIdentity(3, 3) *= -dt;
            else
                dydx.block(ErrorState::ORI, ErrorState::IMU_GYRO, 3, 3).setIdentity(3, 3) *= -dt;

            // Bias derivative
            dydx.block(ErrorState::BGA, ErrorState::BGA, 3, 3).setIdentity(3, 3);
            dydx.block(ErrorState::BAA, ErrorState::BAA, 3, 3).setIdentity(3, 3);
            dydx.block(ErrorState::BAT, ErrorState::BAT, 3, 3).setIdentity(3, 3);

            // Predicted values
            if (!imu_as_input)
            {
                dydx.block(ErrorState::IMU_ACC, ErrorState::IMU_ACC, 3, 3).setIdentity(3, 3);
                dydx.block(ErrorState::IMU_GYRO, ErrorState::IMU_GYRO, 3, 3).setIdentity(3, 3);
            }

            // Gravity derivatives
            dydx.block(ErrorState::GRAV, ErrorState::GRAV, 3, 3).setIdentity(3, 3);

            // Extrinsics derivatives
            dydx.block(ErrorState::POS_LIDAR_IMU, ErrorState::POS_LIDAR_IMU, 3, 3).setIdentity(3, 3);
            dydx.block(ErrorState::ORI_LIDAR_IMU, ErrorState::ORI_LIDAR_IMU, 3, 3).setIdentity(3, 3);
        }

        /*................... Noise Jacobian ...................*/
        void process_noise_covariance(double dt)
        {

            /// need to comeback here and split drift only ones away
            double gyro_drift = 1.0;
            {
                if (params->noise_process_BGA > 0.0)
                {
                    const double qc = utils::square(params->noise_process_BGA);
                    const double theta = params->gyro_process_noise_rev;
                    gyro_drift = params->noise_scale * qc;
                    if (theta > 0.0)
                        gyro_drift *= (1 - exp(-2 * dt * theta)) / (2 * theta);
                }
            }

            double acc_drift = 1.0;
            {
                if (params->noise_process_BAA > 0.0)
                {
                    const double qc = utils::square(params->noise_process_BAA);
                    const double theta = params->acc_process_noise_rev;
                    acc_drift = params->noise_scale * qc;
                    if (theta > 0.0)
                        acc_drift *= (1 - exp(-2 * dt * theta)) / (2 * theta);
                }
            }

            // velocity noise
            R = utils::vec4d_to_rmat(nominal_state.rot);
            Q.block(ErrorState::Q_VEL, ErrorState::Q_VEL, 3, 3) = R.transpose() * acc_drift * dt;

            // orientation noise
            Q.block(ErrorState::Q_ORI, ErrorState::Q_ORI, 3, 3).setIdentity(3, 3) *= gyro_drift * dt;

            Q.block(ErrorState::Q_GYRO, ErrorState::Q_GYRO, 3, 3).setIdentity(3, 3) *= gyro_drift;
            Q.block(ErrorState::Q_ACC, ErrorState::Q_ACC, 3, 3).setIdentity(3, 3) *= acc_drift;
            Q.block(ErrorState::Q_BAT, ErrorState::Q_BAT, 3, 3).setIdentity(3, 3) *= utils::square(params->init_bat_noise) * params->noise_scale * dt;

            if (!imu_as_input)
            {
                // imu_gyro
                if (params->gyro_process_noise > 0.0)
                {
                    double qc = utils::square(params->gyro_process_noise);
                    double theta = params->gyro_process_noise_rev;
                    gyro_drift = params->noise_scale * qc;
                    if (theta > 0.0)
                        gyro_drift *= (1 - exp(-2 * dt * theta)) / (2 * theta);

                    Q.block(ErrorState::Q_IMU_GYRO, ErrorState::Q_IMU_GYRO, 3, 3).setIdentity(3, 3) *= gyro_drift;
                }

                // imu_acc
                if (params->acc_process_noise > 0.0)
                {
                    double qc = utils::square(params->acc_process_noise);
                    double theta = params->acc_process_noise_rev;
                    acc_drift = params->noise_scale * qc;
                    if (theta > 0.0)
                        acc_drift *= (1 - exp(-2 * dt * theta)) / (2 * theta);

                    Q.block(ErrorState::Q_IMU_ACC, ErrorState::Q_IMU_ACC, 3, 3).setIdentity(3, 3) *= acc_drift;
                }
            }
        }

        StateAttributes update_nominal(double dt)
        {
            // position
            StateAttributes nom(true);
            nom.pos = nominal_state.vel * dt;

            // rotation : assumed readings in local coordinates
            nom.rot = utils::dquat_left_multiply(corrected_gyro, nominal_state.rot, dt);
            A = utils::extract_ori_ori(corrected_gyro, dt);
            R = utils::vec4d_to_rmat(nom.rot);

            // velocity
            nom.vel = (R.transpose() * corrected_acc + nominal_state.grav) * dt;

            // Use noise to scale readings
            double acc_scale = params->noise_process_BAA > 0.0 ? exp(-dt * params->acc_process_noise_rev) : 1.0;
            double gyro_scale = params->noise_process_BGA > 0.0 ? exp(-dt * params->gyro_process_noise_rev) : 1.0;

            // for bias and imu values.. we scale the nominal state by some exponential noise.. and add noise to the
            // error state as well. From HYBVIO Paper
            nom.acc_bias = nominal_state.acc_bias * acc_scale;
            nom.gyro_bias = nominal_state.gyro_bias * gyro_scale;

            if (!imu_as_input)
            {
                acc_scale = params->acc_process_noise > 0.0 ? exp(-dt * params->acc_process_noise_rev) : 1.0;
                gyro_scale = params->gyro_process_noise > 0.0 ? exp(-dt * params->gyro_process_noise_rev) : 1.0;
                nom.imu_acc = nominal_state.imu_acc * acc_scale;
                nom.imu_gyro = nominal_state.imu_gyro * gyro_scale;
            }

            return nom;
        }

        void nominal_kinematics(double dt)
        {
            const auto n_m = update_nominal(dt);

            // increment imu_world_pos.
            nominal_state.pos += n_m.pos;

            // update velocity
            nominal_state.vel += n_m.vel;

            // update orientation
            nominal_state.rot = n_m.rot;

            // update bias
            nominal_state.acc_bias = n_m.acc_bias;
            nominal_state.gyro_bias = n_m.gyro_bias;

            // update gyro
            if (!imu_as_input)
            {
                nominal_state.imu_acc = n_m.imu_acc;
                nominal_state.imu_gyro = n_m.imu_gyro;
            }
        }

        void predict(const Eigen::Vector3d &xg, const Eigen::Vector3d &xa, double dt, bool predict_state, bool prop_cov)
        {
            if (imu_as_input)
            {
                corrected_acc = nominal_state.mult_bias.asDiagonal() * xa - nominal_state.acc_bias;
                measured_acc = xa;

                corrected_gyro = xg - nominal_state.gyro_bias;
                measured_gyro = xg;
            }
            else
            {
                corrected_acc = nominal_state.imu_acc;
                corrected_gyro = nominal_state.imu_gyro;
            }

            if (predict_state)
                nominal_kinematics(dt);

            if (prop_cov)
            {
                process_noise_covariance(dt);
                dydx_setup(dt);
                P = dydx * P * dydx.transpose() + dydq * Q * dydq.transpose();
            }
        }

        void update_G(const Eigen::Vector3d &d_theta, const Eigen::Vector3d &d_offset_theta)
        {
            G.setIdentity();
            G.block(ErrorState::ORI, ErrorState::ORI, 3, 3) += 0.5 * utils::skew_matrix(d_theta);
            G.block(ErrorState::ORI_LIDAR_IMU, ErrorState::ORI_LIDAR_IMU, 3, 3) += 0.5 * utils::skew_matrix(d_offset_theta);

            P = G * P * G.transpose();
        }

        /*................... Lidar Extraction Functions ...................*/
        Eigen::MatrixXd P_measurement_props_lidar()
        {
            /*
             * To return state properties affected by measurement and used in the iterative step.
             * properies are: pos, rot, offset_trans_lidar_imu, offset_rot_lidar_imu
             * note the offset_trans_lidar_imu to be put in quaternion form
             */

            Eigen::MatrixXd props = Eigen::MatrixXd::Zero(ErrorStateDim, ErrorState::h_matrix_col);

            // imu_world parameters
            props.block(0, 0, ErrorStateDim, 3) = P.block(0, ErrorState::POS, ErrorStateDim, 3);
            props.block(0, 3, ErrorStateDim, 3) = P.block(0, ErrorState::ORI, ErrorStateDim, 3);

            // lidar imu positions
            props.block(0, 6, ErrorStateDim, 6) = P.block(0, ErrorState::POS_LIDAR_IMU, ErrorStateDim, 6);

            return props;
        }

        Eigen::MatrixXd row_properties_lidar(const Eigen::MatrixXd &PHT)
        {
            /*
             * Given the PHT matrix want to extract relevant information from the column
             * properies are: pos, rot, offset_trans_lidar_imu, offset_rot_lidar_imu
             * note the offset_trans_lidar_imu to be put in quaternion form
             */

            const int num_meas = PHT.cols();
            const int half_col = ErrorState::h_matrix_col / 2;

            Eigen::MatrixXd props = Eigen::MatrixXd::Zero(ErrorState::h_matrix_col, num_meas);
            // imu_world parameters
            props.block(0, 0, 3, num_meas) = PHT.block(ErrorState::POS, 0, 3, num_meas);
            props.block(3, 0, 3, num_meas) = PHT.block(ErrorState::ORI, 0, 3, num_meas);

            // lidar imu positions
            props.block(half_col, 0, half_col, num_meas) = PHT.block(ErrorState::POS_LIDAR_IMU, 0, half_col, num_meas);

            return props;
        }

        /*................... Imu Extraction Functions ...................*/
        Eigen::MatrixXd P_measurement_props_imu()
        {
            /*
             * To return state properties affected by measurement and used in the iterative step.
             * properies are: gyro_bias, acc_bias, mult_bias, imu_gyro, imu_acc
             */

            Eigen::MatrixXd props = Eigen::MatrixXd::Zero(ErrorStateDim, h_imu_jacob_col);
            props.block(0, 0, ErrorStateDim, 3) = P.block(0, ErrorState::BGA, ErrorStateDim, 3);
            props.block(0, 3, ErrorStateDim, 3) = P.block(0, ErrorState::BAA, ErrorStateDim, 3);
            props.block(0, 6, ErrorStateDim, 3) = P.block(0, ErrorState::BAT, ErrorStateDim, 3);
            props.block(0, 9, ErrorStateDim, 3) = P.block(0, ErrorState::IMU_GYRO, ErrorStateDim, 3);
            props.block(0, 12, ErrorStateDim, 3) = P.block(0, ErrorState::IMU_ACC, ErrorStateDim, 3);

            return props;
        }

        Eigen::MatrixXd row_properties_imu(const Eigen::MatrixXd &PHT)
        {
            /*
             * Given the PHT matrix want to extract relevant information from the column
             * properies are: rotation, gyro_bias, acc_bias, mult_bias, imu_gyro, imu_acc
             */

            const int num_meas = PHT.cols();

            Eigen::MatrixXd props = Eigen::MatrixXd::Zero(h_imu_jacob_col, num_meas);
            props.block(0, 0, 3, num_meas) = PHT.block(ErrorState::BGA, 0, 3, num_meas);
            props.block(3, 0, 3, num_meas) = PHT.block(ErrorState::BAA, 0, 3, num_meas);
            props.block(6, 0, 3, num_meas) = PHT.block(ErrorState::BAT, 0, 3, num_meas);
            props.block(9, 0, 3, num_meas) = PHT.block(ErrorState::IMU_GYRO, 0, 3, num_meas);
            props.block(12, 0, 3, num_meas) = PHT.block(ErrorState::IMU_ACC, 0, 3, num_meas);

            return props;
        }

        /*................... True state jacobian ...................*/
        void true_state_jacobian()
        {
            // Relevant sections to be multiplied with the measurement jacobian
            TSJ.block(NominalState::ORI, ErrorState::ORI, 4, 3) = utils::ori_ts_jacobian(nominal_state.rot);
            TSJ.block(NominalState::ORI_LIDAR_IMU, ErrorState::ORI_LIDAR_IMU, 4, 3) = utils::ori_ts_jacobian(nominal_state.offset_R_L_I);
        }

        // extract relevant imu information from the jacobian of the true state
        Eigen::MatrixXd imu_ts_jacobian()
        {
            true_state_jacobian();
            Eigen::MatrixXd props = Eigen::MatrixXd::Zero(h_imu_jacob_col, h_imu_jacob_col);

            props.block(0, 0, 3, 3) = TSJ.block(NominalState::BGA, ErrorState::BGA, 3, 3);
            props.block(3, 3, 3, 3) = TSJ.block(NominalState::BAA, ErrorState::BAA, 3, 3);
            props.block(6, 6, 3, 3) = TSJ.block(NominalState::BAT, ErrorState::BAT, 3, 3);
            props.block(9, 9, 3, 3) = TSJ.block(NominalState::IMU_GYRO, ErrorState::IMU_GYRO, 3, 3);
            props.block(12, 12, 3, 3) = TSJ.block(NominalState::IMU_ACC, ErrorState::IMU_ACC, 3, 3);
            return props;
        }

        // extract relevant lidar information from the jacobian of the true state
        Eigen::MatrixXd lidar_ts_jacobian()
        {
            true_state_jacobian();
            Eigen::MatrixXd props = Eigen::MatrixXd::Zero(NominalState::h_matrix_col, ErrorState::h_matrix_col);

            props.block(0, 0, 3, 3) = TSJ.block(NominalState::POS, ErrorState::POS, 3, 3);
            props.block(3, 3, 4, 3) = TSJ.block(NominalState::ORI, ErrorState::ORI, 4, 3);
            props.block(7, 6, 3, 3) = TSJ.block(NominalState::POS_LIDAR_IMU, ErrorState::POS_LIDAR_IMU, 3, 3);
            props.block(10, 9, 4, 3) = TSJ.block(NominalState::ORI_LIDAR_IMU, ErrorState::ORI_LIDAR_IMU, 4, 3);

            return props;
        }
        // getters and setters
        Sophus::SE3d get_L_I_pose()
        {
            return utils::get_pose(nominal_state.offset_R_L_I, nominal_state.offset_T_L_I);
        }

        Sophus::SE3d get_IMU_pose()
        {
            return utils::get_pose(nominal_state.rot, nominal_state.pos);
        }

        void print_nominal_attrbiutes()
        {
            nominal_state.print_attributes();
        }

        double get_noise_scale()
        {
            return params->noise_scale;
        }

    public:
        StateAttributes nominal_state, error_state;
        Eigen::MatrixXd P; // state covariance
    private:
        Parameters::Ptr params;
        Eigen::MatrixXd Q, G; // process covariance
        Eigen::MatrixXd dydx; // used in predict step
        Eigen::MatrixXd dydq; // for noise covariance
        Eigen::MatrixXd TSJ;  // true state jacobian
        Eigen::Matrix3d R, A;
        Eigen::Vector3d corrected_acc, corrected_gyro, measured_gyro, measured_acc;
        int ErrorStateDim, NominalStateDim;
        bool is_ground_vehicle, imu_as_input;
        std::string file_loc;
    };

}
#endif