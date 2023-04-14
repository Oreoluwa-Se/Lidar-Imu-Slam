#ifndef SENSOR_INIT_HPP
#define SENSOR_INIT_HPP

#include "common.hpp"
#include "ceres/ceres.h"

// inertial and lidar initialization
typedef Eigen::Matrix3d M3D;
typedef Eigen::Vector3d V3D;
const V3D STD_GRAV = V3D(0, 0, -gravity);

struct CalibState
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    M3D rot_end;
    V3D ang_vel;
    V3D linear_vel;
    V3D ang_acc;
    V3D linear_acc;
    double timeStamp;

    CalibState()
    {
        rot_end = rot_end.setIdentity();
        ang_vel = ang_vel.setZero();
        linear_vel = V3D::Zero();
        ang_acc = V3D::Zero();
        linear_acc = V3D::Zero();
        timeStamp = 0.0;
    };

    CalibState(const V3D &omg, const V3D &acc, const double &timestamp)
        : rot_end(rot_end.setIdentity()),
          ang_vel(omg), linear_vel(V3D::Zero()),
          ang_acc(V3D::Zero()), linear_acc(acc),
          timeStamp(timestamp) {}

    CalibState(const M3D &rot, const V3D &omg, const V3D &lin_vel, const double &timestamp)
        : rot_end(rot), ang_vel(omg), linear_vel(lin_vel),
          ang_acc(V3D::Zero()), linear_acc(V3D::Zero()),
          timeStamp(timestamp) {}

    CalibState(const CalibState &b)
    {
        this->rot_end = b.rot_end;
        this->ang_vel = b.ang_vel;
        this->ang_acc = b.ang_acc;
        this->linear_vel = b.linear_vel;
        this->linear_acc = b.linear_acc;
        this->timeStamp = b.timeStamp;
    };

    CalibState operator*(const double &coeff)
    {
        CalibState a;
        a.ang_vel = this->ang_vel * coeff;
        a.ang_acc = this->ang_acc * coeff;
        a.linear_vel = this->linear_vel * coeff;
        a.linear_acc = this->linear_acc * coeff;
        return a;
    };

    CalibState &operator+=(const CalibState &b)
    {
        this->ang_vel += b.ang_vel;
        this->ang_acc += b.ang_acc;
        this->linear_vel += b.linear_vel;
        this->linear_acc += b.linear_acc;
        return *this;
    };

    CalibState &operator-=(const CalibState &b)
    {
        this->ang_vel -= b.ang_vel;
        this->ang_acc -= b.ang_acc;
        this->linear_vel -= b.linear_vel;
        this->linear_acc -= b.linear_acc;
        return *this;
    };

    CalibState &operator=(const CalibState &b)
    {
        this->ang_vel = b.ang_vel;
        this->ang_acc = b.ang_acc;
        this->linear_vel = b.linear_vel;
        this->linear_acc = b.linear_acc;
        this->timeStamp = b.timeStamp;
        return *this;
    };
};

struct Angular_Vel_Cost_only_Rot
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Angular_Vel_Cost_only_Rot(const V3D &IMU_ang_vel_, const V3D &Lidar_ang_vel_)
        : IMU_ang_vel(IMU_ang_vel_), Lidar_ang_vel(Lidar_ang_vel_) {}

    template <typename T>
    bool operator()(const T *q, T *residual) const
    {
        const Eigen::Quaternion<T> q_LI(q);
        const Eigen::Matrix<T, 3, 1> IMU_ang_vel_T = IMU_ang_vel.cast<T>();
        const Eigen::Matrix<T, 3, 1> Lidar_ang_vel_T = Lidar_ang_vel.cast<T>();

        Eigen::Matrix<T, 3, 1> resi = q_LI.toRotationMatrix() * Lidar_ang_vel_T - IMU_ang_vel_T;
        residual[0] = resi[0];
        residual[1] = resi[1];
        residual[2] = resi[2];
        return true;
    }

    static ceres::CostFunction *Create(const V3D &IMU_ang_vel_, const V3D &Lidar_ang_vel_)
    {
        return (new ceres::AutoDiffCostFunction<Angular_Vel_Cost_only_Rot, 3, 4>(
            new Angular_Vel_Cost_only_Rot(IMU_ang_vel_, Lidar_ang_vel_)));
    }

    V3D IMU_ang_vel;
    V3D Lidar_ang_vel;
};

struct Angular_Vel_Cost
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Angular_Vel_Cost(
        V3D IMU_ang_vel_, V3D IMU_ang_acc_,
        V3D Lidar_ang_vel_, double deltaT_LI_)
        : IMU_ang_vel(IMU_ang_vel_), IMU_ang_acc(IMU_ang_acc_),
          Lidar_ang_vel(Lidar_ang_vel_), deltaT_LI(deltaT_LI_) {}

    template <typename T>
    bool operator()(const T *q, const T *b_g, const T *t, T *residual) const
    {
        // Known parameters used for Residual Construction
        Eigen::Matrix<T, 3, 1> IMU_ang_vel_T = IMU_ang_vel.template cast<T>();
        Eigen::Matrix<T, 3, 1> IMU_ang_acc_T = IMU_ang_acc.template cast<T>();
        Eigen::Matrix<T, 3, 1> Lidar_ang_vel_T = Lidar_ang_vel.template cast<T>();
        T deltaT_LI_T{static_cast<T>(deltaT_LI)};

        // Unknown Parameters, needed to be estimated
        Eigen::Quaternion<T> q_LI{q[0], q[1], q[2], q[3]};
        Eigen::Matrix<T, 3, 3> R_LI = q_LI.toRotationMatrix(); // Rotation
        Eigen::Matrix<T, 3, 1> bias_g{b_g[0], b_g[1], b_g[2]}; // Bias of gyroscope
        T td{t[0]};                                            // Time lag (IMU wtr Lidar)

        // Residual
        Eigen::Matrix<T, 3, 1> resi = R_LI * Lidar_ang_vel_T - IMU_ang_vel_T - (deltaT_LI_T + td) * IMU_ang_acc_T + bias_g;
        residual[0] = resi[0];
        residual[1] = resi[1];
        residual[2] = resi[2];

        return true;
    }

    static ceres::CostFunction *Create(
        const V3D &IMU_ang_vel_, const V3D &IMU_ang_acc_,
        const V3D &Lidar_ang_vel_, const double deltaT_LI_)
    {
        return (new ceres::AutoDiffCostFunction<Angular_Vel_Cost, 3, 4, 3, 1>(
            new Angular_Vel_Cost(IMU_ang_vel_, IMU_ang_acc_, Lidar_ang_vel_, deltaT_LI_)));
    }

    V3D IMU_ang_vel;
    V3D IMU_ang_acc;
    V3D Lidar_ang_vel;
    double deltaT_LI;
};

struct Linear_acc_Cost
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Linear_acc_Cost(CalibState LidarState_, M3D R_LI_, V3D IMU_linear_acc_) : LidarState(LidarState_), R_LI(R_LI_), IMU_linear_acc(IMU_linear_acc_) {}

    template <typename T>
    bool operator()(const T *q, const T *b_a, const T *trans, const T *mult_bias, T *residual) const
    {
        // Known parameters used for Residual Construction
        Eigen::Matrix<T, 3, 3> R_LL0_T = LidarState.rot_end.cast<T>();
        Eigen::Matrix<T, 3, 3> R_LI_T_transpose = R_LI.transpose().cast<T>();
        Eigen::Matrix<T, 3, 1> IMU_linear_acc_T = IMU_linear_acc.cast<T>();
        Eigen::Matrix<T, 3, 1> Lidar_linear_acc_T = LidarState.linear_acc.cast<T>();

        // Unknown Parameters, needed to be estimated
        Eigen::Quaternion<T> q_GL0{q[0], q[1], q[2], q[3]};
        Eigen::Matrix<T, 3, 3> R_GL0 = q_GL0.toRotationMatrix();   // Rotation from Gravitational to First Lidar frame
        Eigen::Matrix<T, 3, 1> bias_aL{b_a[0], b_a[1], b_a[2]};    // Bias of Linear acceleration
        Eigen::Matrix<T, 3, 1> T_IL{trans[0], trans[1], trans[2]}; // Translation of I-L (IMU wtr Lidar)
        Eigen::Matrix<T, 3, 3> Mult_Bias_L;                        // multiplicative bias for acc correction.
        Mult_Bias_L.setZero();
        Mult_Bias_L.diagonal() << mult_bias[0], mult_bias[1], mult_bias[2];

        // Residual Construction
        M3D Lidar_omg_SKEW, Lidar_angacc_SKEW;
        Lidar_omg_SKEW << utils::skew_matrix(LidarState.ang_vel);
        Lidar_angacc_SKEW << utils::skew_matrix(LidarState.ang_acc);
        M3D Jacob_trans = Lidar_omg_SKEW * Lidar_omg_SKEW + Lidar_angacc_SKEW;
        Eigen::Matrix<T, 3, 3> Jacob_trans_T = Jacob_trans.cast<T>();

        Eigen::Matrix<T, 3, 1> resi = R_LL0_T * R_LI_T_transpose * Mult_Bias_L * IMU_linear_acc_T - R_LL0_T * bias_aL + R_GL0 * STD_GRAV - Lidar_linear_acc_T - R_LL0_T * Jacob_trans_T * T_IL;

        residual[0] = resi[0];
        residual[1] = resi[1];
        residual[2] = resi[2];
        return true;
    }

    static ceres::CostFunction *Create(const CalibState LidarState_, const M3D R_LI_, const V3D IMU_linear_acc_)
    {
        return (new ceres::AutoDiffCostFunction<Linear_acc_Cost, 4, 4, 3, 3, 3>(
            new Linear_acc_Cost(LidarState_, R_LI_, IMU_linear_acc_)));
    }

    CalibState LidarState;
    M3D R_LI;
    V3D IMU_linear_acc;
};

class SensorInit
{
public:
    typedef std::shared_ptr<SensorInit> Ptr;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    SensorInit(){};

    void zero_phase_filt(
        const std::deque<CalibState> &signal_in,
        std::deque<CalibState> &signal_out);

    void LI_Initialization(double orig_odom_freq, double &timediff_imu_wrt_lidar, const double &move_start_time);
    void xcorr_temporal_init(const double &odom_freq);
    void IMU_time_compensate(const double &lag_time, const bool &is_discard);
    void downsample_interpolate_imu(const double &move_start_time);
    void push_IMU_CalibState(const V3D &omg, const V3D &acc, const double &timestamp);
    void push_Lidar_CalibState(const M3D &rot, const V3D &omg, const V3D &lin_vel, const double &timestamp);
    void normalize_acc(std::deque<CalibState> &signal_in);
    void set_IMU_state(const std::deque<CalibState> &IMU_states);
    void set_Lidar_state(const std::deque<CalibState> &Lidar_states);
    void set_states_2nd_filter(const std::deque<CalibState> &IMU_states, const std::deque<CalibState> &Lidar_states);
    void solve_Rot_bias_gyro(double &timediff_imu_wrt_lidar);
    void solve_trans_biasacc_grav();
    void acc_interpolate();
    void solve_Rotation_only();
    void cut_sequence_tail();
    void central_diff();
    bool data_sufficiency_assess(Eigen::MatrixXd &Jacobian_rot, int &frame_num, V3D &lidar_omg, double &orig_odom_freq);
    void print_initialization_result(
        double &time_L_I, M3D &R_L_I, V3D &p_L_I, V3D &bias_g,
        V3D &bias_a, V3D &grav, V3D &bias_at);

    std::deque<CalibState> get_IMU_state()
    {
        return IMU_state_group;
    }

    std::deque<CalibState> get_Lidar_state()
    {
        return Lidar_state_group;
    }

    inline double get_lag_time_1()
    {
        return time_lag_1;
    }

    inline double get_lag_time_2()
    {
        return time_lag_2;
    }

    inline double get_total_time_lag()
    {
        return time_delay_IMU_wtr_Lidar;
    }

    inline V3D get_Grav_L0()
    {
        return Grav_L0;
    }

    inline M3D get_R_LI()
    {
        return Rot_Lidar_wrt_IMU;
    }

    inline V3D get_T_LI()
    {
        return Trans_Lidar_wrt_IMU;
    }

    inline V3D get_gyro_bias()
    {
        return gyro_bias;
    }

    inline V3D get_acc_bias()
    {
        return acc_bias;
    }

    inline V3D get_mult_bias()
    {
        return mult_bias;
    }

    void clear_imu_buffer()
    {
        IMU_state_group_ALL.clear();
    }

public:
    double data_accum_length;

private:
    struct Butterworth // removes high frequency noise
    {
        // Coefficients of 6 order butterworth low pass filter, omega = 0.15
        double Coeff_b[7] = {0.000076, 0.000457, 0.001143, 0.001524, 0.0011, 0.000457, 0.000076};
        double Coeff_a[7] = {1.0000, -4.182389, 7.491611, -7.313596, 4.089349, -1.238525, 0.158428};
        int Coeff_size = 7;
        int extend_num = 0;
    };

    // functions
    void Butter_filt(const std::deque<CalibState> &signal_in, std::deque<CalibState> &signal_out);

    // holders and attributes
    std::deque<CalibState> IMU_state_group;
    std::deque<CalibState> Lidar_state_group;
    std::deque<CalibState> IMU_state_group_ALL;

    /// Parameters needed to be calibrated
    M3D Rot_Grav_wrt_Init_Lidar;     // Rotation from inertial frame G to initial Lidar frame L_0
    V3D Grav_L0;                     // Gravity vector in the initial Lidar frame L_0
    M3D Rot_Lidar_wrt_IMU;           // Rotation from Lidar frame L to IMU frame I
    V3D Trans_Lidar_wrt_IMU;         // Translation from Lidar frame L to IMU frame I
    V3D gyro_bias;                   // gyro bias
    V3D acc_bias;                    // acc bias
    V3D mult_bias;                   // mult bias
    double time_delay_IMU_wtr_Lidar; //(Soft) time delay between IMU and Lidar = time_lag_1 + time_lag_2
    double time_lag_1;               // Time offset estimated by cross-correlation
    double time_lag_2;               // Time offset estimated by unified optimization
    int lag_IMU_wtr_Lidar;           // positive: timestamp of IMU is larger than that of LiDAR
};

#endif
