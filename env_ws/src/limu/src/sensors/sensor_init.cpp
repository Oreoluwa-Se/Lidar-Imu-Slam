#include "limu/sensors/sensor_init.hpp"
#include "tbb/parallel_for.h"
#include <tbb/task_scheduler_init.h>

namespace
{
    const int tail_trim_dim = 20;
    // allowed time to the move_start_time - used in the downsampling step
    const double init_time_allowance = 3.0;
}
void SensorInit::LI_Initialization(
    int &orig_odom_freq, int &cut_frame_num,
    double &timediff_imu_wrt_lidar, const double &move_start_time)
{
    // downsample and interpolate imu time
    downsample_interpolate_imu(move_start_time);
    IMU_time_compensate(0.0, true);

    std::deque<CalibState> IMU_after_zero_phase;
    std::deque<CalibState> Lidar_after_zero_phase;

    // filter passing to reduce noise
    zero_phase_filt(get_IMU_state(), IMU_after_zero_phase);
    normalize_acc(IMU_after_zero_phase);
    zero_phase_filt(get_Lidar_state(), Lidar_after_zero_phase);
    set_IMU_state(IMU_after_zero_phase);
    set_Lidar_state(Lidar_after_zero_phase);

    // Align size of two sequences
    cut_sequence_tail();

    // calculate the time lag
    xcorr_temporal_init(orig_odom_freq * cut_frame_num);
    IMU_time_compensate(get_lag_time_1(), false);

    central_diff();

    // filter passing to reduce noise
    std::deque<CalibState> IMU_after_2nd_zero_phase;
    std::deque<CalibState> Lidar_after_2nd_zero_phase;
    zero_phase_filt(get_IMU_state(), IMU_after_2nd_zero_phase);
    zero_phase_filt(get_Lidar_state(), Lidar_after_2nd_zero_phase);
    set_states_2nd_filter(IMU_after_2nd_zero_phase, Lidar_after_2nd_zero_phase);

    // .............| functions |...............
    solve_Rotation_only();
    solve_Rot_bias_gyro(timediff_imu_wrt_lidar);
    acc_interpolate();

    solve_trans_biasacc_grav();
    double time_L_I = timediff_imu_wrt_lidar + time_delay_IMU_wtr_Lidar;
    print_initialization_result(time_L_I, Rot_Lidar_wrt_IMU, Trans_Lidar_wrt_IMU, gyro_bias, acc_bias, Grav_L0, mult_bias);

    std::cout << "============================================================\n\n";
    std::cout << "[Initialization] Lidar IMU initialization done.\n\n";
    std::cout << "============================================================\n\n";
}

void SensorInit::Butter_filt(const std::deque<CalibState> &signal_in, std::deque<CalibState> &signal_out)
{
    Butterworth butter;
    const auto &coeff_size = butter.Coeff_size;
    butter.extend_num = 10 * (coeff_size - 1);

    signal_out.resize(signal_in.size() + 2 * butter.extend_num);

    // extending signal -> using repeated signals
    for (int i = 0; i < butter.extend_num; ++i)
    {
        signal_out[i] = signal_in.front();
        signal_out[signal_out.size() - i - 1] = signal_in.back();
    }
    std::copy(signal_in.begin(), signal_in.end(), signal_out.begin() + butter.extend_num);

    const auto &coeff_b = butter.Coeff_b;
    const auto &coeff_a = butter.Coeff_a;
    // applying filter
    tbb::parallel_for(
        tbb::blocked_range<int>(coeff_size - 1, signal_out.size()),
        [&](const tbb::blocked_range<int> &range)
        {
            for (int i = range.begin(); i != range.end(); ++i)
            {
                CalibState temp_state;
                for (int j = 0; j < coeff_size; ++j)
                {
                    temp_state += signal_out[i - j] * coeff_b[j];
                }
                for (int j = 1; j < coeff_size; ++j)
                {
                    temp_state -= signal_out[i - j] * coeff_a[j];
                }

                signal_out[i - coeff_size + 1] = temp_state;
            }
        },
        tbb::auto_partitioner());
}

void SensorInit::zero_phase_filt(const std::deque<CalibState> &signal_in, std::deque<CalibState> &signal_out)
{
    std::deque<CalibState> sig_out1;
    Butter_filt(signal_in, sig_out1);

    std::deque<CalibState> sig_rev(sig_out1);
    std::reverse(sig_rev.begin(), sig_rev.end()); // Reverse the elements

    Butter_filt(sig_rev, signal_out);
    std::reverse(signal_out.begin(), signal_out.end()); // Reverse the elements
}

void SensorInit::cut_sequence_tail()
{
    std::deque<CalibState>::iterator imu_iter, lidar_iter;
    // trim elements from tail of sequence
    if (IMU_state_group.size() > 3 * tail_trim_dim)
    {
        imu_iter = IMU_state_group.end();
        lidar_iter = Lidar_state_group.end();
        std::advance(imu_iter, -tail_trim_dim);
        std::advance(lidar_iter, -tail_trim_dim);
        IMU_state_group.erase(imu_iter, IMU_state_group.end());
        Lidar_state_group.erase(lidar_iter, Lidar_state_group.end());
    }
    imu_iter = IMU_state_group.begin();
    lidar_iter = Lidar_state_group.begin();
    while (lidar_iter->timeStamp < imu_iter->timeStamp)
    {
        Lidar_state_group.pop_front();
        ++lidar_iter;
    }

    while (lidar_iter->timeStamp > std::next(imu_iter)->timeStamp)
    {
        IMU_state_group.pop_front();
        ++imu_iter;
    }

    const size_t final_size = std::min(IMU_state_group.size(), Lidar_state_group.size());
    IMU_state_group.resize(final_size);
    Lidar_state_group.resize(final_size);
}

void SensorInit::normalize_acc(std::deque<CalibState> &signal_in)
{
    V3D mean_acc(0, 0, 0);

    for (int i = 1; i < 10; i++)
    {
        mean_acc += (signal_in[i].linear_acc - mean_acc) / i;
    }

    for (int i = 0; i < signal_in.size(); i++)
    {
        signal_in[i].linear_acc = signal_in[i].linear_acc / mean_acc.norm() * gravity;
    }
}

void SensorInit::IMU_time_compensate(const double &lag_time, const bool &is_discard)
{
    if (is_discard)
    {
        // Discard first 10 Lidar estimations and corresponding IMU measurements due to long time interval
        Lidar_state_group.erase(Lidar_state_group.begin(), Lidar_state_group.begin() + 10);
        IMU_state_group.erase(IMU_state_group.begin(), IMU_state_group.begin() + 10);
    }

    // Subtract lag_time from IMU timestamps
    std::transform(
        IMU_state_group.begin(), IMU_state_group.end() - 1, IMU_state_group.begin(),
        [lag_time](CalibState &state)
        { state.timeStamp -= lag_time; });

    // Find index of first Lidar state with timestamp >= first IMU state timestamp
    auto first_lidar_it = std::find_if(
        Lidar_state_group.begin(), Lidar_state_group.end(),
        [&](const CalibState &state)
        { return state.timeStamp >= IMU_state_group.front().timeStamp; });
    auto lidar_offset = std::distance(Lidar_state_group.begin(), first_lidar_it);

    // Find index of first IMU state with timestamp > first Lidar state timestamp
    auto first_imu_it = std::lower_bound(
        IMU_state_group.begin(), IMU_state_group.end(),
        Lidar_state_group.front().timeStamp,
        [](const CalibState &state, const double &timestamp)
        { return state.timeStamp < timestamp; });
    auto imu_offset = std::distance(IMU_state_group.begin(), first_imu_it);

    // Erase excess elements to align sequence sizes
    auto num_elements = std::min(Lidar_state_group.size() - lidar_offset, IMU_state_group.size() - imu_offset);
    auto lidar_begin = std::next(Lidar_state_group.begin(), lidar_offset);
    auto imu_begin = std::next(IMU_state_group.begin(), imu_offset);

    std::deque<CalibState> Lidar_state_group_new(lidar_begin, lidar_begin + num_elements);
    std::deque<CalibState> IMU_state_group_new(imu_begin, imu_begin + num_elements);

    Lidar_state_group = std::move(Lidar_state_group_new);
    IMU_state_group = std::move(IMU_state_group_new);
}

void SensorInit::downsample_interpolate_imu(const double &move_start_time)
{
    double time_diff = move_start_time - init_time_allowance;
    auto it_imu_start = std::lower_bound(
        IMU_state_group_ALL.begin(), IMU_state_group_ALL.end(),
        time_diff, [](const CalibState &state, double t)
        { return state.timeStamp < t; });
    IMU_state_group_ALL.erase(IMU_state_group_ALL.begin(), it_imu_start);

    auto it_lidar_start = std::lower_bound(
        Lidar_state_group.begin(), Lidar_state_group.end(),
        time_diff, [](const CalibState &state, double t)
        { return state.timeStamp < t; });
    Lidar_state_group.erase(Lidar_state_group.begin(), it_lidar_start);

    // Original IMU measurements
    std::vector<CalibState> IMU_states_all_origin(IMU_state_group_ALL.begin(), IMU_state_group_ALL.end() - 1);

    int mean_filt_size = 2;
    // Reduce noise effect by taking region average
    for (auto it = IMU_state_group_ALL.begin() + mean_filt_size; it != IMU_state_group_ALL.end() - mean_filt_size; ++it)
    {
        V3D acc_real = V3D::Zero();
        for (int k = -mean_filt_size; k <= mean_filt_size; ++k)
        {
            acc_real += (IMU_states_all_origin[it - IMU_state_group_ALL.begin() + k].linear_acc - acc_real) / (k + mean_filt_size + 1);
        }
        it->linear_acc = acc_real;
    }

    // now we downsample
    for (const auto &lidar_state : Lidar_state_group)
    {
        // find lowebound imu time less than the current lidar time
        auto it_imu = std::lower_bound(
            IMU_state_group_ALL.begin(), IMU_state_group_ALL.end(),
            lidar_state.timeStamp, [](const CalibState &state, double t)
            { return state.timeStamp < t; });

        if (it_imu == IMU_state_group_ALL.end())
            continue;

        if (it_imu == IMU_state_group_ALL.begin())
            push_IMU_CalibState(it_imu->ang_vel, it_imu->linear_acc, lidar_state.timeStamp);
        else
        {
            auto it_imu_prev = it_imu - 1;
            double delta_t = it_imu->timeStamp - it_imu_prev->timeStamp;
            double delta_t_right = it_imu->timeStamp - lidar_state.timeStamp;
            double s = delta_t_right / delta_t;
            CalibState IMU_state_interpolation;
            IMU_state_interpolation.ang_vel = s * it_imu_prev->ang_vel + (1 - s) * it_imu->ang_vel;
            IMU_state_interpolation.linear_acc = s * it_imu_prev->linear_acc + (1 - s) * it_imu->linear_acc;
            push_IMU_CalibState(IMU_state_interpolation.ang_vel, IMU_state_interpolation.linear_acc,
                                lidar_state.timeStamp);
        }
    }
}

void SensorInit::push_IMU_CalibState(const V3D &omg, const V3D &acc, const double &timestamp)
{
    IMU_state_group.emplace_back(omg, acc, timestamp);
}

void SensorInit::set_IMU_state(const std::deque<CalibState> &IMU_states)
{
    IMU_state_group.assign(IMU_states.begin(), IMU_states.end() - 1);
}

void SensorInit::set_Lidar_state(const std::deque<CalibState> &Lidar_states)
{
    Lidar_state_group.assign(Lidar_states.begin(), Lidar_states.end() - 1);
}

void SensorInit::xcorr_temporal_init(const double &odom_freq)
{
    /*
        This code performs a zero-centered cross-correlation between two sets of data:
        IMU angular velocity and LiDAR angular velocity.
        The purpose of this cross-correlation is to find the time delay between the two sets of data,
        which can be used to synchronize the measurements and improve the accuracy of the sensor fusion.
    */

    int N = IMU_state_group.size();
    // Calculate mean value of IMU and LiDAR angular velocity - switching to normal mean
    double mean_IMU_ang_vel = 0, mean_LiDAR_ang_vel = 0;
    for (int i = 0; i < N; i++)
    {
        mean_IMU_ang_vel += IMU_state_group[i].ang_vel.norm();
        mean_LiDAR_ang_vel += Lidar_state_group[i].ang_vel.norm();
    }
    mean_IMU_ang_vel /= N;
    mean_LiDAR_ang_vel /= N;

    // Calculate zero-centered cross correlation
    double max_corr = -DBL_MAX;
    for (int lag = -N + 1; lag < N; lag++)
    {
        double corr = 0;
        for (int i = 0; i < N; i++)
        {
            int j = i + lag;
            if (j < 0 || j > N - 1)
                continue;
            else
            {
                corr += (IMU_state_group[i].ang_vel.norm() - mean_IMU_ang_vel) *
                        (Lidar_state_group[j].ang_vel.norm() - mean_LiDAR_ang_vel); // Zero-centered cross correlation
            }
        }

        if (corr > max_corr)
        {
            max_corr = corr;
            lag_IMU_wtr_Lidar = -lag;
        }
    }

    time_lag_1 = lag_IMU_wtr_Lidar / odom_freq;
    std::cout << "Max Cross-correlation: IMU lag wtr Lidar : " << -lag_IMU_wtr_Lidar << std::endl;
}

void SensorInit::central_diff()
{
    std::cout << "Central difference estimation." << std::endl;
    auto it_IMU_state = IMU_state_group.begin() + 1;
    auto it_Lidar_state = Lidar_state_group.begin() + 1;
    for (; it_IMU_state != IMU_state_group.end() - 2 && it_Lidar_state != Lidar_state_group.end() - 2; it_IMU_state++, it_Lidar_state++)
    {
        auto last_imu = it_IMU_state - 1;
        auto next_imu = it_IMU_state + 1;
        double dt_imu = next_imu->timeStamp - last_imu->timeStamp;
        it_IMU_state->ang_acc = (next_imu->ang_vel - last_imu->ang_vel) / dt_imu;
        std::cout << std::setprecision(12) << it_IMU_state->ang_vel.transpose() << " "
                  << it_IMU_state->ang_vel.norm() << " "
                  << it_IMU_state->linear_acc.transpose() << " "
                  << it_IMU_state->ang_acc.transpose() << " "
                  << it_IMU_state->timeStamp << std::endl;

        auto last_lidar = it_Lidar_state - 1;
        auto next_lidar = it_Lidar_state + 1;
        double dt_lidar = next_lidar->timeStamp - last_lidar->timeStamp;
        it_Lidar_state->ang_acc = (next_lidar->ang_vel - last_lidar->ang_vel) / dt_lidar;
        it_Lidar_state->linear_acc = (next_lidar->linear_vel - last_lidar->linear_vel) / dt_lidar;
        std::cout << std::setprecision(12) << it_Lidar_state->ang_vel.transpose() << " " << it_Lidar_state->ang_vel.norm()
                  << " " << (it_Lidar_state->linear_acc - STD_GRAV).transpose() << " "
                  << it_Lidar_state->ang_acc.transpose() << " " << it_Lidar_state->timeStamp << std::endl;
    }
}

void SensorInit::set_states_2nd_filter(const std::deque<CalibState> &IMU_states, const std::deque<CalibState> &Lidar_states)
{
    for (int i = 0; i < IMU_state_group.size(); i++)
    {
        IMU_state_group[i].ang_acc = IMU_states[i].ang_acc;
        Lidar_state_group[i].ang_acc = Lidar_states[i].ang_acc;
        Lidar_state_group[i].linear_acc = Lidar_states[i].linear_acc;
    }
}

void SensorInit::solve_Rotation_only()
{
    double R_LI_quat[4];
    R_LI_quat[0] = 1;
    R_LI_quat[1] = 0;
    R_LI_quat[2] = 0;
    R_LI_quat[3] = 0;

    ceres::LocalParameterization *quatParam = new ceres::QuaternionParameterization();
    ceres::Problem problem_rot;
    problem_rot.AddParameterBlock(R_LI_quat, 4, quatParam);

    for (int i = 0; i < IMU_state_group.size(); i++)
    {
        M3D Lidar_angvel_skew;
        Lidar_angvel_skew << utils::skew_matrix(Lidar_state_group[i].ang_vel);
        problem_rot.AddResidualBlock(
            Angular_Vel_Cost_only_Rot::Create(IMU_state_group[i].ang_vel, Lidar_state_group[i].ang_vel),
            nullptr, R_LI_quat);
    }
    /* --------------------------------- */
    ceres::Solver::Options options_quat;
    ceres::Solver::Summary summary_quat;
    ceres::Solve(options_quat, &problem_rot, &summary_quat);
    Eigen::Quaterniond q_LI(R_LI_quat[0], R_LI_quat[1], R_LI_quat[2], R_LI_quat[3]);
    Rot_Lidar_wrt_IMU = q_LI.matrix();
}

void SensorInit::solve_Rot_bias_gyro(double &timediff_imu_wrt_lidar)
{
    Eigen::Quaterniond quat(Rot_Lidar_wrt_IMU);
    double R_LI_quat[4];
    R_LI_quat[0] = quat.w();
    R_LI_quat[1] = quat.x();
    R_LI_quat[2] = quat.y();
    R_LI_quat[3] = quat.z();

    double bias_g[3]; // Initial value of gyro bias
    bias_g[0] = 0;
    bias_g[1] = 0;
    bias_g[2] = 0;

    double time_lag2 = 0; // Second time lag (IMU wtr Lidar)

    ceres::LocalParameterization *quatParam = new ceres::QuaternionParameterization();
    ceres::Problem problem_ang_vel;

    problem_ang_vel.AddParameterBlock(R_LI_quat, 4, quatParam);
    problem_ang_vel.AddParameterBlock(bias_g, 3);

    for (int i = 0; i < IMU_state_group.size(); i++)
    {
        double deltaT = Lidar_state_group[i].timeStamp - IMU_state_group[i].timeStamp;
        problem_ang_vel.AddResidualBlock(
            Angular_Vel_Cost::Create(
                IMU_state_group[i].ang_vel,
                IMU_state_group[i].ang_acc,
                Lidar_state_group[i].ang_vel,
                deltaT),
            nullptr,
            R_LI_quat,
            bias_g,
            &time_lag2);
    }

    ceres::Solver::Options options_quat;
    ceres::Solver::Summary summary_quat;
    ceres::Solve(options_quat, &problem_ang_vel, &summary_quat);

    Eigen::Quaterniond q_LI(R_LI_quat[0], R_LI_quat[1], R_LI_quat[2], R_LI_quat[3]);
    Rot_Lidar_wrt_IMU = q_LI.matrix();

    gyro_bias = V3D(bias_g[0], bias_g[1], bias_g[2]);

    time_lag_2 = time_lag2;
    time_delay_IMU_wtr_Lidar = time_lag_1 + time_lag_2;
    // std::cout << "Total time delay (IMU wtr Lidar): " << time_delay_IMU_wtr_Lidar + timediff_imu_wrt_lidar << " s" << std::endl;
    // std::cout << "Using LIO: SUBTRACT this value from IMU timestamp" << std::endl
    //           << "           or ADD this value to LiDAR timestamp." << std::endl
    //           << std::endl;

    // The second temporal compensation
    IMU_time_compensate(get_lag_time_2(), false);

    // for (int i = 0; i < Lidar_state_group.size(); i++)
    // {
    //     fout_after_rot << setprecision(12) << (Rot_Lidar_wrt_IMU * Lidar_state_group[i].ang_vel + gyro_bias).transpose()
    //                    << " " << Lidar_state_group[i].timeStamp << endl;
    // }
}

void SensorInit::acc_interpolate()
{
    // Interpolation to get acc_I(t_L)
    for (int i = 1; i < Lidar_state_group.size() - 1; i++)
    {
        double deltaT = Lidar_state_group[i].timeStamp - IMU_state_group[i].timeStamp;
        if (deltaT > 0)
        {
            double DeltaT = IMU_state_group[i + 1].timeStamp - IMU_state_group[i].timeStamp;
            double s = deltaT / DeltaT;
            IMU_state_group[i].linear_acc = s * IMU_state_group[i + 1].linear_acc +
                                            (1 - s) * IMU_state_group[i].linear_acc;
            IMU_state_group[i].timeStamp += deltaT;
        }
        else
        {
            double DeltaT = IMU_state_group[i].timeStamp - IMU_state_group[i - 1].timeStamp;
            double s = -deltaT / DeltaT;
            IMU_state_group[i].linear_acc = s * IMU_state_group[i - 1].linear_acc +
                                            (1 - s) * IMU_state_group[i].linear_acc;
            IMU_state_group[i].timeStamp += deltaT;
        }
    }
}

void SensorInit::solve_trans_biasacc_grav()
{
    M3D Rot_Init = M3D::Zero();
    Rot_Init.diagonal() = V3D(1, 1, 1);
    Eigen::Quaterniond quat(Rot_Init);
    double R_GL0_quat[4];
    R_GL0_quat[0] = quat.w();
    R_GL0_quat[1] = quat.x();
    R_GL0_quat[2] = quat.y();
    R_GL0_quat[3] = quat.z();

    double bias_aL[3]; // Initial value of acc bias
    bias_aL[0] = 0;
    bias_aL[1] = 0;
    bias_aL[2] = 0;

    double Trans_IL[3]; // Initial value of Translation of IL (IMU with respect to Lidar)
    Trans_IL[0] = 0.0;
    Trans_IL[1] = 0.0;
    Trans_IL[2] = 0.0;

    double MultBias_TL[3]; // Initial value of multiplicative bias to correct acc
    MultBias_TL[0] = 0.0;
    MultBias_TL[1] = 0.0;
    MultBias_TL[2] = 0.0;

    ceres::LocalParameterization *quatParam = new ceres::QuaternionParameterization();
    ceres::Problem problem_acc;

    problem_acc.AddParameterBlock(R_GL0_quat, 4, quatParam);
    problem_acc.AddParameterBlock(bias_aL, 3);
    problem_acc.AddParameterBlock(Trans_IL, 3);
    problem_acc.AddParameterBlock(MultBias_TL, 3);

    // Jacobian of acc_bias, gravity, Translation, mult_bias
    int Jaco_size = 4 * Lidar_state_group.size();
    Eigen::MatrixXd Jacobian(Jaco_size, 12);
    Jacobian.setZero();

    // Jacobian of Translation
    Eigen::MatrixXd Jaco_Trans(Jaco_size, 3);
    Jaco_Trans.setZero();

    for (int i = 0; i < IMU_state_group.size(); i++)
    {
        problem_acc.AddResidualBlock(
            Linear_acc_Cost::Create(Lidar_state_group[i],
                                    Rot_Lidar_wrt_IMU,
                                    IMU_state_group[i].linear_acc),
            nullptr,
            R_GL0_quat,
            bias_aL,
            Trans_IL,
            MultBias_TL);

        // wrt bias_acc
        Jacobian.block<3, 3>(4 * i, 0) = -Lidar_state_group[i].rot_end;
        // wrt gravity - initial value included
        Jacobian.block<3, 3>(4 * i, 3) << utils::skew_matrix(STD_GRAV);
        M3D omg_skew, angacc_skew;

        // wrt to point between imu_lidar
        omg_skew << utils::skew_matrix(Lidar_state_group[i].ang_vel);
        angacc_skew << utils::skew_matrix(Lidar_state_group[i].ang_acc);
        M3D Jaco_trans_i = omg_skew * omg_skew + angacc_skew;
        Jacobian.block<3, 3>(4 * i, 6) = Jaco_trans_i;

        // wrt to mult_bias
        Jacobian.block<3, 3>(4 * i, 9) = -Lidar_state_group[i].rot_end;

        // translation jacobian
        Jaco_Trans.block<3, 3>(4 * i, 0) = Jaco_trans_i;
    }

    for (int index = 0; index < 3; ++index)
    {
        problem_acc.SetParameterUpperBound(bias_aL, index, 0.01);
        problem_acc.SetParameterLowerBound(bias_aL, index, -0.01);

        problem_acc.SetParameterUpperBound(MultBias_TL, index, 0.01);
        problem_acc.SetParameterLowerBound(MultBias_TL, index, -0.01);
    }

    ceres::Solver::Options options_acc;
    ceres::Solver::Summary summary_acc;
    ceres::Solve(options_acc, &problem_acc, &summary_acc);

    Eigen::Quaterniond q_GL0(R_GL0_quat[0], R_GL0_quat[1], R_GL0_quat[2], R_GL0_quat[3]);
    Rot_Grav_wrt_Init_Lidar = q_GL0.matrix();
    Grav_L0 = Rot_Grav_wrt_Init_Lidar * STD_GRAV;

    V3D bias_a_Lidar(bias_aL[0], bias_aL[1], bias_aL[2]);
    acc_bias = Rot_Lidar_wrt_IMU * bias_a_Lidar;

    V3D Mult_Bias_Lidar(MultBias_TL[0], MultBias_TL[1], MultBias_TL[2]);
    mult_bias = Rot_Lidar_wrt_IMU * Mult_Bias_Lidar;

    V3D Trans_IL_vec(Trans_IL[0], Trans_IL[1], Trans_IL[2]);
    Trans_Lidar_wrt_IMU = -Rot_Lidar_wrt_IMU * Trans_IL_vec;

    for (int i = 0; i < IMU_state_group.size(); i++)
    {
        V3D acc_I = Lidar_state_group[i].rot_end * Rot_Lidar_wrt_IMU.transpose() * IMU_state_group[i].linear_acc -
                    Lidar_state_group[i].rot_end * bias_a_Lidar;
        V3D acc_L = Lidar_state_group[i].linear_acc +
                    Lidar_state_group[i].rot_end * Jaco_Trans.block<3, 3>(3 * i, 0) * Trans_IL_vec - Grav_L0;
        // fout_acc_cost << setprecision(10) << acc_I.transpose() << " " << acc_L.transpose() << " "
        //               << IMU_state_group[i].timeStamp << " " << Lidar_state_group[i].timeStamp << endl;
    }

    // M3D Hessian_Trans = Jaco_Trans.transpose() * Jaco_Trans;
    // Eigen::EigenSolver<M3D> es_trans(Hessian_Trans);
    // M3D EigenValue_mat_trans = es_trans.pseudoEigenvalueMatrix();
    // M3D EigenVec_mat_trans = es_trans.pseudoEigenvectors();
}

void SensorInit::print_initialization_result(double &time_L_I, M3D &R_L_I, V3D &p_L_I, V3D &bias_g, V3D &bias_a, V3D &grav, V3D &bias_at)
{
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "[Init Result] ";
    std::cout << "Rotation LiDAR to IMU    = " << utils::rotation_matrix_to_euler_angles(R_L_I).transpose() * 57.3 << " deg" << std::endl;
    std::cout << "[Init Result] ";
    std::cout << "Translation LiDAR to IMU = " << p_L_I.transpose() << " m" << std::endl;
    std::cout << "[Init Result] ";
    std::cout << "Time Lag IMU to LiDAR    = " << std::setprecision(8) << time_L_I << " s" << std::endl;
    std::cout << "[Init Result] ";
    std::cout << "Bias of Gyroscope        = " << bias_g.transpose() << " rad/s" << std::endl;
    std::cout << "[Init Result] ";
    std::cout << "Bias of Accelerometer    = " << bias_a.transpose() << " m/s^2" << std::endl;
    std::cout << "[Init Result] ";
    std::cout << "Multiplicative Bias of Accelerometer    = " << bias_at.transpose() << std::endl;
    std::cout << "[Init Result] ";
    std::cout << "Gravity in World Frame   = " << grav.transpose() << " m/s^2" << std::endl
              << std::endl;
}
