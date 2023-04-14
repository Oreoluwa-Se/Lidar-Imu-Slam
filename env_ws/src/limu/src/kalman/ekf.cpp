#include "limu/kalman/ekf.hpp"
#include "limu/kalman/helper.hpp"
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <type_traits>
#include <iomanip>

namespace
{
    auto weight = [](double res_sq, double th)
    { return utils::square(th) / utils::square(th + res_sq); };

    void printEigenMatrix(const Eigen::MatrixXd &matrix, const std::string &tableName)
    {
        std::cout << "Table Name: " << tableName << std::endl;
        std::cout << "Dimensions: " << matrix.rows() << " x " << matrix.cols() << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        for (int i = 0; i < matrix.rows(); ++i)
        {
            for (int j = 0; j < matrix.cols(); ++j)
            {
                std::cout << std::setw(10) << std::setprecision(4) << matrix(i, j) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "----------------------------------------" << std::endl;
    }

    void printMatchedPoints(const std::vector<Eigen::Vector3d> &src, const std::vector<Eigen::Vector3d> &targ)
    {
        std::cout << "Matched Points" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << std::setw(15) << "Source X" << std::setw(15) << "Source Y" << std::setw(15) << "Source Z";
        std::cout << std::setw(15) << "Target X" << std::setw(15) << "Target Y" << std::setw(15) << "Target Z" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        for (int i = 0; i < src.size(); ++i)
        {
            std::cout << std::setw(15) << std::setprecision(4) << src[i][0] << std::setw(15) << std::setprecision(4) << src[i][1] << std::setw(15) << std::setprecision(4) << src[i][2];
            std::cout << std::setw(15) << std::setprecision(4) << targ[i][0] << std::setw(15) << std::setprecision(4) << targ[i][1] << std::setw(15) << std::setprecision(4) << targ[i][2] << std::endl;
        }
        std::cout << "----------------------------------------" << std::endl;
    }

    Eigen::Matrix<double, 3, 4> jacobian_wrt_quat(const Eigen::Vector4d &quat_v, const Eigen::Vector3d &p)
    {
        /* General function for calculation Jacobian wrt quaternion */
        const double w = quat_v[0];
        const Eigen::Vector3d v(quat_v[1], quat_v[2], quat_v[3]);

        const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

        Eigen::Matrix<double, 3, 4> int_mat;
        // quaternion solutions
        int_mat.block<3, 1>(0, 0) = 2 * (w * p + v.cross(p));
        int_mat.block<3, 3>(0, 1) = 2 * (v.transpose() * p * I + v * p.transpose() - p * v.transpose() - w * utils::skew_matrix(p));

        return int_mat;
    }
}

namespace odometry
{

    utils::Vec3dVector EKF::pre_initialization(frame::LidarImuInit::Ptr &meas, bool init)
    {
        // convert to vector3d points
        original_points = utils::pointcloud2eigen(*(meas->processed_frame));

        // convert from lidar-world to imu-world frame.
        utils::Vec3dVector eigen_frame = points_body_to_world(original_points);

        // downsample map points.
        utils::Vec3_Vec3Tuple downsampled_frame = icp_ptr->voxelize(eigen_frame);

        // downsampled version used in mapping source used to calculate points.
        curr_downsampled_frame = std::get<1>(downsampled_frame); // this is what is used for mapping

        // if map is empty return
        if (init)
            icp_ptr->update_map(curr_downsampled_frame, icp_ptr->init_guess);

        return std::get<0>(downsampled_frame);
    }

    void EKF::initialize_map(frame::LidarImuInit::Ptr &meas)
    {
        const auto src = pre_initialization(meas, true);

        init_map = true;
    }

    /*........................................... LIDAR H MATRIX MODEL ...................................................*/
    template <typename StateType>
    void EKF::h_model_lidar(StateType &state, dyn_share_modified<double> &data, frame::LidarImuInit::Ptr &meas)
    {
        auto source = pre_initialization(meas);
        utils::Vec3dVector orig_points_copy(original_points.begin(), original_points.end());
        original_points.clear();
        utils::Vec3dVector src, targ;
        double th;
        {
            // Get motion prediction and adaptive threshold
            const double sigma = icp_ptr->get_adaptive_threshold();
            th = sigma / 3.0;

            // for each downsampled point find corresponding closest point from the map
            const auto result = icp_ptr->local_map.get_correspondences(source, 3.0 * sigma);
            const utils::Vec3dVector &src_temp = std::get<0>(result);
            const utils::Vec3dVector &targ_temp = std::get<1>(result);

            std::cout << "Pre filtered points" << std::endl;
            printMatchedPoints(src_temp, targ_temp);
            int idx = 0;
            for (const auto &targ_p : targ_temp)
            {
                if (targ_p != Eigen::Vector3d::Zero())
                {
                    targ.push_back(std::move(targ_temp[idx]));
                    src.push_back(std::move(src_temp[idx]));
                    original_points.push_back(std::move(orig_points_copy[idx]));
                }
                idx++;
            }
        }

        if (targ.empty())
        {
            data.valid = false;
            return;
        }

        std::cout << "MATCHED POINTS COMPARISON" << std::endl;
        printMatchedPoints(src, targ);

        const int matched_size = targ.size();
        data.M_noise = laser_point_cov;

        // h_x matrix arrangement: imw_world_pos, imw_world_ori, lidar_imu_pos, lidar_imu_ori
        data.h_x = Eigen::MatrixXd::Zero(3 * matched_size, h_matrix_col);
        // data.z.resize(matched_size);
        data.z.resize(3 * matched_size);
        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, src.size()),
            [&](const tbb::blocked_range<std::size_t> &r)
            {
                for (std::size_t i = r.begin(); i < r.end(); ++i)
                {
                    const auto &point = src[i]; // This point has been transformed
                    calculate_h_lidar<StateType>(data, i, state, point);
                    // residual computation
                    Eigen::Vector3d residual = point - targ[i];
                    Eigen::Vector3d weight_v(
                        weight(residual.x(), th), weight(residual.y(), th), weight(residual.z(), th));
                    // data.z.segment(3 * i, 3) = weight(residual.squaredNorm(), th) * residual;
                    data.z.segment(3 * i, 3) = residual.cwiseProduct(weight_v);
                }
            });
    }

    template <typename State>
    void EKF::calculate_h_lidar(dyn_share_modified<double> &data, int idx, const State &m, const Eigen::Vector3d &point_in_imu_frame)
    {
        /*
         * General observation equation: y = q * [point_in_imu_frame] * q_c + t - p_w
         * point_imu_world_frame: q_L_I * p_l * q_L_I_c + t_L_I
         * q_c = q conjugate
         */

        // location in h_matrix
        const int h_idx = 3 * idx;

        /* ........................... H-MATRIX CALCULATION .....................................*/
        const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

        /*
         * Case 1: dy/dt && dy/dq
         * dy/dt => only t -> Identity matrix
         * dy/dq => Jacobian of rotation [q*[point_in_imu_frame]*q_c] wrt quaternion
         */

        data.h_x.block<3, 3>(h_idx, 0) = I;
        data.h_x.block<3, 4>(h_idx, 3) = jacobian_wrt_quat(m.rot, point_in_imu_frame);

        if (est_extrinsic)
        {
            /*
             * Case 2:
             * To find the derivative with respect to q_L_I or t_L_I we use chain rule notation.
             * Let A = q_L_I*p_l*q_L_I* + t_L_I
             * dy/d(q_L_I) = dy/dA * dA/d(q_L_I)
             * dy/d(t_L_I) = dy/dA * dA/d(t_L_I)
             */
            const Eigen::Vector4d &imu_rot = m.rot;
            const double w = imu_rot[0];
            const Eigen::Vector3d v(imu_rot[1], imu_rot[2], imu_rot[3]);

            // calculating dy/dA
            Eigen::Matrix3d dy_da = w * w * I + 2 * (w * utils::skew_matrix(v) + v * v.transpose()) - v.squaredNorm() * I;

            // dA/d(t_L_I) is an Identity matrix
            data.h_x.block<3, 3>(h_idx, 7) = dy_da * I;

            // dA/d(q_L_I) = Jacobian of rotation wrt quaternion
            data.h_x.block<3, 4>(idx, 10) = jacobian_wrt_quat(m.offset_R_L_I, original_points[idx]);
        }
    }

    void EKF::h_model_input(dyn_share_modified<double> &data, frame::LidarImuInit::Ptr &meas)
    {
        h_model_lidar<StateInput>(inp_state, data, meas);
    }

    void EKF::h_model_output(dyn_share_modified<double> &data, frame::LidarImuInit::Ptr &meas)
    {
        h_model_lidar<StateOutput>(out_state, data, meas);
    }

    template <int H_Dim>
    bool EKF::update_h_model(frame::LidarImuInit::Ptr &meas)
    {
        dyn_share_modified<double> dyn_share;
        bool check = H_Dim == STATE_IN_DIM;

        auto h_model = [&](auto &data, auto &meas)
        {
            if (check)
                return h_model_input(data, meas);
            else
                return h_model_output(data, meas);
        };

        // iterative loop
        const int max_iter = icp_ptr->config.icp_max_iteration;
        const double max_thresh = icp_ptr->config.estimation_threshold;

        for (int idx = 0; idx < max_iter; idx++)
        {
            dyn_share.valid = true;

            h_model(dyn_share, meas);

            if (!dyn_share.valid)
                return false;

            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> z = dyn_share.z;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> h_x = dyn_share.h_x;
            printEigenMatrix(h_x, "h_x_matrix");
            Eigen::MatrixXd iter_props;
            {
                if (check)
                    iter_props = inp_state.iterative_properties<STATE_IN_DIM>();
                else
                    iter_props = out_state.iterative_properties<INNER_DIM>();
            }
            printEigenMatrix(iter_props, "iter_props_table");
            double m_noise = dyn_share.M_noise;
            const int dof_Measurement = h_x.rows();

            Eigen::MatrixXd HPHT, K;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> PHT;
            {
                PHT = iter_props * h_x.transpose();

                if (check)
                    HPHT = h_x * inp_state.col_iterative_properties<STATE_IN_DIM>(PHT);
                else
                    HPHT = h_x * out_state.col_iterative_properties<INNER_DIM>(PHT);

                for (int m = 0; m < dof_Measurement; m++)
                    HPHT(m, m) += m_noise;

                K = PHT * HPHT.inverse();
                printEigenMatrix(K, "Gain");
            }

            dyn_share.converge = true;
            double pos_norm;
            printEigenMatrix(z, "z");
            if (check)
            {
                Eigen::Matrix<double, STATE_IN_DIM, 1> dx = K * z;
                printEigenMatrix(dx, "dx");

                inp_state += dx;
                inp_state.P = inp_state.P + K * h_x * iter_props.transpose();
                pos_norm = dx.segment(POS, 3).norm();
            }
            else
            {
                Eigen::Matrix<double, INNER_DIM, 1> dx = K * z;
                printEigenMatrix(dx, "dx");
                out_state += dx;
                out_state.P = out_state.P + K * h_x * iter_props.transpose();
                pos_norm = dx.segment(POS, 3).norm();
            }
            std::cout << "Pos_norm: " << pos_norm << std::endl;
            // add rotation criteria from fast-lio2
            if (pos_norm < max_thresh)
                break;
        }

        return true;
    }

    bool EKF::update_h_model_lidar_output(frame::LidarImuInit::Ptr &meas, bool input)
    {
        if (input)
            return update_h_model<STATE_IN_DIM>(meas);

        return update_h_model<INNER_DIM>(meas);
    }

    /*.................................... IMU OUTPUT .............................................................*/
    bool EKF::update_h_model_IMU_output(frame::LidarImuInit::Ptr &meas)
    {

        dyn_share_modified<double> dyn_share;
        int max_iter = 1;
        for (int idx = 0; idx < max_iter; idx++)
        {
            dyn_share.valid = true;
            h_model_IMU_output(dyn_share, meas);

            Eigen::Matrix<double, 6, 1> z = dyn_share.z_IMU;
            Eigen::Matrix<double, INNER_DIM, 6> PHT;
            Eigen::Matrix<double, 6, INNER_DIM> HP;
            Eigen::Matrix<double, 6, 6> HPHT;
            PHT.setZero();
            HP.setZero();
            HPHT.setZero();

            for (int i = 0; i < 3; i++)
            {
                PHT.col(i) = out_state.P.col(BGA + i) + out_state.P.col(IMU_GYRO + i);
                PHT.col(i + 3) = out_state.P.col(BAA + i) + out_state.P.col(IMU_ACC + i);

                HP.row(i) = out_state.P.row(BGA + i) + out_state.P.row(IMU_GYRO + i);
                HP.row(i + 3) = out_state.P.row(BAA + i) + out_state.P.row(IMU_ACC + i);
            }

            for (int i = 0; i < 3; i++)
            {
                HPHT.col(i) = HP.col(BGA + i) + HP.col(IMU_GYRO + i);
                HPHT.col(i + 3) = HP.col(BAA + i) + HP.col(IMU_ACC + i);
            }

            for (int i = 0; i < 6; i++)
                HPHT(i, i) = dyn_share.R_IMU(i);

            Eigen::Matrix<double, INNER_DIM, 6> K = PHT * HPHT.inverse();
            Eigen::Matrix<double, INNER_DIM, 1> dx = K * z;

            out_state.P -= K * HP;
            out_state += dx;
        }

        return true;
    }

    void EKF::h_model_IMU_output(dyn_share_modified<double> &data, const frame::LidarImuInit::Ptr &meas)
    {
        constexpr int imu_data_size = 6;
        std::array<bool, imu_data_size> satu_check = {};

        const double gyro_satu_thresh = 0.99 * satu_gyro;
        const double acc_satu_thresh = 0.99 * satu_acc;
        const bool b_acc_satu_check = acc_satu_thresh > 0;
        const bool b_gyro_satu_check = gyro_satu_thresh > 0;

        const StateOutput &s = out_state;
        Eigen::Vector3d z_omg = ang_vel_read - s.imu_gyro - s.gyro_bias;
        // linear acc already scaled by gravity in update_imu_reading function
        Eigen::Vector3d z_acc = acc_vel_read - s.mult_bias.asDiagonal() * s.imu_acc - s.acc_bias;

        const double h_size = imu_data_size / 2;
        for (int i = 0; i < h_size; ++i)
        {
            const double &gyro_read = ang_vel_read(i);
            if (b_gyro_satu_check && std::fabs(gyro_read) >= gyro_satu_thresh)
            {
                satu_check[i] = true;
                data.z_IMU(i) = 0.0;
            }
            else
                data.z_IMU(i) = z_omg(i);

            const double &acc_read = acc_vel_read(i);
            if (b_acc_satu_check && std::fabs(acc_read) >= acc_satu_thresh)
            {
                satu_check[i + h_size] = true;
                data.z_IMU(i + h_size) = 0.0;
            }
            else
                data.z_IMU(i + h_size) = z_acc(i);
        }

        (data.R_IMU << s.Q.block(Q_GYRO, Q_GYRO, 3, 3).diagonal(), s.Q.block(Q_ACC, Q_ACC, 3, 3).diagonal()).finished();
        std::copy(satu_check.begin(), satu_check.end(), data.satu_check);
    }

    std::tuple<Eigen::Vector3d, Eigen::Quaterniond> EKF::update_map()
    {
        Eigen::Vector3d translation;
        Eigen::Quaterniond quat;
        if (!use_imu_as_input)
        {
            translation = out_state.pos;
            quat = Eigen::Quaterniond(out_state.rot);
        }
        else
        {
            translation = inp_state.pos;
            quat = Eigen::Quaterniond(inp_state.rot);
        }

        Sophus::SE3d pose(quat, translation);
        icp_ptr->update_map(curr_downsampled_frame, pose);
        std::cout << "Current predicted pose:\n"
                  << pose.matrix() << std::endl;
        return {translation, quat};
    }

    utils::Vec3dVector EKF::points_body_to_world(const utils::Vec3dVector &points)
    {
        utils::Vec3dVector world;
        const size_t size = points.size();
        world.resize(size);

        tbb::parallel_for(
            std::size_t(0), size,
            [&](std::size_t idx)
            {
                auto &world_point = world[idx];
                if (est_extrinsic)
                {
                    if (!use_imu_as_input)
                        world_point = out_state.rot_m() * (out_state.L_I_rot_m() * points[idx] + out_state.offset_T_L_I) + out_state.pos;
                    else
                        world_point = inp_state.rot_m() * (inp_state.L_I_rot_m() * points[idx] + inp_state.offset_T_L_I) + inp_state.pos;
                }
                else
                {
                    if (!use_imu_as_input)
                        world_point = out_state.rot_m() * (Lidar_R_wrt_IMU * points[idx] + Lidar_T_wrt_IMU) + out_state.pos;
                    else
                        world_point = inp_state.rot_m() * (Lidar_R_wrt_IMU * points[idx] + Lidar_T_wrt_IMU) + inp_state.pos;
                }
            });

        return world;
    }

    Sophus::SE3d EKF::get_lidar_pos()
    {
        std::cout << "Attempting to get lidar information." << std::endl;
        Eigen::Quaterniond rot_q;
        // current position
        Sophus::SE3d I_W_pose;
        {
            std::cout << "Attempting to get imu world pose." << std::endl;
            if (!use_imu_as_input)
            {

                rot_q = Eigen::Quaterniond(out_state.rot);
                std::cout << "rotation extracted." << std::endl;
                I_W_pose = Sophus::SE3d(rot_q, out_state.pos);
            }
            else
            {
                std::cout << inp_state.rot.transpose() << std::endl;
                rot_q = Eigen::Quaterniond(inp_state.rot);
                std::cout << "rotation extracted." << std::endl;
                I_W_pose = Sophus::SE3d(rot_q, inp_state.pos);
            }
        }
        std::cout << "imu pose extracted." << std::endl;
        Sophus::SE3d L_I_pose;
        if (est_extrinsic)
        {

            if (!use_imu_as_input)
            {
                rot_q = Eigen::Quaterniond(out_state.offset_R_L_I);
                L_I_pose = Sophus::SE3d(rot_q, out_state.offset_T_L_I);
            }
            else
            {
                rot_q = Eigen::Quaterniond(inp_state.offset_R_L_I);
                L_I_pose = Sophus::SE3d(rot_q, inp_state.offset_T_L_I);
            }
        }
        else
        {
            L_I_pose = Sophus::SE3d(utils::rmat2quat(Lidar_R_wrt_IMU), Lidar_T_wrt_IMU);
        }

        Sophus::SE3d pose = I_W_pose * L_I_pose;

        return pose;
    }

    double EKF::position_norm()
    {
        Sophus::SE3d pose = get_lidar_pos();

        return pose.translation().norm();
    }

    bool EKF::initialize_sensors(
        double odom_freq, double timestamp, bool &enabled,
        double &timediff_imu_wrt_lidar, const double &move_start_time)
    {
        // get_lidar information
        std::cout << "in initialize sensors" << std::endl;
        Sophus::SE3d l_pose = get_lidar_pos();
        Eigen::Quaterniond quat = l_pose.unit_quaternion();
        const Eigen::Matrix3d rot = utils::quat2rmat(quat);
        std::cout << "post lidar orientation data extracted" << std::endl;
        Eigen::Vector3d omega_used;
        if (!use_imu_as_input)
        {
            omega_used = inp_state.gyro_bias;
            sensor_init->push_Lidar_CalibState(rot, inp_state.gyro_bias, inp_state.vel, timestamp);
        }
        else
        {
            omega_used = out_state.gyro_bias;
            sensor_init->push_Lidar_CalibState(rot, out_state.gyro_bias, out_state.vel, timestamp);
        }

        // data accumulation appraisal
        std::cout << "Accumulating data" << std::endl;
        if (sensor_init->data_sufficiency_assess(Jaco_rot, frame_num, omega_used, odom_freq))
        {
            std::cout << "Enough data accumulated" << std::endl;
            sensor_init->LI_Initialization(odom_freq, timediff_imu_wrt_lidar, move_start_time);
            data_accum_finished = true;

            if (!use_imu_as_input)
            {
                Eigen::Quaterniond rot_q = utils::rmat2quat(sensor_init->get_R_LI());
                inp_state.offset_R_L_I = rot_q.coeffs();
                inp_state.offset_T_L_I = sensor_init->get_T_LI();
                inp_state.grav = sensor_init->get_Grav_L0();
                inp_state.gyro_bias = sensor_init->get_gyro_bias();
                inp_state.acc_bias = sensor_init->get_acc_bias();
                inp_state.mult_bias = sensor_init->get_mult_bias();
            }
            else
            {
                Eigen::Quaterniond rot_q = utils::rmat2quat(sensor_init->get_R_LI());
                out_state.offset_R_L_I = rot_q.coeffs();
                out_state.offset_T_L_I = sensor_init->get_T_LI();
                out_state.grav = sensor_init->get_Grav_L0();
                out_state.gyro_bias = sensor_init->get_gyro_bias();
                out_state.acc_bias = sensor_init->get_acc_bias();
                out_state.mult_bias = sensor_init->get_mult_bias();
            }

            timediff_imu_wrt_lidar = sensor_init->get_total_time_lag();

            std::cout << "Initialization result:" << std::endl;
        }

        return data_accum_finished;
    }

}