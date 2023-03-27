#include "limu/kalman/ekf.hpp"
#include "limu/kalman/helper.hpp"
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace
{
    auto weight = [](double res_sq, double th)
    { return utils::square(th) / utils::square(th + res_sq); };
}

namespace odometry
{
    void EKF::h_model_input(dyn_share_modified<double> &data, frame::LidarImuInit::Ptr &meas)
    {
        // convert to vector3d points
        utils::Vec3dVector eigen_frame = utils::pointcloud2eigen(*(meas->processed_frame));
        // convert from lidar-world to imu-world frame.
        eigen_frame = points_body_to_world(eigen_frame);
        // downsample map points.
        utils::Vec3_Vec3Tuple downsampled_frame = icp_ptr->voxelize(eigen_frame);

        // downsampled version used in mapping source used to calculate points.
        curr_downsampled_frame = std::get<1>(downsampled_frame); // this is what is used for mapping
        auto source = std::get<0>(downsampled_frame);

        utils::Vec3dVector src, targ;
        double th;
        {
            // Get motion prediction and adaptive threshold
            const double sigma = icp_ptr->get_adaptive_threshold();
            // for each downsampled point find corresponding closest point from the map
            const auto result = icp_ptr->local_map.get_correspondences(source, 3.0 * sigma);
            th = sigma / 3.0;
            std::tie(src, targ) = result;
        }

        if (targ.size() == 0)
        {
            data.valid = false;
            return;
        }

        const int matched_size = targ.size();
        data.M_noise = laser_point_cov;
        data.h_x = Eigen::MatrixXd::Zero(matched_size, h_matrix_col);
        data.z.resize(matched_size);

        const Eigen::Vector3d A(1, 1, 1);
        Eigen::Vector3d point_in_imu_frame;

        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, src.size()),
            [&](const tbb::blocked_range<std::size_t> &r)
            {
                for (std::size_t i = r.begin(); i < r.end(); ++i)
                {
                    const auto &point = src[i];
                    if (est_extrinsic)
                    {
                        /* ........................... H-MATRIX CALCULATION .....................................*/
                        point_in_imu_frame = utils::quat_mult_point(inp_state.offset_R_L_I, point) + inp_state.offset_T_L_I;

                        Eigen::Quaterniond inp_rot_quat(inp_state.rot);
                        Eigen::Quaterniond conjugate = inp_rot_quat.conjugate();
                        Eigen::Vector4d inp_rot_quat_conjugate_coeffs = conjugate.coeffs();

                        // calculate observation wrt rot_imu_w
                        Eigen::Vector4d H_drot_imu_w = utils::ohm(point_in_imu_frame) * inp_rot_quat_conjugate_coeffs;

                        // calculate observation wrt pos_imu_w
                        Eigen::Vector3d H_dp_imu_w = A;

                        // calculate observation wrt rot_lidar_imu
                        Eigen::Quaterniond quat(inp_state.offset_R_L_I);
                        Eigen::Vector4d interim = quat.conjugate().coeffs();
                        interim = utils::ohm(point) * interim;
                        Eigen::Quaterniond res = utils::quat_mult_norm(interim, inp_rot_quat_conjugate_coeffs);
                        Eigen::Vector4d H_drot_lidar_imu = res.coeffs();

                        // calculate observation wrt pos_lidar_imu
                        Eigen::Vector4d H_dp_lidar_imu = conjugate.coeffs();

                        // pos, rot, offset_trans_lidar_imu, offset_rot_lidar_imu
                        data.h_x.block<1, 3>(i, 0) = H_dp_imu_w.transpose();
                        data.h_x.block<1, 4>(i, 3) = H_drot_imu_w.transpose();
                        data.h_x.block<1, 4>(i, 7) = H_dp_lidar_imu.transpose();
                        data.h_x.block<1, 4>(i, 11) = H_drot_lidar_imu.transpose();
                    }
                    else
                    {
                        point_in_imu_frame = Lidar_R_wrt_IMU * point + Lidar_T_wrt_IMU;
                        Eigen::Quaterniond inp_rot_quat(inp_state.rot);
                        Eigen::Vector4d inp_rot_quat_conjugate_coeffs = inp_rot_quat.conjugate().coeffs();

                        // calculate observation wrt rot_imu_w
                        Eigen::Vector4d H_drot_imu_w = utils::ohm(point_in_imu_frame) * inp_rot_quat_conjugate_coeffs;

                        // calculate observation wrt pos_imu_w
                        Eigen::Vector3d H_dp_imu_w = A;

                        // pos, rot, offset_trans_lidar_imu, offset_rot_lidar_imu
                        data.h_x.block<1, 3>(i, 0) = H_dp_imu_w.transpose();
                        data.h_x.block<1, 4>(i, 3) = H_drot_imu_w.transpose();
                    }

                    // residual computation
                    Eigen::Vector3d residual = point - targ[i];
                    data.z(i) = weight(residual.squaredNorm(), th) * residual.norm();
                    ;
                }
            });
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

    void EKF::h_model_output(dyn_share_modified<double> &data, frame::LidarImuInit::Ptr &meas)
    {
        // convert to vector3d points
        utils::Vec3dVector eigen_frame = utils::pointcloud2eigen(*(meas->processed_frame));
        // convert from lidar-world to imu-world frame.
        eigen_frame = points_body_to_world(eigen_frame);
        // downsample map points.
        utils::Vec3_Vec3Tuple downsampled_frame = icp_ptr->voxelize(eigen_frame);

        // downsampled version used in mapping source used to calculate points.
        curr_downsampled_frame = std::get<1>(downsampled_frame); // this is what is used for mapping
        auto source = std::get<0>(downsampled_frame);

        utils::Vec3dVector src, targ;
        double th;
        {
            // Get motion prediction and adaptive threshold
            const double sigma = icp_ptr->get_adaptive_threshold();
            // for each downsampled point find corresponding closest point from the map
            const auto result = icp_ptr->local_map.get_correspondences(source, 3.0 * sigma);
            std::tie(src, targ) = result;

            th = sigma / 3.0;
        }

        if (targ.size() == 0)
        {
            data.valid = false;
            return;
        }

        const int matched_size = targ.size();
        data.M_noise = laser_point_cov;
        data.h_x = Eigen::MatrixXd::Zero(matched_size, h_matrix_col);
        data.z.resize(matched_size);

        const Eigen::Vector3d A(1, 1, 1);
        Eigen::Vector3d point_in_imu_frame;

        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, src.size()),
            [&](const tbb::blocked_range<std::size_t> &r)
            {
                for (std::size_t i = r.begin(); i < r.end(); ++i)
                {
                    const auto &point = src[i];
                    if (est_extrinsic)
                    {
                        /* ........................... H-MATRIX CALCULATION .....................................*/
                        point_in_imu_frame = utils::quat_mult_point(out_state.offset_R_L_I, point) + out_state.offset_T_L_I;

                        Eigen::Quaterniond out_rot_quat(out_state.rot);
                        Eigen::Quaterniond conjugate = out_rot_quat.conjugate();
                        Eigen::Vector4d out_rot_quat_conjugate_coeffs = conjugate.coeffs();

                        // calculate observation wrt rot_imu_w
                        Eigen::Vector4d H_drot_imu_w = utils::ohm(point_in_imu_frame) * out_rot_quat_conjugate_coeffs;

                        // calculate observation wrt pos_imu_w
                        Eigen::Vector3d H_dp_imu_w = A;

                        // calculate observation wrt rot_lidar_imu
                        Eigen::Quaterniond quat(out_state.offset_R_L_I);
                        Eigen::Vector4d interim = quat.conjugate().coeffs();
                        interim = utils::ohm(point) * interim;
                        Eigen::Quaterniond res = utils::quat_mult_norm(interim, out_rot_quat_conjugate_coeffs);
                        Eigen::Vector4d H_drot_lidar_imu = res.coeffs();

                        // calculate observation wrt pos_lidar_imu
                        Eigen::Vector4d H_dp_lidar_imu = conjugate.coeffs();

                        // pos, rot, offset_trans_lidar_imu, offset_rot_lidar_imu
                        data.h_x.block<1, 3>(i, 0) = H_dp_imu_w.transpose();
                        data.h_x.block<1, 4>(i, 3) = H_drot_imu_w.transpose();
                        data.h_x.block<1, 4>(i, 7) = H_dp_lidar_imu.transpose();
                        data.h_x.block<1, 4>(i, 11) = H_drot_lidar_imu.transpose();
                    }
                    else
                    {
                        point_in_imu_frame = Lidar_R_wrt_IMU * point + Lidar_T_wrt_IMU;
                        Eigen::Quaterniond out_rot_quat(out_state.rot);
                        Eigen::Vector4d out_rot_quat_conjugate_coeffs = out_rot_quat.conjugate().coeffs();

                        // calculate observation wrt rot_imu_w
                        Eigen::Vector4d H_drot_imu_w = utils::ohm(point_in_imu_frame) * out_rot_quat_conjugate_coeffs;

                        // calculate observation wrt pos_imu_w
                        Eigen::Vector3d H_dp_imu_w = A;

                        // pos, rot, offset_trans_lidar_imu, offset_rot_lidar_imu
                        data.h_x.block<1, 3>(i, 0) = H_dp_imu_w.transpose();
                        data.h_x.block<1, 4>(i, 3) = H_drot_imu_w.transpose();
                    }

                    // residual computation
                    Eigen::Vector3d residual = point - targ[i];
                    data.z(i) = weight(residual.squaredNorm(), th) * residual.norm();
                }
            });
    }

    void EKF::h_model_IMU_output(dyn_share_modified<double> &data, const frame::LidarImuInit::Ptr &meas)
    {
        const StateOutput &s = out_state;
        data.z_IMU.block<3, 1>(0, 0) = meas->curr_ang_vel - s.imu_gyro - s.gyro_bias;
        data.z_IMU.block<3, 1>(3, 0) = meas->curr_acc * gravity / meas->get_mean_acc_norm() - s.mult_bias.asDiagonal() * s.imu_acc - s.acc_bias;
        (data.R_IMU << s.Q.block(Q_GYRO, Q_GYRO, 3, 3).diagonal(), s.Q.block(Q_ACC, Q_ACC, 3, 3).diagonal()).finished();
        // probably should include saturation check but can't find saturation information onliine.
    }

    // ahould do one for the input too
    bool EKF::update_h_model_input(frame::LidarImuInit::Ptr &meas)
    {

        dyn_share_modified<double> dyn_share;

        // iterative loop
        for (int idx = 0; idx < icp_ptr->config.icp_max_iteration; idx++)
        {
            dyn_share.valid = true;
            h_model_input(dyn_share, meas);

            if (!dyn_share.valid)
                return false;

            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> z = dyn_share.z;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> h_x = dyn_share.h_x;
            Eigen::MatrixXd iter_props = inp_state.iterative_properties<STATE_IN_DIM>();
            const int dof_Measurement = h_x.rows();
            double m_noise = dyn_share.M_noise;
            Eigen::MatrixXd HPHT, K;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> PHT;
            {
                PHT = iter_props * h_x.transpose();
                HPHT = h_x * inp_state.col_iterative_properties<STATE_IN_DIM, Eigen::Dynamic>(PHT);
                for (int m = 0; m < dof_Measurement; m++)
                {
                    HPHT(m, m) += m_noise;
                }
                K = PHT * HPHT.inverse();
            }
            Eigen::Matrix<double, STATE_IN_DIM, 1> dx = K * z;
            inp_state += dx;

            dyn_share.converge = true;
            inp_state.P = inp_state.P + K * h_x * iter_props.transpose();

            if (dx.segment(POS, 3).norm() < icp_ptr->config.estimation_threshold)
                break;
        }

        return true;
    }

    bool EKF::update_h_model_output(frame::LidarImuInit::Ptr &meas)
    {

        dyn_share_modified<double> dyn_share;

        // iterative loop
        for (int idx = 0; idx < icp_ptr->config.icp_max_iteration; idx++)
        {
            dyn_share.valid = true;
            h_model_output(dyn_share, meas);

            if (!dyn_share.valid)
                return false;

            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> z = dyn_share.z;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> h_x = dyn_share.h_x;
            Eigen::MatrixXd iter_props = out_state.iterative_properties<INNER_DIM>();
            int dof_Measurement = h_x.rows();
            double m_noise = dyn_share.M_noise;
            Eigen::MatrixXd HPHT, K;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> PHT;
            {
                PHT = iter_props * h_x.transpose();
                HPHT = h_x * out_state.col_iterative_properties<INNER_DIM, Eigen::Dynamic>(PHT);
                for (int m = 0; m < dof_Measurement; m++)
                {
                    HPHT(m, m) += m_noise;
                }
                K = PHT * HPHT.inverse();
            }
            Eigen::Matrix<double, INNER_DIM, 1> dx = K * z;
            out_state += dx;

            dyn_share.converge = true;
            out_state.P = out_state.P + K * h_x * iter_props.transpose();

            if (dx.segment(POS, 3).norm() < icp_ptr->config.estimation_threshold)
                break;

            return true;
        }
    }

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
}