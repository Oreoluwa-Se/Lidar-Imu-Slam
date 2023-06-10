#include "limu/kalman/ekf.hpp"
#include "limu/kalman/helper.hpp"
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <type_traits>
#include <iomanip>
#include <sys/stat.h>
#include <filesystem>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

namespace
{
    // initial points threshold
    constexpr int init_points_req = 100;

    auto weight = [](double res_sq, double th)
    { return utils::square(th) / utils::square(th + res_sq); };

    int save_pose_to_log(const std::vector<Sophus::SE3d> &poses, const std::string &file_path, int idx)
    {
        // Extract the directory path from the file path
        size_t last_separator_pos = file_path.find_last_of("/\\");
        std::string dir_path = file_path.substr(0, last_separator_pos);

        // Check if the directory exists
        struct stat info;
        if (stat(dir_path.c_str(), &info) != 0 || !(info.st_mode & S_IFDIR))
        {
            // Directory doesn't exist, create it
#ifdef _WIN32
            int status = _mkdir(dir_path.c_str());
#else
            int status = mkdir(dir_path.c_str(), 0777);
#endif

            if (status != 0)
            {
                std::cerr << "Failed to create directory: " << dir_path << std::endl;
                return false;
            }
        }

        // Open the file in append mode
        std::ofstream file;
        if (idx == 0)
            file.open(file_path);
        else
            file.open(file_path, std::ios_base::app);

        if (!file.is_open())
        {
            std::cerr << "Error opening file: " << file_path << std::endl;
            return 0;
        }

        // Set the output precision and fixed format for Euler angles and translations
        file << std::fixed << std::setprecision(5);

        // Write the table headers
        if (file.tellp() == 0)
        {
            file << "Pose\t\t\t\t\troll\t\tpitch\t\tyaw\t\ttranslation_x\ttranslation_y\ttranslation_z\n";
            file << "-----------------------------------------------------------------------------------------------------\n";
        }

        // Iterate over each pose and save it to the file
        for (size_t i = 0; i < poses.size(); ++i)
        {
            const auto &pose = poses[i];

            // Get the Euler angles from the pose
            Eigen::Vector3d euler = utils::rmat_to_euler(pose.rotationMatrix());

            // Write the pose index and pose values to the file
            file << "Pose " << std::setw(2) << idx + i + 1 << ":\t";
            file << std::setw(10) << euler[0] << "\t";
            file << std::setw(10) << euler[1] << "\t";
            file << std::setw(10) << euler[2] << "\t";
            file << std::setw(15) << pose.translation()[0] << "\t";
            file << std::setw(15) << pose.translation()[1] << "\t";
            file << std::setw(15) << pose.translation()[2] << "\n";
        }

        file.close();
        return idx + static_cast<int>(poses.size());
    }
}

namespace odometry
{

    bool EKF::initialize_map(const frame::SensorData &data)
    {
        const auto downsampled_frame = icp_ptr->voxelize(data.lidar_data.pc);
        auto &eigen_frame = std::get<1>(downsampled_frame); // store map_points.

        std::move(eigen_frame.begin(), eigen_frame.end(), std::back_inserter(initial_frames));

        // store some initial points.
        if (static_cast<int>(initial_frames.size()) < init_points_req)
            return false;

        // insert points into map
        lidar_point_cloud = initial_frames;
        update_map(initial_frames);
        initial_frames.clear();

        init_map = true;

        return true;
    }

    /*........................................... LIDAR H MATRIX MODEL ...................................................*/
    void EKF::outlier_removal(
        const std::vector<MapPoint> &targ, const std::vector<int> &returned_idxs,
        std::vector<MapPoint> &targ_calc, std::vector<int> &tracked_idx)
    {
        const int targ_size = targ.size();
        targ_calc.clear();
        tracked_idx.clear();

        std::vector<double> distance;
        distance.reserve(targ_size);

        std::transform(
            returned_idxs.begin(), returned_idxs.end(),
            std::back_inserter(distance),
            [&](const auto &idx)
            {
                return (*tracker.points_imu_world_coord[idx] - targ[idx]).squaredNorm();
            });

        // {lower, higher, iqr_val}
        const auto iqr_vec = utils::IQR(distance);
        double mad = utils::calculate_mad(distance);
        double data_median = utils::calculate_median(distance, 0, static_cast<size_t>(distance.size() - 1));

        double low_bound = iqr_vec[0] - IQR_TUCHEY * iqr_vec[2];
        double high_bound = iqr_vec[1] + IQR_TUCHEY * iqr_vec[2];

        auto filter_func = [&](size_t i)
        {
            const double dist = distance[i];
            const double dist_diff = std::abs(dist - data_median);
            return (dist >= low_bound && dist <= high_bound && dist_diff < MAD_THRESH_VAL * mad);
        };

        tracked_idx.reserve(targ_size);
        targ_calc.reserve(targ_size);

        for (size_t i = 0; i < targ_size; i++)
        {
            if (!filter_func(i))
                continue;

            tracked_idx.push_back(returned_idxs[i]);
            targ_calc.emplace_back(std::move(targ[i]));
        }
    }

    bool EKF::h_model_lidar(dyn_share_modified<double> &data)
    {
        // Find closest correspondence. {index to source, vector of closest points}
        const auto &local_map = icp_ptr->local_map;
        const auto result = local_map->get_correspondences(tracker.points_imu_world_coord, 3.0 * sigma, num_corresp);
        tracked_idx.clear();

        const auto &returned_idxs = std::get<0>(result);
        const auto &targ = std::get<1>(result);
        const int targ_size = targ.size();

        PointsVector targ_calc = targ;
        tracked_idx = returned_idxs;
        if (targ_size > 4 && use_outliers)
            outlier_removal(targ, returned_idxs, targ_calc, tracked_idx);

        if (targ_calc.empty())
        {
            data.valid = false;
            return false;
        }

        PointsVector source;
        std::transform(
            tracked_idx.begin(), tracked_idx.end(),
            std::back_inserter(source),
            [&](const auto &val)
            {
                return *tracker.points_imu_world_coord[val];
            });

        if (!tracker.first_matched)
        {
            tracker.matched[0] = {source, targ_calc};
            tracker.first_matched = true;
        }
        else
            tracker.matched[1] = {source, targ_calc};

        if (print_matrices)
        {
            utils::printMatchedPoints(source, targ_calc);

            // Print results after distance to Map
            std::cout << "Results after distance to Map" << std::endl;
            std::cout << "Total points with matched points: " << targ_size << std::endl;
            if (targ_calc.size() > 4)
                std::cout << "Total points post outlier removal: " << targ_calc.size() << std::endl;
            else
                std::cout << "Not enough points [" << targ_calc.size() << "] to trim" << std::endl;
        }

        const int matched_size = targ_calc.size();
        data.h_x = Eigen::MatrixXd::Zero(3 * matched_size, NominalState::h_matrix_col);
        data.weight.resize(3 * matched_size);
        data.z.resize(3 * matched_size);
        data.M_noise = laser_point_cov;

        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, matched_size),
            [&](const tbb::blocked_range<std::size_t> &r)
            {
                for (std::size_t i = r.begin(); i < r.end(); ++i)
                {
                    calculate_h_lidar(data, i);
                    Eigen::Vector3d residual = (*tracker.points_imu_world_coord[tracked_idx[i]] - targ_calc[i]).point;

                    std::lock_guard<std::mutex> lock(data.mutex);
                    data.z.segment(3 * i, 3) = -residual;

                    data.weight(3 * i) = weight(residual.x() * residual.x(), threshold);
                    data.weight(3 * i + 1) = weight(residual.y() * residual.y(), threshold);
                    data.weight(3 * i + 2) = weight(residual.z() * residual.z(), threshold);
                }
            });

        return true;
    }

    void EKF::calculate_h_lidar(dyn_share_modified<double> &data, int idx)
    {
        std::lock_guard<std::mutex> lock(data.mutex);

        /*
         * Calculate Jacobian of point-to-point icp equation: q x [point_in_imu_frame] x q_c + t - p_w
         * point_imu_frame: q_b x p_l x q_b* + t_b
         */
        const int h_idx = 3 * idx;
        const Eigen::Vector3d &point_in_imu_frame = tracker.points_imu_coord[tracked_idx[idx]]->point;

        // Case 1: dy/dt
        data.h_x.block<3, 3>(h_idx, 0) = Eigen::Matrix3d::Identity();

        // Case 2: dy/dq => Jacobian of equation with respect to quaternion
        data.h_x.block<3, 4>(h_idx, 3) = utils::jacobian_wrt_quat(state->nominal_state.rot, point_in_imu_frame);

        if (est_extrinsic)
        {
            // Case 3: dy/dt_b => Jacobian of equation with respect to lidar imu translation
            data.h_x.block<3, 3>(h_idx, 7) = utils::vec4d_to_rmat(state->nominal_state.rot);

            // Case 4: dy/dq_b => Jacobian of equation with respect to lidar imu orientation
            const Eigen::Vector3d &point_in_lidar_frame = tracker.orig_point_lidar[tracked_idx[idx]]->point;
            data.h_x.block<3, 4>(h_idx, 10) = data.h_x.block<3, 3>(h_idx, 7) * utils::jacobian_wrt_quat(state->nominal_state.offset_R_L_I, point_in_lidar_frame);
        }
    }

    bool EKF::update_h_model(const frame::SensorData &data)
    {
        dyn_share_modified<double> dyn_share;
        reset_parameters();
        tracker.clear();
        /*.............................. Main function .................................*/
        // voxelize and transform points
        const auto downsampled_frame = icp_ptr->voxelize(data.lidar_data.pc);

        // points used for mapping.
        tracker.map_points = std::get<1>(downsampled_frame);

        // identifies what we use for pose estimation.
        tracker.orig_point_lidar = std::get<0>(downsampled_frame);

        // reference for lidar points later
        lidar_point_cloud = tracker.map_points;
        tracker.initial_transform(icp_ptr->init_guess);
        // tracker.transform_points(state);

        // Chamber of ICP secrets
        for (int idx = 0; idx < max_icp_iter; ++idx)
        {
            dyn_share.valid = true;
            if (!h_model_lidar(dyn_share))
            {
                if (!dyn_share.valid)
                {
                    // if we can't find matching targets and it's the first loop
                    if (idx == 0)
                    {
                        if (verbose)
                            std::cout << "Adding new points to the map!" << std::endl;
                        update_map(tracker.map_points);
                    }
                    else if (verbose)
                        std::cout << "Invalid dyn_share" << std::endl;
                }
                return false;
            }

            dx_norm = align_points(dyn_share);
            tracker.transform_points(state);

            if (dx_norm <= estimation_threshold)
            {
                dyn_share.converge = true;
                break;
            }
        }

        // update the map
        update_map(tracker.map_points);

        return true;
    }

    double EKF::align_points(dyn_share_modified<double> &dyn_share)
    {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> z = dyn_share.z;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> h_x = dyn_share.h_x;
        Eigen::MatrixXd noise = Eigen::MatrixXd::Identity(z.rows(), z.rows()) * dyn_share.M_noise * state->get_noise_scale();

        // outlier weights applied
        h_x = dyn_share.weight.asDiagonal() * h_x * state->lidar_ts_jacobian();
        z = dyn_share.weight.asDiagonal() * z;

        return kf_propagation(h_x, state->P_measurement_props_lidar(), z, noise, false);
    }

    /*.................................... IMU OUTPUT ....................................*/
    void EKF::h_model_IMU_output(dyn_share_modified<double> &data, double gravity_sf)
    {
        constexpr int imu_data_size = 6;
        std::array<bool, imu_data_size> satu_check = {};

        const double gyro_satu_thresh = 0.99 * satu_gyro;
        const double acc_satu_thresh = 0.99 * satu_acc;
        const bool b_acc_satu_check = acc_satu_thresh > 0.0;
        const bool b_gyro_satu_check = gyro_satu_thresh > 0.0;

        // compute residual gyro
        data.z_IMU.block<3, 1>(0, 0) = (ang_vel_read - state->nominal_state.gyro_bias) - state->nominal_state.imu_gyro;

        // compute residual acc
        Eigen::Matrix3d mult_bias = state->nominal_state.mult_bias.asDiagonal();
        Eigen::Matrix3d mult_bias_s = mult_bias.inverse();
        Eigen::Vector3d scaled_acc_vel = acc_vel_read;
        data.z_IMU.block<3, 1>(3, 0) = mult_bias * scaled_acc_vel - state->nominal_state.acc_bias - state->nominal_state.imu_acc;

        if (print_matrices)
            print_imu(state);

        // jacobian
        data.h_x = Eigen::MatrixXd::Zero(h_imu_jacob_row, h_imu_jacob_col);
        {
            // bomg
            data.h_x.block(0, 0, 3, 3).setIdentity();

            // Igyro
            data.h_x.block(0, 9, 3, 3).setIdentity();

            // bacc
            data.h_x.block(3, 3, 3, 3) = mult_bias_s;

            // for bat
            Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
            for (int i = 0; i < 3; ++i)
                J(i, i) = -1 / (mult_bias_s(i, i) * mult_bias_s(i, i));
            data.h_x.block(3, 6, 3, 3) = J * (state->nominal_state.imu_acc + state->nominal_state.acc_bias).asDiagonal();

            // Iacc
            data.h_x.block(3, 12, 3, 3) = mult_bias_s;
        }

        const int h_size = imu_data_size / 2;
        for (int i = 0; i < h_size; ++i)
        {
            const double gyro_read = std::fabs(ang_vel_read(i));
            if (b_gyro_satu_check && gyro_read >= gyro_satu_thresh)
            {
                satu_check[i] = true;
                data.z_IMU(i) = 0.0;
            }

            const double acc_read = std::fabs(acc_vel_read(i));
            if (b_acc_satu_check && acc_read >= acc_satu_thresh)
            {
                satu_check[i + h_size] = true;
                data.z_IMU(i + h_size) = 0.0;
            }
        }

        (data.R_IMU << imu_gyro_meas_cov, imu_gyro_meas_cov, imu_gyro_meas_cov, imu_acc_meas_cov, imu_acc_meas_cov, imu_acc_meas_cov).finished();
        std::copy(satu_check.begin(), satu_check.end(), data.satu_check);
    }

    void EKF::print_imu(const State::Ptr &state)
    {
        Eigen::Matrix3d mult_bias = state->nominal_state.mult_bias.asDiagonal();
        Eigen::Matrix3d mult_bias_s = mult_bias.inverse();

        std::cout << std::left << std::setw(20) << "-----------------------------------------------------------\n";
        std::cout << std::left << std::setw(20) << "Measured Imu:\n";
        std::cout << std::left << std::setw(10) << "Gyro: " << ang_vel_read.transpose() << "\n";
        std::cout << std::left << std::setw(10) << "Acc: " << acc_vel_read.transpose() << "\n";
        std::cout << std::left << std::setw(20) << "-----------------------------------------------------------\n";

        std::cout << std::left << std::setw(20) << "Estimated Imu:\n";
        std::cout << std::left << std::setw(10) << "Gyro: " << state->nominal_state.imu_gyro.transpose() << "\n";
        std::cout << std::left << std::setw(10) << "Acc: " << state->nominal_state.imu_acc.transpose() << "\n";
        std::cout << std::left << std::setw(20) << "-----------------------------------------------------------\n";

        std::cout << std::left << std::setw(20) << "Bias Estimation\n";
        std::cout << std::left << std::setw(20) << "-----------------------------------------------------------\n";
        std::cout << std::left << std::setw(10) << "Gyro bias: " << state->nominal_state.gyro_bias.transpose() << "\n";
        std::cout << std::left << std::setw(10) << "Acc bias: " << state->nominal_state.acc_bias.transpose() << "\n";
        std::cout << std::left << std::setw(10) << "Mult bias: " << state->nominal_state.mult_bias.transpose() << "\n";
        std::cout << std::left << std::setw(20) << "-----------------------------------------------------------\n";

        std::cout << std::left << std::setw(20) << "Combined\n";
        Eigen::Vector3d acc_comb = mult_bias_s * (state->nominal_state.imu_acc + state->nominal_state.acc_bias);
        Eigen::Vector3d gyro_comb = state->nominal_state.imu_gyro + state->nominal_state.gyro_bias;
        std::cout << std::left << std::setw(20) << "-----------------------------------------------------------\n";
        std::cout << std::left << std::setw(10) << "Gyro Combined: " << gyro_comb.transpose() << "\n";
        std::cout << std::left << std::setw(10) << "Acc Combined: " << acc_comb.transpose() << "\n";
        std::cout << std::left << std::setw(20) << "-----------------------------------------------------------\n";
    }

    bool EKF::update_h_model_IMU_output(double gravity_sf)
    {
        dyn_share_modified<double> dyn_share;
        double iter_count = 0;
        double dx_n = 0;
        for (int idx = 0; idx < max_imu_iter; idx++)
        {
            dyn_share.valid = true;
            h_model_IMU_output(dyn_share, gravity_sf);

            Eigen::Matrix<double, 6, 1> z = dyn_share.z_IMU;
            Eigen::MatrixXd iter_props = state->P_measurement_props_imu();

            Eigen::MatrixXd noise = dyn_share.R_IMU.asDiagonal() * state->get_noise_scale();

            Eigen::MatrixXd h_x = dyn_share.h_x * state->imu_ts_jacobian();

            // remove effects from saturated columns
            Eigen::MatrixXd satu = Eigen::MatrixXd::Identity(h_imu_jacob_row, h_imu_jacob_row);
            {
                for (int i = 0; i < h_imu_jacob_row; i++)
                {
                    // ignore values that are saturated
                    if (dyn_share.satu_check[i])
                        satu(i, i) = 0;
                }

                h_x.block(0, 9, 6, 6) *= satu;
                iter_props.block(0, 9, 6, 6) *= satu;
            }

            if (print_matrices)
            {
                utils::printEigenMatrix(h_x, "h_x");
                utils::printEigenMatrix(iter_props, "iter_props");
            }

            dx_n = kf_propagation(h_x, iter_props, z, noise, true);
            iter_count = idx;

            if (dx_n < estimation_threshold)
            {
                dyn_share.converge = true;
                break;
            }
        }

        tracker.update_pose(state);
        icp_ptr->update_pose(tracker.T_I_W);

        if (verbose)
        {
            if (dyn_share.converge)
                std::cout << "Number of iterations till convergence: " << iter_count << std::endl;
            else
                std::cout << "Max iterations reached " << std::endl;

            if (!print_matrices)
                print_imu(state);

            std::cout << "Dx_norm: " << dx_n << std::endl;
            tracker.curr_info();
        }

        return true;
    }

    /*.................................... Kalman Filter Propagation ....................................*/
    double EKF::kf_propagation(const Eigen::MatrixXd &h_x, const Eigen::MatrixXd &iter_props, const Eigen::Matrix<double, -1, 1> &z, const Eigen::MatrixXd &noise, bool is_imu)
    {
        Eigen::MatrixXd HPHT, K, PHT;
        HPHT.setZero();
        K.setZero();
        PHT.setZero();
        {
            PHT = iter_props * h_x.transpose();
            if (!is_imu)
                HPHT = h_x * state->row_properties_lidar(PHT);
            else
                HPHT = h_x * state->row_properties_imu(PHT);

            K = PHT * (HPHT + noise).inverse();
        }

        // section updates the state, covariance and resets.
        Eigen::Matrix<double, Eigen::Dynamic, 1> dx = K * z;
        {
            *state += dx;
            state->P = state->P - K * h_x * iter_props.transpose();

            // Reset covariance matrix
            state->update_G(
                dx.segment(ErrorState::ORI, 3),
                dx.segment(ErrorState::ORI_LIDAR_IMU, 3));
        }

        // termination criteria
        double dx_norm = 0.0;
        if (!is_imu)
            dx_norm = dx.segment(ErrorState::ORI, 3).norm() + dx.segment(ErrorState::ORI_LIDAR_IMU, 3).norm();
        else
            dx_norm = dx.segment(ErrorState::IMU_ACC, 3).norm() + dx.segment(ErrorState::IMU_GYRO, 3).norm();

        // print stuff
        if (print_matrices)
        {
            std::cout << "......... IMU MATRICES ............" << std::endl;
            utils::printEigenMatrix(z, "z");
            utils::printEigenMatrix(h_x, "h_x");
            utils::printEigenMatrix(PHT, "PHT");
            utils::printEigenMatrix(HPHT, "HPHT");
            utils::printEigenMatrix(K, "gain");
            utils::printEigenMatrix(dx, "dx");
            state->print_nominal_attrbiutes();
        }

        // returns difference
        return dx_norm;
    }

    void EKF::predict(double dt, bool predict_state, bool prop_cov)
    {
        if (use_imu_as_input)
            state->predict(ang_vel_read, acc_vel_read, dt, predict_state, prop_cov);
        else
            state->predict(zero, zero, dt, predict_state, prop_cov);
    }

    void EKF::pose_trail_tracker()
    {
        if (key_poses.size() >= frames_to_keep)
        {
            // dump to log file
            key_tracker = save_pose_to_log(key_poses, pose_storage, key_tracker);
            if (static_cast<int>(key_poses.size()) > 150)
                key_poses.erase(key_poses.begin(), key_poses.end() - 100);

            icp_ptr->reset_poses(key_poses);
            key_poses.clear();
        }

        // keep key poses
        if (!icp_ptr->is_empty())
            key_poses.emplace_back(icp_ptr->poses_().back());

        // keep main poses
        if (icp_ptr->poses_().size() >= frames_to_keep)
        {
            if (store_all)
                all_tracker = save_pose_to_log(icp_ptr->poses_(), all_pose_storage, all_tracker);

            icp_ptr->reset_poses(key_poses);
        }
    }

    void EKF::update_map(PointsPtrVector &points)
    {
        tracker.update_pose(state);
        utils::transform_points(tracker.T_total, points);
        icp_ptr->update_map(points, tracker.T_I_W, false);

        if (verbose)
        {
            std::cout << "Dx norm: " << dx_norm << std::endl;
            tracker.print_match_info();
            tracker.curr_info();
        }

        tracker.clear();
    }
}