#include "limu/kalman/ekf.hpp"
#include "limu/kalman/helper.hpp"
#include <cassert>
#include <tbb/parallel_for.h>

namespace
{
    using namespace kalman;

    // from HYBVIO code base
    void update_common(Eigen::VectorXd &m, Eigen::MatrixXd &P, const Eigen::MatrixXd &HP, const Eigen::MatrixXd &K)
    {
        P.noalias() -= K * HP;

        // normalize orientations
        m.segment(ORI, 4).normalize();
        m.segment(ROT_IMU_LIDAR, 4).normalize();
    }

    void update_common_joseph_form(
        Eigen::MatrixXd &P, const Eigen::MatrixXd &H, const Eigen::MatrixXd &R,
        const Eigen::MatrixXd &K, Eigen::MatrixXd &tmpP1, Eigen::MatrixXd &tmpP0)
    {
        const int n = P.rows();
        assert(H.cols() == n);

        tmpP1.noalias() = -K * H;
        tmpP1 += Eigen::MatrixXd::Identity(n, n);
        tmpP0.noalias() = tmpP1 * P;

        P.noalias() = tmpP0 * tmpP1.transpose();
        tmpP0.noalias() = R * K.transpose();
        P.noalias() += K * tmpP0;
    }

    void update(Eigen::VectorXd &m, Eigen::MatrixXd &P, const Eigen::VectorXd &y,
                const Eigen::MatrixXd &H, const Eigen::MatrixXd &R, Eigen::MatrixXd &K,
                Eigen::MatrixXd &HP, Eigen::MatrixXd &tmpP0,
                Eigen::LDLT<Eigen::MatrixXd> &invS)
    {
        const int l = H.cols();
        assert(l <= P.rows());

        HP.noalias() = H * P.topRows(l);

        Eigen::MatrixXd &S = tmpP0;
        S = R;
        S.noalias() += HP.leftCols(l) * H.transpose();
        invS.compute(S);
        tmpP0 = invS.solve(HP);
        K = tmpP0.transpose();

        Eigen::MatrixXd &v = tmpP0;
        v.noalias() = -H * m.topRows(l);
        v += y;
        m.noalias() += K * v; // m += K * (y - H * m)

        // allegedly more robust to numerical instability
        update_common(m, P, HP, K);
    }
}

namespace kalman
{
    EKF::EKF(EKF_PARAMETERS::Ptr parameters)
        : params(parameters), noise_scale(params->noise_scale * params->noise_scale),
          lidar_pose_count(params->lidar_pose_trail), state_dim(INNER_DIM + lidar_pose_count * POSE_DIM),
          augment_count(0), augment_times(), time(0.0), ZUPTtime(-1.0), ZRUPTtime(-1.0), initZUPTtime(-1.0),
          was_stationary(false), prev_sampleT(-1.0), first_sampleT(-1.0), first_sample(true),
          mc_tracker(std::make_unique<EKF::Tracker>()), last_lidar_end_time(0.0),
          orientation_initialized(false)
    {
        // initialize tracker
        mc_tracker->acc_s_last = Eigen::Vector3d::Zero();
        mc_tracker->ang_vel_last = Eigen::Vector3d::Zero();
        mc_tracker->mean_acc = Eigen::Vector3d::Zero();
        mc_tracker->mean_gyr = Eigen::Vector3d::Zero();

        // just for initialization
        (grav << 0, 0, -gravity).finished();

        m = Eigen::MatrixXd::Zero(state_dim, 1);
        P = Eigen::MatrixXd::Zero(state_dim, state_dim);

        const int max_HRows = state_dim;

        // preallocate all temp matrices
        H = Eigen::MatrixXd::Zero(max_HRows, state_dim);
        HP = H;
        K = Eigen::MatrixXd::Zero(max_HRows, max_HRows);
        tmp_1 = P;
        tmp_0 = P;
        S = K;
        R = K;
        invS = Eigen::LDLT<Eigen::MatrixXd>(max_HRows);

        dydx.setZero();
        dydq.setZero();

        // intitialize P and m values
        (m.segment(ORI, 4) << 1, 0, 0, 0).finished();
        m.segment(ROT_IMU_LIDAR, 4) = m.segment(ORI, 4);
        (m.segment(BAT, 3) << 1.0, 1.0, 1.0).finished();

        initialize_process_covariance(
            P, params->init_pos_noise, params->init_vel_noise,
            params->init_bga_noise, params->init_baa_noise, params->init_bat_noise,
            params->init_lidar_imu_time_noise, params->init_pos_trail_noise,
            params->init_ori_trail_noise, noise_scale);

        // covariance noise Q setup
        Q = Eigen::MatrixXd::Zero(Q_DIM, Q_DIM);
        Q.block(Q_ACC, Q_ACC, 3, 3).setIdentity(3, 3) *= utils::square(params->acc_process_noise);
        Q.block(Q_GYRO, Q_GYRO, 3, 3).setIdentity(3, 3) *= utils::square(params->gyro_process_noise);

        Q *= noise_scale;

        // prepping constant matrices used in visual update and augmentation.
        for (int dropped_pose_idx = 0; dropped_pose_idx < lidar_pose_count; ++dropped_pose_idx)
        {
            visAugA.emplace_back(Eigen::SparseMatrix<double>(state_dim, state_dim));
            Eigen::SparseMatrix<double> &A = visAugA.back();
            // primary state unchanged.
            for (int i = 0; i < LIDAR; i++)
                A.insert(i, i) = 1;

            // shift poses by one, until the index we are dropping.
            for (int i = LIDAR; i < LIDAR + dropped_pose_idx * POSE_DIM; ++i)
            {
                assert(i + POSE_DIM < state_dim);
                A.insert(i + POSE_DIM, i) = 1;
            }

            // remaining states are unchanged
            for (int i = LIDAR + (dropped_pose_idx + 1) * POSE_DIM; i < state_dim; i++)
                A.insert(i, i) = 1;

            A.makeCompressed();
        }

        // undo augmentation matrix -> Probably not necessary here.
        visUnaugmentA = Eigen::SparseMatrix<double>(state_dim, state_dim);
        {
            for (int i = 0; i < LIDAR; i++)
                // Don't change main state.
                visUnaugmentA.insert(i, i) = 1;
            for (int i = LIDAR; i + POSE_DIM < state_dim; i++)
            {
                // Shift poses by one, dropping the first one and replacing
                // the last one (if present) with zeros
                visUnaugmentA.insert(i, i + POSE_DIM) = 1;
            }
            for (int i = state_dim; i < state_dim; ++i)
            { // won;t do anything for us.
                // Don't change hybrid map
                visUnaugmentA.insert(i, i) = 1;
            }
            visUnaugmentA.makeCompressed();
        }

        visAugH = Eigen::SparseMatrix<double>(POSE_DIM, state_dim);
        {
            for (int i = 0; i < 3; i++)
            {
                // Match new pose position to main state position.
                visAugH.insert(i, POS + i) = 1;
                visAugH.insert(i, LIDAR + i) = -1;
            }

            for (int i = 0; i < 4; i++)
            {
                // Match new pose orientation to main state orientation.
                visAugH.insert(3 + i, ORI + i) = 1;
                visAugH.insert(3 + i, LIDAR + 3 + i) = -1;
            }
            visAugH.makeCompressed();
        }

        // Noise corresponding to the first augmented pose.
        visAugQ = Eigen::SparseMatrix<double>(state_dim, state_dim);
        {
            for (int i = LIDAR; i < LIDAR + 3; i++)
                visAugQ.insert(i, i) = utils::square(params->init_pos_trail_noise);

            for (int i = LIDAR + 3; i < LIDAR + POSE_DIM; i++)
                visAugQ.insert(i, i) = utils::square(params->init_ori_trail_noise);

            visAugQ *= noise_scale;
            visAugQ.makeCompressed();
        }
    }

    // initialize odometry orientation from accelerometer sample
    void EKF::initialize_imu_global_orientation(
        const Eigen::Vector3d &xa, const Eigen::Vector3d &calc_grav)
    {
        Eigen::Quaterniond qq = Eigen::Quaterniond::FromTwoVectors(calc_grav, xa);
        Eigen::Vector3d q;
        (q << qq.w(), qq.x(), qq.y(), qq.z()).finished();

        m.segment(ORI, 4) = q;
        m.segment(GRAV, 3) = calc_grav;
        assert(q[3] == 0);
        (P.block(ORI, ORI, 4, 4) << 1, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 1, 0,
         0, 0, 0, 0)
            .finished();

        P.block(ORI, ORI, 4, 4) *= utils::square(params->init_ori_noise) * noise_scale;
    }

    // Forward propagation step: Predict
    void EKF::predict(
        double t, const Eigen::Vector3d &xg, const Eigen::Vector3d &xa,
        const Eigen::Vector3d &calc_grav, const Eigen::Vector3d &trans_lidar_imu,
        const Eigen::Matrix3d &rot_lidar_imu)
    {

        // dt: time
        double dt = 0.0;
        {
            if (!first_sample)
            {
                dt = t - prev_sampleT;
                time = t - first_sampleT;
            }
            else
            {
                first_sampleT = t;
                first_sample = false;
            }

            prev_sampleT = t;
            if (dt <= 0.0)
            {
                if (time > 0)
                    std::cout << "Skipping KF predict, dt " << dt << " <=0.0\n";
                return;
            }
        }

        // random walk bias for gyro
        if (params->gyro_process_noise > 0.0)
        {
            const double qc = utils::square(params->gyro_process_noise);
            const double theta = params->gyro_process_noise_rev;
            Q.block(Q_BGA_DRIFT, Q_BGA_DRIFT, 3, 3).setIdentity(3, 3) *= noise_scale * qc;

            if (theta > 0.0)
                Q.block(Q_BGA_DRIFT, Q_BGA_DRIFT, 3, 3) *= (1 - exp(-2 * dt * theta)) / (2 * theta);
        }

        // random walk bias for accelerometer
        if (params->acc_process_noise > 0.0)
        {
            const double qc = utils::square(params->acc_process_noise);
            const double theta = params->acc_process_noise_rev;
            Q.block(Q_BAA_DRIFT, Q_BAA_DRIFT, 3, 3).setIdentity(3, 3) *= noise_scale * qc;

            if (theta > 0.0)
                Q.block(Q_BAA_DRIFT, Q_BAA_DRIFT, 3, 3) *= (1 - exp(-2 * dt * theta)) / (2 * theta);
        }

        // xg: gyro measurement (in device coordinates)
        Eigen::Matrix4d S = calculate_S(xg, m, dt);
        Eigen::Matrix4d A = S.exp();

        // Obtain rotation and derivative {Eigen::Matrix3d, std::unique_ptr<Eigen::Matrix3d[]>}
        auto rot_dr = utils::extract_rot_dr(A * m.segment(ORI, 4));
        const Eigen::Matrix3d &R = std::get<0>(rot_dr);
        std::unique_ptr<Eigen::Matrix3d[]> dR = std::move(std::get<1>(rot_dr));

        const auto update_out = propagate_state(
            m, A, R, rot_lidar_imu, trans_lidar_imu, dt, calc_grav, xa,
            params->acc_process_noise_rev, params->gyro_process_noise);

        const Eigen::Vector3d &T_ab = std::get<0>(update_out);
        const Eigen::Vector4d &prev_quat = std::get<1>(update_out);

        // update jacobian wrt state and noise
        initialize_state_jacobians(dydx, dydq, T_ab, prev_quat, A, R, std::move(dR), xa, dt);

        // FORWARD PROPAGATION: P = dydx * P * dydx' + dydq * Q * dydq'
        P.topLeftCorner<INNER_DIM, INNER_DIM>() = dydx * P.topLeftCorner<INNER_DIM, INNER_DIM>() * dydx.transpose() + dydq * Q * dydq.transpose();
        tmp_0.noalias() = P.leftCols<INNER_DIM>().bottomRows(state_dim - INNER_DIM) * dydx.transpose();
        P.block(INNER_DIM, 0, state_dim - INNER_DIM, INNER_DIM) = tmp_0;
        tmp_0.noalias() = dydx * P.topRows<INNER_DIM>().rightCols(state_dim - INNER_DIM);
        P.block(0, INNER_DIM, INNER_DIM, state_dim - INNER_DIM) = tmp_0;
    }

    void EKF::motion_compensation_with_imu(frame::LidarImuInit::Ptr &meas)
    {
        auto &v_imu = meas->imu_buffer;
        v_imu.emplace_front(mc_tracker->last_imu);

        const double pcl_beg_time = meas->lidar_beg_time;
        const double imu_end_time = v_imu.back()->header.stamp.toSec();
        const double pcl_end_time = pcl_beg_time + meas->processed_frame->points.back().curvature / double(1000);

        // Initialize imu pose and tracking parameters
        mc_tracker->vel = velocity();
        mc_tracker->pos = position();
        const auto quat = orientation();
        mc_tracker->rot = utils::quat2rmat(quat);
        mc_tracker->imu_pose.clear();
        mc_tracker->populate_imu_pose(0.0);

        // initialize starting parameters for motion compensation.
        double dt = 0;
        Eigen::Vector3d xa, xg;
        Eigen::Vector4d prev_quat = quat;
        Eigen::MatrixXd curr_cov = P;

        for (auto it_imu = v_imu.begin(); it_imu != v_imu.end() - 1; ++it_imu)
        {
            auto &head = **it_imu;
            auto &tail = **(it_imu + 1);

            const auto tail_time_sec = tail.header.stamp.toSec();
            const auto head_time_sec = head.header.stamp.toSec();
            if (tail_time_sec < last_lidar_end_time)
                continue;

            (xg << 0.5 * (head.angular_velocity.x + tail.angular_velocity.x),
             0.5 * (head.angular_velocity.y + tail.angular_velocity.y),
             0.5 * (head.angular_velocity.z + tail.angular_velocity.z))
                .finished();
            (xa << 0.5 * (head.linear_acceleration.x + tail.linear_acceleration.x),
             0.5 * (head.linear_acceleration.y + tail.linear_acceleration.y),
             0.5 * (head.linear_acceleration.z + tail.linear_acceleration.z))
                .finished();

            // Eigen::Vector3d angvel_now(head.angular_velocity.x, head.angular_velocity.y, head.angular_velocity.z);
            // Eigen::Vector3d acc_now(head.linear_acceleration.x, head.linear_acceleration.y, head.linear_acceleration.z);

            if (head_time_sec < last_lidar_end_time)
                dt = tail_time_sec - last_lidar_end_time;
            else
                dt = tail_time_sec - head_time_sec;

            // reset matrices
            dydx_m.setZero();
            dydq_m.setZero();
            P_m = Eigen::MatrixXd::Zero(state_dim, state_dim);

            // Fill in parameters for current timestep
            Eigen::Matrix4d S = calculate_S(xg, m, dt);
            Eigen::Matrix4d A = S.exp();

            // Obtain rotation and derivative {Eigen::Matrix3d, std::unique_ptr<Eigen::Matrix3d[]>}
            auto rot_dr = utils::extract_rot_dr(A * prev_quat);
            const Eigen::Matrix3d &R = std::get<0>(rot_dr);
            std::unique_ptr<Eigen::Matrix3d[]> dR = std::move(std::get<1>(rot_dr));

            // scale up the accleration reading
            xa = xa / meas->get_mean_acc_norm() * gravity;
            Eigen::Vector3d T_ab = m.segment(BAT, 3).asDiagonal() * xa - m.segment(BAA, 3);

            initialize_state_jacobians(dydx_m, dydq_m, T_ab, prev_quat, A, R, std::move(dR), xa, dt);

            // Process covariance matrix updates -> dt is the noise in this case.
            initialize_process_covariance(P_m, dt, dt, dt, dt, dt, dt, dt, dt, 1.0);

            // Forward propagation using covariance.
            curr_cov.topLeftCorner<INNER_DIM, INNER_DIM>() = dydx_m * curr_cov.topLeftCorner<INNER_DIM, INNER_DIM>() * dydx_m.transpose() + P_m;
            tmp_0_m.noalias() = curr_cov.leftCols<INNER_DIM>().bottomRows(state_dim - INNER_DIM) * dydx_m.transpose();
            curr_cov.block(INNER_DIM, 0, state_dim - INNER_DIM, INNER_DIM) = tmp_0_m;
            tmp_0_m.noalias() = dydx_m * curr_cov.topRows<INNER_DIM>().rightCols(state_dim - INNER_DIM);
            curr_cov.block(0, INNER_DIM, INNER_DIM, state_dim - INNER_DIM) = tmp_0_m;

            /* global update rotation -dt because calcualte_S setup for local frame propagation*/
            S = calculate_S(xg, m, -dt);
            A = S.exp();
            prev_quat = A * prev_quat;
            mc_tracker->rot = utils::quat2rmat(prev_quat);

            /*global update velocity */
            T_ab = m.segment(BAT, 3).asDiagonal() * xa - m.segment(BAA, 3);
            mc_tracker->vel += (mc_tracker->rot.transpose() * T_ab + m.segment(GRAV, 3)) * dt;

            /*global update position */
            mc_tracker->pos += mc_tracker->vel * dt;

            /*global Update the accelrometer values */
            mc_tracker->acc_s_last = xa;
            mc_tracker->ang_vel_last = xg;

            // store in pose vector.
            mc_tracker->populate_imu_pose(tail_time_sec - pcl_beg_time);
        }

        /* calculate the position and attiude prediction at the frame end*/
        double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
        dt = note * (pcl_end_time - imu_end_time);

        /* Getting position and oritentation prediction at the end of the frame */
        Eigen::Vector3d pos_end, vel_end;
        Eigen::Matrix3d rot_end;
        { // xg contains most readings at end of imu vector
            Eigen::Matrix4d S = calculate_S(xg, m, dt);
            Eigen::Matrix4d A = S.exp();
            prev_quat = A * prev_quat;
            rot_end = utils::quat2rmat(prev_quat);

            Eigen::Vector3d T_ab = m.segment(BAT, 3).asDiagonal() * xa - m.segment(BAA, 3);
            vel_end = mc_tracker->vel + (rot_end.transpose() * T_ab + m.segment(GRAV, 3)) * dt;

            pos_end = mc_tracker->pos + vel_end * dt;
        }

        /*un-distort each lidar point*/
        mc_tracker->last_imu = v_imu.back();
        last_lidar_end_time = pcl_end_time;

        // position of lidar at end point.
        const auto pos_lidar_end = rot_end * m.segment(POS_IMU_LIDAR, 3) + pos_end;
        auto it_pcl = meas->processed_frame->points.end() - 1;

        for (auto it_kp = mc_tracker->imu_pose.end() - 1; it_kp != mc_tracker->imu_pose.begin(); it_kp--)
        {
            auto &head = *(it_kp - 1);

            // regular updates here
            const auto &R_imu = head.rot;
            const auto &pos_imu = head.pos;
            const auto &vel_imu = head.vel;
            const auto &acc_imu = head.acc;
            const auto &angvel_avr = head.gyr;

            for (; it_pcl->curvature / double(1000) > head.offset_time; it_pcl--)
            {
                dt = it_pcl->curvature / double(1000) - head.offset_time;

                /* ................ LIDAR-IMU-INITIALIZATION ..........................
                 * Transform to the 'scan-end' IMU frame（I_k frame）using only rotation
                 * Note: Compensation direction is INVERSE of Frame's moving direction
                 * So if we want to compensate a point at timestamp-i to the frame-e
                 * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame
                 */

                Eigen::Matrix3d R_i(R_imu * utils::ang_vel_to_rmat(angvel_avr, dt));
                Eigen::Vector3d T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * utils::square(dt) + R_i * m.segment(POS_IMU_LIDAR, 3) - pos_lidar_end);
                Eigen::Vector3d P_i(it_pcl->x, it_pcl->y, it_pcl->z);

                Eigen::Vector3d pos_comp = rot_end.transpose() * (R_i * P_i + T_ei);

                // This is in lidar frame coordinates
                it_pcl->x = pos_comp(0);
                it_pcl->y = pos_comp(1);
                it_pcl->z = pos_comp(2);

                if (it_pcl == meas->processed_frame->points.begin())
                    break;
            }
        }

        // copy into deskewed vector
        const size_t size = meas->processed_frame->points.size();
        const auto &frame = meas->processed_frame->points;
        meas->deskewed.resize(size);
        tbb::parallel_for(
            std::size_t(0), size,
            [&](std::size_t idx)
            {
                utils::Vec3d point(frame[idx].x, frame[idx].y, frame[idx].z);
                meas->deskewed[idx] = point;
            });
    }

    Eigen::Matrix4d EKF::calculate_S(const Eigen::Vector3d &xg, const Eigen::VectorXd &m, const double dt)
    {
        const Eigen::Vector3d w = xg - m.segment(BGA, 3);
        Eigen::Matrix4d S;
        (S << 0, -w[0], -w[1], -w[2],
         w[0], 0, -w[2], w[1],
         w[1], w[2], 0, -w[0],
         w[2], -w[1], w[0], 0)
            .finished();

        S *= -dt / 2;

        return S;
    }

    std::tuple<Eigen::Vector3d, Eigen::Vector4d> EKF::propagate_state(
        Eigen::VectorXd &state, Eigen::Matrix4d &A, const Eigen::Matrix3d &R, const Eigen::Matrix3d &rot_lidar_imu,
        const Eigen::Vector3d &trans_lidar_imu, const double dt, const Eigen::Vector3d &calc_grav,
        const Eigen::Vector3d &xa, const double acc_process_noise_rev, const double gyro_process_noise)
    {

        // imu world position
        state.segment(POS, 3) += state.segment(VEL, 3) * dt;

        // imu world velocity
        Eigen::Vector3d T_ab = state.segment(BAT, 3).asDiagonal() * xa - state.segment(BAA, 3);
        state.segment(VEL, 3) += (R.transpose() * T_ab + state.segment(GRAV, 3)) * dt;

        // orientation
        Eigen::Vector4d prev_quat = state.segment(ORI, 4);
        state.segment(ORI, 4) = A * state.segment(ORI, 4);

        // BGA BAA mean reversion
        if (acc_process_noise_rev > 0.0)
            state.segment(BAA, 3) *= exp(-dt * acc_process_noise_rev);

        if (gyro_process_noise > 0.0)
            state.segment(BGA, 3) *= exp(-dt * gyro_process_noise);

        // initialize gravity position e.t.c
        state.segment(GRAV, 3) = calc_grav;
        state.segment(POS_IMU_LIDAR, 3) = trans_lidar_imu;

        Eigen::Quaterniond quat(rot_lidar_imu);
        Eigen::Vector4d quat_coeffs = quat.coeffs();
        state.segment(ROT_IMU_LIDAR, 4) = quat_coeffs;

        return {T_ab, prev_quat};
    }

    void EKF::initialize_state_jacobians(
        Eigen::Matrix<double, INNER_DIM, INNER_DIM> &Fx, Eigen::Matrix<double, INNER_DIM, Q_DIM> &Fw,
        const Eigen::Vector3d &T_ab, const Eigen::Vector4d &prev_quat, const Eigen::Matrix4d &A,
        const Eigen::Matrix3d &R, std::unique_ptr<Eigen::Matrix3d[]> dR, const Eigen::Vector3d &xa,
        const double dt)
    {
        // Fx matrix -> dy/dx
        Fx.block(POS, POS, 3, 3).setIdentity(3, 3);
        Fx.block(VEL, VEL, 3, 3).setIdentity(3, 3);
        Fx.block(POS, VEL, 3, 3).setIdentity(3, 3) *= dt;
        Fx.block(BGA, BGA, 3, 3).setIdentity(3, 3);
        Fx.block(BAA, BAA, 3, 3).setIdentity(3, 3);
        Fx.block(BAT, BAT, 3, 3).setIdentity(3, 3);

        // use calculated value
        Fx.block(GRAV, GRAV, 3, 3).setIdentity(3, 3);
        Fx.block(POS_IMU_LIDAR, POS_IMU_LIDAR, 3, 3).setIdentity(3, 3);
        Fx.block(ROT_IMU_LIDAR, ROT_IMU_LIDAR, 4, 4).setIdentity(4, 4);

        // derivatives of velocity w.r.t quaternion
        for (int i = 0; i < 4; i++)
        {
            Fx.block(VEL, ORI + i, 3, 1) = dR[i].transpose() * T_ab * dt;
        }
        Fx.block(VEL, ORI, 3, 4) = Fx.block(VEL, ORI, 3, 4) * A;

        // derivative of quaternion wrt self
        Fx.block(ORI, ORI, 4, 4) = A;

        // derivatives of velocity wrt acceleration noise
        Fw.block(VEL, Q_ACC, 3, 3) = R.transpose() * dt;

        // quaternion derivatices w.r.t gyroscope noise
        Eigen::Matrix4d dS0, dS1, dS2;
        (dS0 << 0, dt / 2, 0, 0, -dt / 2, 0, 0, 0, 0, 0, 0, dt / 2, 0, 0, -dt / 2, 0).finished();
        (dS1 << 0, 0, dt / 2, 0, 0, 0, 0, -dt / 2, -dt / 2, 0, 0, 0, 0, dt / 2, 0, 0).finished();
        (dS2 << 0, 0, 0, dt / 2, 0, 0, dt / 2, 0, 0, -dt / 2, 0, 0, -dt / 2, 0, 0, 0).finished();
        Fw.block(ORI, Q_GYRO, 4, 1) = A * dS0 * prev_quat;
        Fw.block(ORI, Q_GYRO + 1, 4, 1) = A * dS1 * prev_quat;
        Fw.block(ORI, Q_GYRO + 2, 4, 1) = A * dS2 * prev_quat;
        Fw.block(BGA, Q_BGA_DRIFT, 3, 3).setIdentity(3, 3);
        Fw.block(BAA, Q_BAA_DRIFT, 3, 3).setIdentity(3, 3);

        // velocity and orientation wrt sensors
        Fw.block(VEL, Q_GYRO, 3, 3) = Fx.block(VEL, ORI, 3, 4) * Fw.block(ORI, Q_GYRO, 4, 3);

        // derivative of velocity w.r.t gyro bias
        Fx.block(VEL, BGA, 3, 3) = -Fw.block(VEL, Q_GYRO, 3, 3);

        // quaternion derivative wrt gyro bias
        Fx.block(ORI, BGA, 4, 3) = -Fw.block(ORI, Q_GYRO, 4, 3);

        // derivatives of the velocity w.r.t the acc. bias
        Fx.block(VEL, BAA, 3, 3) = -R.transpose() * dt;

        // derivatives of the velocity w.r.t the acc. transformation
        Fx.block(VEL, BAT, 3, 3) = R.transpose() * xa.asDiagonal() * dt;
    }

    void EKF::initialize_process_covariance(
        Eigen::MatrixXd &Pc, double init_pos_noise, double init_vel_noise,
        double init_bga_noise, double init_baa_noise, double init_bat_noise,
        double init_lidar_imu_time_noise, double init_pos_trail_noise,
        double init_ori_trail_noise, double noise_scale)
    {
        // imu - global extrinsic
        Pc.block(POS, POS, 3, 3).setIdentity(3, 3) *= utils::square(init_pos_noise);
        Pc.block(VEL, VEL, 3, 3).setIdentity(3, 3) *= utils::square(init_vel_noise);
        Pc.block(ORI, ORI, 4, 4).setIdentity(4, 4);

        // imu bias initialization
        Pc.block(BGA, BGA, 3, 3).setIdentity(3, 3) *= utils::square(init_bga_noise);
        Pc.block(BAA, BAA, 3, 3).setIdentity(3, 3) *= utils::square(init_baa_noise);
        Pc.block(BAT, BAT, 3, 3).setIdentity(3, 3) *= utils::square(init_bat_noise);
        Pc.block(GRAV, GRAV, 3, 3).setIdentity(3, 3) *= utils::square(init_lidar_imu_time_noise);

        // imu -lidar extrinsic
        Pc.block(POS_IMU_LIDAR, POS_IMU_LIDAR, 3, 3)
            .setIdentity(3, 3) *= utils::square(init_pos_noise);
        Pc.block(ROT_IMU_LIDAR, ROT_IMU_LIDAR, 4, 4).setIdentity(4, 4);

        // time estimation
        Pc(SFT, SFT) = utils::square(init_lidar_imu_time_noise);

        // initial parameters for trail poses
        const double pos_trail_noise = utils::square(init_pos_trail_noise);
        const double ori_trail_noise = utils::square(init_ori_trail_noise);

        for (int idx = 0; idx < lidar_pose_count; idx++)
        {
            int m_idx = LIDAR + idx * POSE_DIM;
            Pc.block(m_idx, m_idx, 3, 3).setIdentity(3, 3) *= pos_trail_noise;
            Pc.block(m_idx + 3, m_idx + 3, 4, 4).setIdentity(4, 4) *= ori_trail_noise;
        }

        Pc *= noise_scale;
    }

    void EKF::normalize_quaternions(bool only_current)
    {
        m.segment(ORI, 4).normalize();
        m.segment(ROT_IMU_LIDAR, 4).normalize();

        if (only_current)
            return;

        // normalize quaternion of trailing poses as well.
        for (int i = 0; i < lidar_pose_count; i++)
        {
            // If the segment is just zeroes (before first augments), then the normalized
            // segment becomes also just zeroes.
            m.segment(LIDAR + POSE_DIM * i + 3, 4).normalize();
        }
    }

    // // run kalman filters
    // void EKF::process(
    //     frame::LidarImuInit::Ptr &meas, double t, const Eigen::Vector3d &xg,
    //     const Eigen::Vector3d &xa, const Eigen::Vector3d &calc_grav,
    //     const Eigen::Vector3d &trans_lidar_imu, const Eigen::Matrix3d &rot_lidar_imu)
    // {
    //     // initialize orientation.
    //     if (!orientation_initialized)
    //     {
    //         initialize_imu_global_orientation(xa, calc_grav);
    //         orientation_initialized = true;
    //     }

    //     // kalman filter predict step
    //     predict(t, xg, xa, calc_grav, trans_lidar_imu, rot_lidar_imu);
    //     // normalize quaternions
    //     normalize_quaternions(true);
    //     // compensate motion
    //     motion_compensation(meas);
    // }

    void EKF::zero_vel_update(
        Eigen::VectorXd &state, Eigen::MatrixXd &K_g, Eigen::MatrixXd &P_cov,
        Eigen::MatrixXd &H_P, Eigen::LDLT<Eigen::MatrixXd> &inv_s,
        Eigen::MatrixXd &H_, Eigen::MatrixXd &R_, double r)
    {
        if (time - ZUPTtime < 0.25)
            return;

        ZUPTtime = time;
        was_stationary = true;

        H_ = Eigen::MatrixXd::Zero(3, VEL + 3); // truncated representation
        H_.block(0, VEL, 3, 3).setIdentity();
        R_ = Eigen::MatrixXd::Identity(3, 3) * r * noise_scale;

        // Note: for very non-trivial reasons, this is better here than
        // Eigen::Vector3d, which would cause a malloc in update
        Eigen::MatrixXd &y = tmp_1;
        y = Eigen::VectorXd::Zero(3);

        update(state, P_cov, y, H_, R_, K_g, H_P, tmp_0, inv_s);
    }

    void EKF::update_and_propagate()
    {
        // check if velocity is zero
        double curr_vel = speed();
        if (std::abs(curr_vel) < 1e-3)
        {
            // set the velocity to zero before making the update.
            zero_vel_update(m, K, P, HP, invS, H, R, params->visualZuptR);
            /*
             * Because using the pose augmentation scheme an issue can arise when the device is stationary
             * it can cause the pose to degenerate into na [max number of pose trail] identical copies of a single point
             * which can destabilize the system. It's more crucial for camera systems but good practice to carry along
             * Ref: https://arxiv.org/pdf/2106.11857.pdf Section 3.9.
             */
            update_undo_augmentation();
        }

        update_visual_pose_aug();
    }

    void EKF::update_visual_pose_aug()
    {
        int pose_idx_to_discard = lidar_pose_count - 1;
        Eigen::SparseMatrix<double> &A = visAugA.at(pose_idx_to_discard);

        // extra prediction step
        tmp_0.noalias() = A * m;
        m = tmp_0;
        tmp_1.noalias() = P * A.transpose();
        tmp_0.noalias() = A * tmp_1;

        P = tmp_0 + visAugQ;
        R = Eigen::MatrixXd::Identity(POSE_DIM, POSE_DIM) * 1e-9 * noise_scale;

        HP.noalias() = visAugH * P;
        S = R;
        S.noalias() += HP * visAugH.transpose();
        invS.compute(S);
        K = invS.solve(HP).transpose();
        Eigen::MatrixXd &v = tmp_0;
        v.noalias() = -visAugH * m;
        m.noalias() += K * v;

        update_common_joseph_form(P, visAugH, R, K, tmp_0, tmp_1);
        maintain_positive_semi_definite();
        normalize_quaternions();

        augment_times.emplace_back(get_current_time());
        if (augment_count > lidar_pose_count)
            augment_count++;
        else
            augment_times.erase(augment_times.begin());

        assert(static_cast<int>(augment_times.size()) == augment_count);
    }

    void EKF::update_undo_augmentation()
    {
        tmp_0.noalias() = visUnaugmentA * m;
        m = tmp_0;

        /*
         * No need to add any noise.. except for most recent update of pose trail
         * which is ignored anyways -> Q=0
         */
        tmp_0.noalias() = P * visUnaugmentA.transpose();
        P.noalias() = visUnaugmentA * tmp_0; // P = A P A^T
        augment_times.pop_back();
        augment_count--;

        // ensure positive semi definite and normalize quaterinions
        maintain_positive_semi_definite();
        normalize_quaternions();

        // additional check for sanity.
        assert(static_cast<int>(augment_times.size()) == augment_count);
    }

    void EKF::maintain_positive_semi_definite()
    {
        // maintains symmetric nature of the P matrix.
        // numerical errors can add non symmetry which begins to accumulate over time.
        tmp_0.noalias() = 0.5 * (P + P.transpose());
        P.swap(tmp_0);
    }

    double EKF::get_current_time() const
    {
        return first_sampleT + time;
    }

    Eigen::Vector3d EKF::position() const
    {
        return m.segment(POS, 3);
    }

    Eigen::Vector3d EKF::velocity() const
    {
        return m.segment(VEL, 3);
    }

    Eigen::Vector4d EKF::orientation() const
    {
        return m.segment(ORI, 4);
    }

    Eigen::Vector3d EKF::gravity_check() const
    {
        return m.segment(GRAV, 3);
    }

    double EKF::speed() const
    {
        return m.segment(VEL, 3).norm();
    }
}