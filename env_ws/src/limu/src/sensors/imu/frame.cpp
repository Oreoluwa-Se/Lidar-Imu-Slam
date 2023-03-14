#include "limu/sensors/imu/frame.hpp"
#include <ros/ros.h>

namespace
{
    const int max_init_count = 200;
}

namespace frame
{
    void Imu::process_data(const sensor_msgs::Imu::ConstPtr &msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex);

        time_in = msg->header.stamp.toSec();

        if (count < config->reset)
        {
            count++;
            // calculate running mean.
            if (config->type == CoordinateType::ned)
            {
                const auto a_cc = utils::Vec3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
                mean_acc += (a_cc - mean_acc) / count;
            }
            else if (config->type == CoordinateType::enu)
            {
                const auto a_cc = utils::Vec3d(msg->linear_acceleration.y, msg->linear_acceleration.x, -1 * msg->linear_acceleration.z);
                mean_acc += (a_cc - mean_acc) / count;
            }

            if (count > 1)
                period += (time_in - prev_time_in - period) / (count - 1);

            if (count == config->reset - 1)
            {
                std::cout << "========" << std::endl;
                std::cout << "Acceleration norm  : " << mean_acc.norm() << std::endl;
                if (period > 0.01)
                {
                    ROS_WARN("IMU data frequency : %f HZ", 1 / period);
                    ROS_WARN("IMU data frequency too low. Higher than 150 Hz is recommended.");
                }
                std::cout << "========" << std::endl;
            }
        }

        prev_time_in = time_in;
        msg_new.reset(new sensor_msgs::Imu(*msg));
    }

    void Imu::imu_time_compensation(double imu_time_wrt_lidar, double imu_lidar_lag)
    {
        msg_new->header.stamp = ros::Time().fromSec(time_in - imu_time_wrt_lidar - imu_lidar_lag);
    }

    void Imu::update_buffer()
    {
        std::lock_guard<std::mutex> lock(data_mutex);
        double time = msg_new->header.stamp.toSec();

        if (time < prev_timestamp)
        {
            ROS_WARN("IMU loop back, clear IMU buffer.");
            buffer.clear();
        }

        buffer.emplace_back(msg_new);
        prev_timestamp = time;
    }

    void Imu::init(kalman::EKF::Ptr &ekf, LidarImuInit::Ptr &meas)
    {
        /**
        ** 1. initializing the gravity, gyro bias, acc and gyro covariance
        ** 2. normalize the acceleration measurements to unit gravity
        **/
        ROS_INFO("IMU Initializing: %.1f %%", double(init_iter_num) / max_init_count * 100);
        Eigen::Vector3d curr_acc, curr_gyr;

        if (first_frame)
        {
            // reset();
            init_iter_num = 1;
            first_frame = false;
            const auto &imu_acc = meas->imu_buffer.front()->linear_acceleration;
            const auto &imu_gyro = meas->imu_buffer.front()->angular_velocity;

            (mean_acc << imu_acc.x, imu_acc.y, imu_acc.z).finished();
            (mean_gyro << imu_gyro.x, imu_gyro.y, imu_gyro.z).finished();
            first_lidar_time = meas->lidar_beg_time;
        }

        for (const auto &imu : meas->imu_buffer)
        {
            const auto &imu_acc = imu->linear_acceleration;
            const auto &imu_gyro = imu->angular_velocity;

            (curr_acc << imu_acc.x, imu_acc.y, imu_acc.z).finished();
            (curr_gyr << imu_gyro.x, imu_gyro.y, imu_gyro.z).finished();

            // update mean and covariances
            mean_acc += (curr_acc - mean_acc) / double(init_iter_num);
            mean_gyro += (curr_gyr - mean_gyro) / double(init_iter_num);

            const auto &N = init_iter_num;
            cov_acc = cov_acc * (N - 1.0) / N + (curr_acc - mean_acc).cwiseProduct(curr_acc - mean_acc) * (N - 1.0) / (N * N);
            cov_gyr = cov_gyr * (N - 1.0) / N + (curr_gyr - mean_gyro).cwiseProduct(curr_gyr - mean_gyro) * (N - 1.0) / (N * N);

            init_iter_num++;
        }

        // setgravity, initialize imu
        Eigen::Vector3d calc_grav = -mean_acc / mean_acc.norm() * gravity;
        Eigen::Vector3d xa = Eigen::Vector3d::Zero();
        ekf->initialize_imu_global_orientation(xa, calc_grav);
        ekf->mc_tracker->last_imu = meas->imu_buffer.back();
    }

    void Imu::initialize_propagate_undistort(kalman::EKF::Ptr &ekf, LidarImuInit::Ptr &meas)
    {
        if (need_init)
        {
            if (!lidar_imu_done)
            {
                init(ekf, meas);
                need_init = true;
                ekf->mc_tracker->last_imu = meas->imu_buffer.back();
                if (init_iter_num > max_init_count)
                {
                    cov_acc *= utils::square(gravity / mean_acc.norm());
                    need_init = false;
                    const Eigen::Vector3d grav_pred = ekf->gravity_check();
                    ROS_INFO("IMU Initialization Done: Gravity: %.4f %.4f %.4f", grav_pred[0], grav_pred[1], grav_pred[2]);
                }
            }
            else
            {
                need_init = false;
                ekf->mc_tracker->last_imu = meas->imu_buffer.back();
            }

            return;
        }

        // compensate motion
        ekf->motion_compensation_with_imu(meas);
        ROS_INFO("Motion Compensated with IMU");
    }
}