#include "limu/sensors/imu/frame.hpp"
#include <ros/ros.h>

namespace
{
    const int max_init_count = 100;
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
            ROS_WARN("[WARN] IMU loop back, clear IMU buffer.");
            buffer.clear();
            clear_calib = true;
        }

        buffer.emplace_back(msg_new);
        prev_timestamp = time;
    }

    void Imu::reset()
    {
        ROS_WARN("[WARN] Reset Imu process.");
        (mean_acc << 0.0, 0.0, -1.0).finished();
        mean_gyro = utils::Vec3d::Zero();
        init_iter_num = 1;
        need_init = true;
    }

    void Imu::process_package(frame::LidarImuInit::Ptr &sync)
    {
        if (enabled)
        {
            if (sync->imu_buffer.empty())
                return;
            ROS_ASSERT(sync->processed_frame != nullptr);

            if (need_init)
            {
                imu_init(sync);
                need_init = true;

                if (init_iter_num > max_init_count)
                    need_init = false;
            }
            return;
        }
    }

    void Imu::imu_init(frame::LidarImuInit::Ptr &sync)
    {
        /** 1. initializing the gravity, gyro bias, acc and gyro covariance
         ** 2. normalize the acceleration measurenments to unit gravity **/
        const double percent_initialized = double(init_iter_num) / max_init_count * 100;
        ROS_INFO("IMU Initializing: %.1f %%", percent_initialized);

        if (first_frame)
        {
            reset();
            first_frame = false;

            // Initialize mean acceleration and mean gyro using the first IMU measurement
            const auto &imu_acc = sync->imu_buffer.front()->linear_acceleration;
            const auto &imu_gyr = sync->imu_buffer.front()->angular_velocity;
            mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
            mean_gyro << imu_gyr.x, imu_gyr.y, imu_gyr.z;
        }

        // Update mean acceleration and mean gyro using all IMU measurements in the buffer
        for (const auto &imu : sync->imu_buffer)
        {
            const auto &imu_acc = imu->linear_acceleration;
            const auto &imu_gyr = imu->angular_velocity;
            const utils::Vec3d cur_acc{imu_acc.x, imu_acc.y, imu_acc.z};
            const utils::Vec3d cur_gyr{imu_gyr.x, imu_gyr.y, imu_gyr.z};

            mean_acc += (cur_acc - mean_acc) / init_iter_num;
            mean_gyro += (cur_gyr - mean_gyro) / init_iter_num;

            init_iter_num++;
        }

        sync->mean_acc = mean_acc;
    }
}