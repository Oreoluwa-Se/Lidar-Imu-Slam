#include "limu/sensors/imu/frame.hpp"
#include "geometry_msgs/Vector3.h"
#include <ros/ros.h>

namespace
{
    utils::Vec3d convert_ned_to_enu(const utils::Vec3d &ned)
    {
        utils::Vec3d enu;
        enu << ned.y(), ned.x(), -ned.z();
        return enu;
    }
}
namespace frame
{
    void Imu::process_data(const sensor_msgs::Imu::ConstPtr &msg)
    {
        std::unique_lock<std::mutex> lock(data_mutex);

        time_in = msg->header.stamp.toSec();
        prev_time_in = time_in;
        double time = msg->header.stamp.toSec();

        if (time < prev_timestamp)
        {
            ROS_WARN("[WARN] IMU loop back, clear IMU buffer.");
            buffer.clear();
        }

        utils::Vec3d cur_acc(
            msg->linear_acceleration.x,
            msg->linear_acceleration.y,
            msg->linear_acceleration.z);

        utils::Vec3d cur_gyr(
            msg->angular_velocity.x,
            msg->angular_velocity.y,
            msg->angular_velocity.z);

        // convert ned to enu coordinates
        if (config->is_ned)
        {
            cur_acc = convert_ned_to_enu(cur_acc);
            cur_gyr = convert_ned_to_enu(cur_gyr);
        }

        buffer.emplace_back(std::make_shared<utils::ImuData>(cur_acc, cur_gyr, time));
        prev_timestamp = time;
    }

    void Imu::reset()
    {
        ROS_WARN("[WARN] Reset Imu process.");
        mean_acc = utils::Vec3d(0.0, 0.0, -1.0);
        mean_gyro = utils::Vec3d::Zero();
        init_iter_num = 1;
        need_init = true;
    }

    void Imu::process_package(frame::LidarImuInit::Ptr &sync)
    {
        bool assigned = false;
        if (!enabled)
            return;

        if (sync->imu_buffer.empty())
            return;

        if (need_init)
        {
            imu_init(sync);
            need_init = true;
            if (!assigned)
            {
                sync->mean_acc = mean_acc;
                assigned = true;
            }

            if (init_iter_num > config->reset)
            {
                ROS_INFO("IMU Initialized: %d %%", 100);
                need_init = false;
            }
        }

        if (!assigned)
            sync->mean_acc = mean_acc;
    }

    void Imu::imu_init(frame::LidarImuInit::Ptr &sync)
    {
        /** 1. initializing the gravity, gyro bias, acc and gyro covariance
         ** 2. normalize the acceleration measurenments to unit gravity **/
        const double percent_initialized = double(init_iter_num) / config->reset * 100;
        ROS_INFO("IMU Initializing: %.1f %%", percent_initialized);

        if (first_frame)
        {
            reset();
            init_iter_num = 1;

            // Initialize mean acceleration and mean gyro using the first IMU measurement
            const auto data = sync->imu_buffer.front();

            if (data)
            {
                mean_acc = data->acc;
                mean_gyro = data->gyro;

                first_frame = false;
            }
        }

        // Update mean acceleration and mean gyro using all IMU measurements in the buffer
        for (const auto &data : sync->imu_buffer)
        {
            mean_acc += (data->acc - mean_acc) / init_iter_num;
            mean_gyro += (data->gyro - mean_gyro) / init_iter_num;

            init_iter_num++;
        }
    }

    void Imu::collect_data(double end_time, std::deque<utils::ImuData::Ptr> &imu_buffer)
    {
        imu_buffer.clear();
        std::unique_lock<std::mutex> lock(data_mutex);

        // Find the first imu point that starts immediately after end_time
        auto upper_iter = std::upper_bound(
            buffer.begin(), buffer.end(), end_time,
            [](const auto &time, const auto &imu)
            {
                return time < imu->timestamp;
            });

        // copy available points and then erase after
        std::move(buffer.begin(), upper_iter, std::back_inserter(imu_buffer));
        buffer.erase(buffer.begin(), upper_iter);
    }
}