#include "limu/sensors/imu/frame.hpp"

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
}