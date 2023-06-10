#ifndef IMU_FRAME_HPP
#define IMU_FRAME_HPP

#include "common.hpp"
#include <ros/ros.h>
#include "sensor_msgs/Imu.h"
#include "limu/sensors/sync_frame.hpp"

namespace frame
{
    class Imu
    {
    public:
        typedef std::shared_ptr<Imu> Ptr;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        struct ProcessInfo
        {
            typedef std::shared_ptr<ProcessInfo> Ptr;
            int reset;
            bool is_ned;
        };

        explicit Imu(std::string &params_loc)
            : config(std::make_shared<ProcessInfo>()),
              prev_timestamp(0.0), period(0.0), prev_time_in(0.0),
              time_in(0.0), enabled(false),
              initialized(false), need_init(true), first_frame(true)
        {
            YAML::Node node = YAML::LoadFile(params_loc);
            config->reset = node["imu_preprocessing"]["reset"].as<double>();
            config->is_ned = node["common"]["imu_is_ned"].as<bool>();
            reset();
        }

        void imu_init(frame::LidarImuInit::Ptr &sync);
        void process_data(const sensor_msgs::Imu::ConstPtr &msg);
        void collect_data(double end_time, std::deque<utils::ImuData::Ptr> &imu_buffer);
        void process_package(frame::LidarImuInit::Ptr &sync);
        void reset();

        double return_prev_ts()
        {
            std::unique_lock<std::mutex> lock(data_mutex);
            return prev_timestamp;
        }

        bool buffer_empty()
        {
            std::unique_lock<std::mutex> lock(data_mutex);
            return buffer.empty();
        }

        double get_prev_timestamp()
        {
            std::unique_lock<std::mutex> lock(data_mutex);
            return prev_timestamp;
        }

        double get_front_time()
        {
            std::unique_lock<std::mutex> lock(data_mutex);
            return buffer.front()->timestamp;
        }

        utils::ImuData::Ptr buffer_front()
        {
            std::unique_lock<std::mutex> lock(data_mutex);
            return buffer.front();
        }

        void pop()
        {
            std::unique_lock<std::mutex> lock(data_mutex);
            buffer.pop_front();
        }

        double mean_acc_norm()
        {
            return mean_acc.norm();
        }

        bool enabled_check()
        {
            std::unique_lock<std::mutex> lock(data_mutex);
            return enabled;
        }

    public:
        int init_iter_num = 1;
        bool enabled, need_init;
        utils::Vec3d mean_acc, mean_gyro, cov_acc, cov_gyr;

    private:
        // attributes
        double time_in, prev_time_in, period, prev_timestamp;
        bool initialized, first_frame;
        std::deque<utils::ImuData::Ptr> buffer;
        ProcessInfo::Ptr config;
        std::mutex data_mutex;
    };
}
#endif