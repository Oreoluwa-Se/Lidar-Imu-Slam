/*Stores general Lidar properties and information*/
#ifndef IMU_FRAME_HPP
#define IMU_FRAME_HPP

#include "common.hpp"
#include <ros/ros.h>
#include "sensor_msgs/Imu.h"
#include "limu/sensors/sync_frame.hpp"

namespace frame
{
    // forward declaration of synchronized frame.
    // struct LidarImuInit;

    // Imu coordinate type
    enum class CoordinateType
    {
        ned, // North-East-Down
        enu  // North-East-Down
    };

    class Imu
    {
    public:
        typedef std::shared_ptr<Imu> Ptr;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        struct ProcessInfo
        {
            typedef std::shared_ptr<ProcessInfo> Ptr;
            int reset;
            CoordinateType type;
        };

        explicit Imu(ros::NodeHandle &nh)
            : config(std::make_shared<ProcessInfo>()),
              prev_timestamp(0.0), period(0.0), prev_time_in(0.0),
              time_in(0.0), count(0), enabled(false),
              initialized(false), need_init(true), first_frame(true),
              clear_calib(false)
        {

            nh.param<int>("imu_reset", config->reset, 100);
            int type;
            nh.param<int>("imu_coordinate", type, 1);
            if (type == 1)
                config->type = CoordinateType::ned;
            else
                config->type = CoordinateType::enu;

            reset();
        }

        void imu_init(frame::LidarImuInit::Ptr &sync);
        void process_data(const sensor_msgs::Imu::ConstPtr &msg);
        void imu_time_compensation(double imu_time_wrt_lidar, double imu_lidar_lag);
        void update_buffer();
        void reset();
        void process_package(frame::LidarImuInit::Ptr &sync);
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
            return buffer.front()->header.stamp.toSec();
        }

        sensor_msgs::Imu::Ptr buffer_front()
        {
            std::unique_lock<std::mutex> lock(data_mutex);
            return buffer.front();
        }

        void pop()
        {
            std::unique_lock<std::mutex> lock(data_mutex);
            buffer.pop_front();
        }

        void recycle()
        {
            imu_last = imu_next;
            imu_next = *(buffer_front());
            pop();
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

        void update_header_stamp(double time_lag_IMU_wtr_lidar)
        {
            std::unique_lock<std::mutex> lock(data_mutex);
            for (int i = 0; i < buffer.size(); i++)
            {
                buffer[i]->header.stamp = ros::Time().fromSec(buffer[i]->header.stamp.toSec() - time_lag_IMU_wtr_lidar);
            }
        }

    public:
        int init_iter_num = 1;
        bool enabled, need_init;
        sensor_msgs::Imu::Ptr imu_last_ptr;
        sensor_msgs::Imu imu_last, imu_next;
        utils::Vec3d mean_acc, mean_gyro, cov_acc, cov_gyr;
        bool clear_calib;

    private:
        // attributes
        double time_in, prev_time_in, period, prev_timestamp;
        bool initialized, first_frame;
        std::deque<sensor_msgs::Imu::Ptr> buffer;
        sensor_msgs::Imu::Ptr msg_new;
        ProcessInfo::Ptr config;
        std::mutex data_mutex;
        int count;
    };
}
#endif