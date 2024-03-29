/*Stores general Lidar properties and information*/
#ifndef IMU_FRAME_HPP
#define IMU_FRAME_HPP

#include "common.hpp"
#include <ros/ros.h>
#include "sensor_msgs/Imu.h"
#include "limu/kalman/ekf.hpp"
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
              time_in(0.0), count(0), first_lidar_time(0.0), enabled(false),
              initialized(false), lidar_imu_done(false), need_init(true),
              imu_deskwed(false), first_frame(false)
        {
            nh.param<int>("imu_reset", config->reset, 100);
            int type;
            nh.param<int>("imu_coordinate", type, 1);
            if (type == 1)
                config->type = CoordinateType::ned;
            else
                config->type = CoordinateType::enu;
        }

        void initialize_propagate_undistort(kalman::EKF::Ptr &ekf, LidarImuInit::Ptr &meas);
        void process_data(const sensor_msgs::Imu::ConstPtr &msg);
        void imu_time_compensation(double imu_time_wrt_lidar, double imu_lidar_lag);
        void update_buffer();
        void init(kalman::EKF::Ptr &ekf, LidarImuInit::Ptr &meas);

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

    public:
        int init_iter_num = 1;
        bool enabled;
        double first_lidar_time;

    private:
        // attributes
        double time_in, prev_time_in, period, prev_timestamp;
        bool initialized, lidar_imu_done, need_init, imu_deskwed, first_frame;
        utils::Vec3d mean_acc = utils::Vec3d::Zero();
        utils::Vec3d mean_gyro = utils::Vec3d::Zero();
        utils::Vec3d cov_acc, cov_gyr;
        std::deque<sensor_msgs::Imu::Ptr> buffer;
        sensor_msgs::Imu::Ptr msg_new;
        ProcessInfo::Ptr config;
        std::mutex data_mutex;
        int count;
    };
}
#endif