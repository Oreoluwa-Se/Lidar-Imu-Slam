#ifndef SYNC_FRAME_HPP
#define SYNC_FRAME_HPP

#include "sensor_msgs/Imu.h"
#include "common.hpp"

namespace frame
{

    using PointCloud = utils::PointCloudXYZI;
    struct LidarImuInit
    {
        typedef std::shared_ptr<LidarImuInit> Ptr;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        LidarImuInit()
        {
            lidar_last_time = 0.0;
            lidar_beg_time = 0.0;
            this->processed_frame.reset(new PointCloud());
        }

        double lidar_beg_time, lidar_last_time;
        std::vector<double> time_buffer;
        std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
        utils::Vec3d mean_acc;
        utils::Vec3d curr_ang_vel, curr_acc;

        PointCloud::Ptr processed_frame; // Holds frames frames processed from initial reading.
        utils::Vec3dVector map_points;   // stored in the map
        utils::Vec3dVector icp_points;   // used for estimating pose

        inline double get_mean_acc_norm()
        {
            return mean_acc.norm();
        }
        // holds points and equivalent timestamps
        // std::vector<std::pair<utils::Vec3d, double>> points_ts
    };
}

#endif
