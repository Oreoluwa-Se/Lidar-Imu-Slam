#ifndef SYNC_FRAME_HPP
#define SYNC_FRAME_HPP

#include "limu/sensors/lidar/frame.hpp"
#include "limu/sensors/imu/frame.hpp"
#include "common.hpp"

namespace frame
{
    using PointCloud = utils::PointCloudXYZI;
    struct LidarImuInit
    {
    public:
        typedef std::shared_ptr<LidarImuInit> Ptr;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        LidarImuInit()
        {
            lidar_beg_time = 0.0;
            this->processed_frame.reset(new PointCloud());
            this->original_frame.reset(new PointCloud());
        }

        double lidar_beg_time;
        std::vector<double> time_buffer;
        std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
        utils::Vec3d mean_acc_norm;

        // points for map update and for icp
        PointCloud::Ptr processed_frame;
        PointCloud::Ptr original_frame;
    };
}

#endif