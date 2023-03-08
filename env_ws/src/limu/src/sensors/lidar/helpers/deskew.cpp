
#include "limu/sensors/lidar/helpers/deskew.hpp"
#include <tbb/parallel_for.h>
#include <iostream>
#include <numeric>

namespace lidar
{

    utils::Vec3dVector MotionCompensator::deskew_scan(
        const utils::PointCloudXYZI &frame, const std::vector<double> &timestamps,
        const SE3d &start_pose, const SE3d &end_pose)
    {
        const auto twist = utils::delta_pose(start_pose, end_pose);
        const size_t size = frame.points.size();
        // / points are normalized.. 0.0 <-> 1.0
        utils::Vec3dVector deskewed_points(size);
        tbb::parallel_for(
            std::size_t(0), size,
            [&](std::size_t idx)
            {
                utils::Vec3d point(frame.points[idx].x, frame.points[idx].y, frame.points[idx].z);
                // why this i guess trying to calculate based on center point?
                const auto motion = Sophus::SE3d::exp((timestamps[idx] - mid_pose_timestamp) * twist);
                deskewed_points[idx] = motion * point;
            });

        return deskewed_points;
    }

}