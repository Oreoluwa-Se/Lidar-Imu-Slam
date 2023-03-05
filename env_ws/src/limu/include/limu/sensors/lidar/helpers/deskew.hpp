#ifndef DESKEW_HPP
#define DESKEW_HPP
#include "common.hpp"

namespace lidar
{
    using SE3d = Sophus::SE3d;
    class MotionCompensator
    {
    public:
        explicit MotionCompensator()
            : mid_pose_timestamp(0.5){};

        utils::Vec3dVector deskew_scan(
            const utils::PointCloudXYZI &frame, const std::vector<double> &timestamps,
            const SE3d &start_pose, const SE3d &end_pose);

    private:
        double mid_pose_timestamp;
    };
}
#endif