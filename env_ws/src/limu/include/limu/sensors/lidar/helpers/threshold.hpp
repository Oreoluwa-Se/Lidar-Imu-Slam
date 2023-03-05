#ifndef THRESHOLD_HPP
#define THRESHOLD_HPP

#include "common.hpp"

namespace lidar
{
    using SE3d = Sophus::SE3d;
    struct AdaptiveThreshold
    {
        AdaptiveThreshold(double init_threshold, double min_motion_th, double max_range)
            : init_threshold(init_threshold), min_motion_th(min_motion_th),
              max_range(max_range){};

        inline void update_model_deviation(const SE3d &curr_dev)
        {
            model_deviation = curr_dev;
        }

        double compute_threshold();

        // config parameters
        double init_threshold;
        double min_motion_th;
        double max_range;

        // computation cache
        double model_error_sq = 0.0;
        int num_samples = 0;

        // local icp correction to be applied to predicted pose.
        SE3d model_deviation = SE3d();
    };
}

#endif