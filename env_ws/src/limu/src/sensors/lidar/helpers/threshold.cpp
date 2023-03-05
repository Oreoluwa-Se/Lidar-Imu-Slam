#include "limu/sensors/lidar/helpers/threshold.hpp"

namespace
{
    double compute_model_error(const Sophus::SE3d &model_dev, double max_range)
    {
        const double theta = Eigen::AngleAxisd(model_dev.rotationMatrix()).angle();
        const double delta_rot = 2.0 * max_range * std::sin(theta / 2.0);
        const double delta_trans = model_dev.translation().norm();

        return delta_rot + delta_trans;
    }
}
namespace lidar
{
    double AdaptiveThreshold::compute_threshold()
    {
        double model_error = compute_model_error(model_deviation, max_range);
        if (model_error > min_motion_th)
        {
            model_error_sq += model_error * model_error;
            num_samples++;
        }

        if (num_samples < 1)
            return init_threshold;

        return std::sqrt(model_error_sq / num_samples);
    }
}