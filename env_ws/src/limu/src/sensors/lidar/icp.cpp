#include "limu/sensors/lidar/icp.hpp"
#include <tsl/robin_map.h>

namespace
{
    // function for downsampling point_cloud
    utils::Vec3dVector voxel_downsample(const utils::Vec3dVector &frame, double vox_size)
    {
        tsl::robin_map<utils::Voxel, utils::Vec3d, utils::VoxelHash> grid;
        grid.reserve(frame.size());
        for (const auto &point : frame)
        {
            const auto vox = utils::get_vox_index(point, vox_size);
            if (grid.contains(vox))
                continue;
            grid.insert(std::make_pair(vox, point));
        }

        utils::Vec3dVector downsampled;
        downsampled.reserve(grid.size());
        std::transform(
            grid.begin(), grid.end(),
            std::back_inserter(downsampled),
            [](const auto &kv)
            { return kv.second; });

        return downsampled;
    }

    std::vector<utils::Point::Ptr> voxel_downsample(const std::vector<utils::Point::Ptr> &frame, double vox_size)
    {
        tsl::robin_map<utils::Voxel, utils::Point::Ptr, utils::VoxelHash> grid;
        grid.reserve(frame.size());
        for (const auto &point : frame)
        {
            const auto vox = utils::get_vox_index(point, vox_size);
            if (grid.contains(vox))
                continue;
            grid.insert(std::make_pair(vox, point));
        }

        std::vector<utils::Point::Ptr> downsampled;
        downsampled.reserve(grid.size());
        std::transform(
            grid.begin(), grid.end(),
            std::back_inserter(downsampled),
            [](const auto &kv)
            { return kv.second; });

        return downsampled;
    }
}

namespace lidar
{
    double KissICP::ICP_setup()
    {
        // Get motion prediction and adaptive threshold
        const double sigma = get_adaptive_threshold();

        // compute initial ICP guess
        const SE3d pred = get_prediction_model();
        const auto last_pose = poses.empty() ? SE3d() : poses.back();
        init_guess = last_pose * pred;

        return sigma;
    }

    void KissICP::update_map(PointsPtrVector &map_points, const SE3d &new_pose, bool transform_point)
    {
        const auto model_dev = init_guess.inverse() * new_pose;
        adaptive_threshold->update_model_deviation(model_dev);

        // points here have been transformed already.. so using identity to maintain code structure
        local_map->update(map_points, new_pose, transform_point);
        poses.emplace_back(new_pose);
    }

    VoxelDownsampleTuple KissICP::voxelize(const PointsPtrVector &frame)
    {
        const double vox_size = voxel_size;
        // convert point clouds to eigen
        const auto downsample = voxel_downsample(frame, vox_size * 0.5);
        const auto source = voxel_downsample(downsample, vox_size * 1.5);

        return {source, downsample};
    }

    // used for robust kernel during ICP
    double KissICP::get_adaptive_threshold()
    {
        if (!has_moved())
            return initial_threshold;

        return adaptive_threshold->compute_threshold();
    }

    SE3d KissICP::get_prediction_model() const
    {
        const std::size_t N = poses.size();
        const SE3d pred = Sophus::SE3d();
        if (N < 2)
            return pred;

        return poses[N - 2].inverse() * poses[N - 1];
    }

    bool KissICP::has_moved()
    {
        if (poses.empty())
            return false;

        const double motion = (poses.front().inverse() * poses.back()).translation().norm();
        return motion > 5.0 * min_motion_th;
    }
}