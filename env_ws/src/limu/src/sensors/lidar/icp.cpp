#include "limu/sensors/lidar/icp.hpp"
#include <tsl/robin_map.h>
#include <tbb/parallel_for.h>
#include <tbb/concurrent_vector.h>

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
}

namespace lidar
{

    utils::Vec3dVector KissICP::deskew_scan(const utils::PointCloudXYZI &frame, const std::vector<double> &timestamps)
    {
        const auto e_frame = utils::pointcloud2eigen(frame);
        const size_t num_poses = poses.size();
        if (!config.deskew)
            return e_frame;

        if (num_poses <= 2)
            return e_frame;

        return compensator.deskew_scan(frame, timestamps, poses[num_poses - 2], poses[num_poses - 1]);
    }

    ReturnTuple KissICP::register_frame(
        const utils::PointCloudXYZI &frame, const std::vector<double> &timestamps)
    {
        const auto post_screw_frame = deskew_scan(frame, timestamps);

        return register_frame(post_screw_frame);
    }

    // register frame
    ReturnTuple KissICP::register_frame(const utils::Vec3dVector &frame)
    {
        // returns {source, frame_downsample}
        utils::Vec3_Vec3Tuple processed_frame = voxelize(frame, config.voxel_size);
        auto &down_sampled = std::get<1>(processed_frame);
        auto &source = std::get<0>(processed_frame);

        // Get motion prediction and adaptive threshold
        const double sigma = get_adaptive_threshold();

        // compute initial ICP guess
        const SE3d pred = get_prediction_model();
        const auto last_pose = poses.empty() ? SE3d() : poses.back();
        const auto init_guess = last_pose * pred;

        // Run Icp
        const SE3d new_pose = ICP(
            local_map, source, init_guess, 3.0 * sigma, sigma / 3.0,
            config.icp_max_iteration, config.estimation_threshold);

        const auto model_dev = init_guess.inverse() * new_pose;
        adaptive_threshold.update_model_deviation(model_dev);

        local_map.update(down_sampled, new_pose);
        poses.emplace_back(new_pose);

        // deskewed, keypoints, pose
        return {down_sampled, source, new_pose};
    }

    utils::Vec3dVector KissICP::iqr_processing(const utils::Vec3dVector &frame)
    {
        // calculate distance for each point
        const size_t frame_size = frame.size();
        std::vector<double> distances(frame_size);
        tbb::parallel_for(
            std::size_t(0), frame_size,
            [&](std::size_t idx)
            {
                const auto &x = frame[idx].x();
                const auto &y = frame[idx].y();
                const auto &z = frame[idx].z();
                distances[idx] = x * x + y * y + z * z;
            });

        const auto &iqr_val = outlier::IQR(distances);
        double low_bound = iqr_val[0] - IQR_TUCHEY * iqr_val[2];
        double high_bound = iqr_val[1] + IQR_TUCHEY * iqr_val[2];

        tbb::concurrent_vector<utils::Vec3d> inliers;
        utils::Vec3dVector out_vec;
        out_vec.reserve(frame_size);
        inliers.reserve(frame_size);

        tbb::parallel_for(
            std::size_t(0), frame_size,
            [&](std::size_t idx)
            {
                const auto &distance = distances[idx];
                if (distance >= low_bound && distance <= high_bound)
                    inliers.emplace_back(frame[idx]);
            });

        std::move(inliers.begin(), inliers.end(), std::back_inserter(out_vec));

        return out_vec;
    }

    utils::Vec3_Vec3Tuple KissICP::voxelize(const utils::Vec3dVector &frame, const double vox_size)
    {
        // convert point clouds to eigen
        const auto downsample = voxel_downsample(frame, vox_size * 0.5);
        const auto source = voxel_downsample(downsample, vox_size * 1.5);

        // want to remove outliers in source
        const auto source_ref = iqr_processing(source);
        return {source_ref, downsample};
    }

    // used for robust kernel during ICP
    double KissICP::get_adaptive_threshold()
    {
        if (!has_moved())
            return config.initial_threshold;

        return adaptive_threshold.compute_threshold();
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
        return motion > 5.0 * config.min_motion_th;
    }

    utils::Vec3Tuple KissICP::current_vel()
    {
        if (!has_moved())
            return {utils::Vec3d::Zero(), utils::Vec3d::Zero()};

        const std::size_t N = poses.size();
        return utils::get_motion(poses[N - 2], poses[N - 1], scan_duration);
    }

}