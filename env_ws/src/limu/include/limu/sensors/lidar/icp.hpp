/*
 @article{vizzo2023ral,
   author    = {Vizzo, Ignacio and Guadagnino, Tiziano and Mersch, Benedikt and Wiesmann, Louis and Behley, Jens and Stachniss, Cyrill},
   title     = {{KISS-ICP: In Defense of Point-to-Point ICP -- Simple, Accurate, and Robust Registration If Done the Right Way}},
   journal   = {IEEE Robotics and Automation Letters (RA-L)},
   pages     = {1-8},
   doi       = {10.1109/LRA.2023.3236571},
   volume    = {8},
   number    = {2},
   year      = {2023},
   codeurl   = {https://github.com/PRBonn/kiss-icp},

   some modifications made for compatibility with current project.
    - Voxelhashmap changed somewhat
    - No Deskewing and Icp is done in the Ekf file.
 }
 */
#ifndef KISS_ICP_HPP
#define KISS_ICP_HPP

#include "sensor_msgs/PointCloud2.h"
#include "helpers/voxel_hash_map.hpp"
#include "helpers/threshold.hpp"
#include "common.hpp"

namespace lidar
{
    using SE3d = Sophus::SE3d;
    using ReturnTuple = std::tuple<utils::Vec3dVector, utils::Vec3dVector, SE3d>;
    using MapPoint = utils::Point;
    using MapPointPtr = MapPoint::Ptr;
    using PointsVector = std::vector<MapPoint>;
    using PointsPtrVector = std::vector<MapPointPtr>;
    using VoxelDownsampleTuple = std::tuple<PointsPtrVector, PointsPtrVector>;

    class KissICP
    {
    public:
        typedef std::shared_ptr<KissICP> Ptr;

        explicit KissICP(const std::string &loc)
        {
            YAML::Node node = YAML::LoadFile(loc);
            voxel_size = node["icp_params"]["voxel_size"].as<double>();
            min_motion_th = node["icp_params"]["min_motion_th"].as<double>();
            initial_threshold = node["icp_params"]["initial_threshold"].as<double>();

            double max_map_range = node["icp_params"]["maximum_map_distance"].as<double>();
            int max_points_per_voxel = node["icp_params"]["max_points_per_voxel"].as<int>();

            local_map = std::make_shared<VoxelHashMap>(voxel_size, max_map_range, max_points_per_voxel);
            adaptive_threshold = std::make_shared<AdaptiveThreshold>(initial_threshold, min_motion_th, max_map_range);
        }

        void update_map(PointsPtrVector &map_points, const SE3d &new_pose, bool transform_point = false);
        VoxelDownsampleTuple voxelize(const PointsPtrVector &frame);

        double ICP_setup();

        SE3d get_prediction_model() const;
        double get_adaptive_threshold();
        bool has_moved();
        void update_pose(const SE3d &pose)
        {
            poses.emplace_back(pose);
        }

        void reset_poses(const std::vector<SE3d> &update)
        {
            poses.clear();
            poses.insert(poses.end(), update.begin(), update.end());
        }

        PointsVector local_map_() const { return local_map->pointcloud(); }
        std::vector<SE3d> poses_() const { return poses; }

    public:
        std::shared_ptr<VoxelHashMap> local_map = nullptr;
        SE3d init_guess;

    private:
        std::shared_ptr<std::mutex> data_mutex;
        std::shared_ptr<AdaptiveThreshold> adaptive_threshold = nullptr;
        std::vector<SE3d> poses;
        double min_motion_th, voxel_size, initial_threshold;
    };
}
#endif