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
 }
 */
#ifndef KISS_ICP_HPP
#define KISS_ICP_HPP

#include "sensor_msgs/PointCloud2.h"
#include "helpers/registration.hpp"
#include "helpers/voxel_hash_map.hpp"
#include "helpers/threshold.hpp"
#include "helpers/deskew.hpp"
#include "common.hpp"
#include "frame.hpp"

namespace lidar
{
    using SE3d = Sophus::SE3d;
    using ReturnTuple = std::tuple<utils::Vec3dVector, utils::Vec3dVector, SE3d>;
    class KissICP
    {
    public:
        typedef std::shared_ptr<KissICP> Ptr;
        explicit KissICP(const frame::Lidar::ProcessingInfo::Ptr &config)
            : config(*config),
              local_map(config->voxel_size, config->max_range, config->max_points_per_voxel, config->vox_side_length),
              adaptive_threshold(config->initial_threshold, config->min_motion_th, config->max_range),
              compensator(), scan_duration(1 / (double(config->frame_split_num) * config->frame_rate)){};

        // register frame
        ReturnTuple register_frame(const utils::PointCloudXYZI &pointcloud, const std::vector<double> &timestamps);
        ReturnTuple register_frame(const utils::Vec3dVector &frame);

        // downsample pointcloud KISS-ICP Downsampling scheme.
        utils::Vec3_Vec3Tuple voxelize(const utils::Vec3dVector &frame, const double vox_size);

        SE3d get_prediction_model() const;
        double get_adaptive_threshold();
        utils::Vec3Tuple current_vel();
        bool has_moved();

        utils::Vec3dVector local_map_() const { return local_map.pointcloud(); }
        std::vector<SE3d> poses_() const { return poses; }

    private:
        std::shared_ptr<std::mutex> data_mutex;
        frame::Lidar::ProcessingInfo config;
        AdaptiveThreshold adaptive_threshold;
        MotionCompensator compensator;
        std::vector<SE3d> poses;
        VoxelHashMap local_map;
        double scan_duration;
    };
}
#endif