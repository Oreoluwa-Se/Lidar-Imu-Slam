#ifndef REGISTRATION_HPP
#define REGISTRATION_HPP

#include "common.hpp"
#include "voxel_hash_map.hpp"

namespace lidar
{
    using SE3d = Sophus::SE3d;
    SE3d align_clouds(const utils::Vec3_Vec3Tuple &points, double th);
    SE3d align_clouds_IQR(const utils::Vec3_Vec3Tuple &points, double th);

    // Run ICP calculation
    SE3d ICP(VoxelHashMap &local_map, const utils::Vec3dVector &points,
             const SE3d &init_guess, const double &max_corresp_dist, const double &kernel,
             const int &icp_max_iteration, const double &est_threshold);

}
#endif