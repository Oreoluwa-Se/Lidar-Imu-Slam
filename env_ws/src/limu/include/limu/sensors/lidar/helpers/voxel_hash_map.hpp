#ifndef VOXEL_HASH_MAP_HPP
#define VOXEL_HASH_MAP_HPP

#include <tsl/robin_map.h>
#include <boost/thread/shared_mutex.hpp>
#include "voxel_block.hpp"
#include "common.hpp"
#include <utility>

namespace lidar
{
    using Vec3dPointer = std::shared_ptr<utils::VecEigenPtrVec3d>;
    using SE3d = Sophus::SE3d;
    class VoxelHashMap
    {
    public:
        VoxelHashMap(double vox_size, double max_distance, int max_points_per_voxel, int vox_side_length = 3)
            : vox_size(vox_size), max_distance(max_distance), max_points_per_voxel(max_points_per_voxel),
              vox_cube(vox_side_length * vox_side_length * vox_side_length){};

        // insert points into map
        void insert_points(const utils::Vec3dVector &points);

        utils::Vec3d get_closest_neighbour(const utils::Vec3d &points);

        utils::Vec3_Vec3Tuple get_correspondences(const utils::Vec3dVector &points, double max_correspondance);

        // update map points
        void update(const utils::Vec3dVector &points, const utils::Vec3d &origin);
        void update(const utils::Vec3dVector &points, const SE3d &pose);

        void remove_points_from_far(const utils::Vec3d &origin);
        utils::Vec3dVector pointcloud() const;

        void clear();
        bool empty() const;

    public:
        // attributes
        tsl::robin_map<utils::Voxel, VoxelBlock, utils::VoxelHash> map;

    private:
        mutable boost::shared_mutex map_mutex;
        int max_points_per_voxel;
        double max_distance;
        double vox_size;
        int vox_cube;
    };
}
#endif