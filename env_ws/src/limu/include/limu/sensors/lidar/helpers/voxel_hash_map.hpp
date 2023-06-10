#ifndef VOXEL_HASH_MAP_HPP
#define VOXEL_HASH_MAP_HPP

#include <tsl/robin_map.h>
#include <boost/thread/shared_mutex.hpp>
#include "voxel_block.hpp"
#include "common.hpp"

namespace lidar
{
    using SE3d = Sophus::SE3d;
    using MapPoint = utils::Point;
    using MapPointPtr = MapPoint::Ptr;
    using PointsVector = std::vector<MapPoint>;
    using PointsPtrVector = std::vector<MapPointPtr>;
    using CorrespondenceTuple = std::tuple<std::vector<int>, PointsVector>;

    class VoxelHashMap
    {
    public:
        VoxelHashMap(double vox_size, double max_distance, int max_points_per_voxel)
            : vox_size(vox_size), max_distance(max_distance),
              max_points_per_voxel(static_cast<size_t>(max_points_per_voxel)),
              num_corresp(1){};

        // insert points into map
        void insert_points(const PointsPtrVector &points);

        PointsVector get_closest_neighbour(const MapPoint &points, double max_corresp, int num_corresp = 1);

        CorrespondenceTuple get_correspondences(const PointsPtrVector &points, double max_corresp, int num_corresp = 1);

        // update map points
        void update(const PointsPtrVector &points, const SE3d &pose, bool transform_points = false);
        void update(const PointsPtrVector &points, const utils::Vec3d &origin);

        void remove_points_from_far(const utils::Vec3d &origin);
        PointsVector pointcloud() const;

        void clear();
        bool empty() const;

    public:
        // attributes
        tsl::robin_map<utils::Voxel, VoxelBlock, utils::VoxelHash> map;
        // introduce a vector for tracking point cloud.. should use a weak pointer
    private:
        mutable boost::shared_mutex map_mutex;
        size_t max_points_per_voxel;
        double max_distance;
        double vox_size;
        int num_corresp;
    };
}
#endif