#ifndef VOXEL_BLOCK_HPP
#define VOXEL_BLOCK_HPP

#include "common.hpp"
#include <utility>
#include <queue>
#include <algorithm>

namespace lidar
{
    using MapPoint = utils::Point;
    using MapPointPtr = MapPoint::Ptr;
    using PointsVector = std::vector<MapPoint>;
    using PointsPtrVector = std::vector<MapPointPtr>;
    using WPointsPtrVector = std::vector<std::weak_ptr<MapPoint>>;

    // work on VoxelBlock struct
    struct VoxelBlock
    {
    public:
        typedef std::shared_ptr<VoxelBlock> Ptr;

        // initialization
        VoxelBlock(){};

        explicit VoxelBlock(const size_t n);
        explicit VoxelBlock(bool use_weak);
        VoxelBlock(const size_t n, bool use_weak);

        // copy constructor
        VoxelBlock(const VoxelBlock &other);

        // move constructor
        VoxelBlock(VoxelBlock &&other);

        // copy assignment
        VoxelBlock &operator=(const VoxelBlock &other);

        // move assignment
        VoxelBlock &operator=(VoxelBlock &&other);

        ~VoxelBlock() = default;

        void add_point(const MapPointPtr &point);

        PointsPtrVector get_closest_points(const MapPoint &point, double max_corresp, int num_points) const;

        MapPoint get_closest_point(const MapPoint &point) const;

        PointsPtrVector get_points() const;

        MapPoint get_first_point() const;

        void remove_points(const utils::Vec3d &point, double distance);

        inline bool empty()
        {
            boost::shared_lock<boost::shared_mutex> lock(mutex);
            return points.empty();
        }
        // attributes
        mutable boost::shared_mutex mutex;
        PointsPtrVector points;    // shared pointers
        WPointsPtrVector w_points; // weak pointers
        bool use_weak;
        size_t max_points;
    };

}
#endif