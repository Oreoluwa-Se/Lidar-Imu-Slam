#ifndef VOXEL_BLOCK_HPP
#define VOXEL_BLOCK_HPP

#include <boost/thread/shared_mutex.hpp>
#include "common.hpp"
#include <utility>

namespace lidar
{
    using Vec3dPointer = std::shared_ptr<utils::VecEigenPtrVec3d>;
    using PointPointer = std::shared_ptr<utils::Vec3d>;
    // work on VoxelBlock struct
    struct VoxelBlock
    {
    public:
        typedef std::shared_ptr<VoxelBlock> Ptr;

        // initialization
        VoxelBlock(){};

        explicit VoxelBlock(const size_t n);

        // copy constructor
        VoxelBlock(const VoxelBlock &other);

        // move constructor
        VoxelBlock(VoxelBlock &&other);

        // copy assignment
        VoxelBlock &operator=(const VoxelBlock &other);

        // move assignment
        VoxelBlock &operator=(VoxelBlock &&other);

        ~VoxelBlock() = default;

        void add_point(const PointPointer &point);

        PointPointer get_closest_point(const utils::Vec3d &point) const;

        Vec3dPointer get_points() const;

        PointPointer get_first_point() const;

        void remove_points(const utils::Vec3d &point, double distance);

        inline bool empty()
        {
            boost::shared_lock<boost::shared_mutex> lock(mutex);
            return points->empty();
        }
        // attributes
        mutable boost::shared_mutex mutex;
        Vec3dPointer points;
        size_t max_points;
    };

}
#endif