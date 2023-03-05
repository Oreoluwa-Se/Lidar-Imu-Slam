#ifndef VOXEL_HASH_MAP_HPP
#define VOXEL_HASH_MAP_HPP

#include <tsl/robin_map.h>
#include <boost/thread/shared_mutex.hpp>
#include "common.hpp"

namespace lidar
{
    using SE3d = Sophus::SE3d;

    class VoxelHashMap
    {
    public:
        typedef std::shared_ptr<VoxelHashMap> Ptr;

        // Voxel points storage.
        struct VoxelBlock
        {
            typedef std::shared_ptr<VoxelBlock> Ptr;

            VoxelBlock() = default;
            // deep copy
            VoxelBlock(const VoxelBlock &other)
                : num_points(other.num_points)
            {
                std::lock_guard<std::mutex> lock(other.mutex);
                points.resize(other.points.size());
                std::transform(
                    other.points.begin(), other.points.end(), points.begin(),
                    [](const auto &point)
                    {
                        return std::make_shared<utils::Vec3d>(*point);
                    });
            }

            VoxelBlock(VoxelBlock &&other)
                : num_points(other.num_points), points(std::move(other.points))
            {
                other.num_points = 0;
            }

            VoxelBlock &operator=(const VoxelBlock &other)
            {
                if (this != &other)
                {
                    std::lock(mutex, other.mutex); // lock both mutexes to avoid deadlock
                    std::lock_guard<std::mutex> lockThis(mutex, std::adopt_lock);
                    std::lock_guard<std::mutex> lockOther(other.mutex, std::adopt_lock);
                    num_points = other.num_points;
                    points = other.points;
                }

                return *this;
            }

            VoxelBlock &operator=(VoxelBlock &&other)
            {
                if (this != &other)
                {
                    std::lock_guard<std::mutex> lock(other.mutex);
                    num_points = other.num_points;
                    points = std::move(other.points);
                    other.num_points = 0;
                }

                return *this;
            }

            ~VoxelBlock()
            {
                for (auto ptr : points)
                    ptr.reset();
            }

            int num_points;
            mutable std::mutex mutex; // mutable allows mutex to be locked and modified even in a const member
            utils::VecEigenPtrVec3d points;

            inline void add_point(const std::shared_ptr<utils::Vec3d> &point)
            {
                if (points.size() < static_cast<std::size_t>(num_points))
                    points.emplace_back(point);
            }
        };

        // voxel block holder.
        VoxelHashMap(double vox_size, double max_distance, int max_points_per_voxel, int vox_side_length = 3)
            : vox_size(vox_size), max_distance(max_distance), max_points_per_voxel(max_points_per_voxel),
              vox_cube(vox_side_length * vox_side_length * vox_side_length){};

        // modifying copy constructor:
        VoxelHashMap(const VoxelHashMap &other)
        {
            std::lock_guard<boost::shared_mutex> lock(other.mutex);
            max_points_per_voxel = other.max_points_per_voxel;
            max_distance = other.max_distance;
            vox_size = other.vox_size;
            vox_cube = other.vox_cube;
            for (const auto &block : other.map)
            {
                // map.insert(std::make_tuple(block.first, block.second));
                map.insert(block);
            }
        }

        // insert points into the map.
        void insert_points(const utils::Vec3dVector &points);

        // inserting one point into the map.
        void insert(const utils::Vec3d &point);

        // get access to vox block.
        VoxelBlock get(const utils::Voxel &vox_idx) const;
        VoxelBlock get(const utils::Vec3d &point) const;

        // get closest neighbor.
        std::shared_ptr<const utils::Vec3d> get_closest_neighboor(const utils::Vec3d &point);

        // get correspondences.
        utils::Vec3_Vec3Tuple get_correspondences(const utils::Vec3dVector &points, double max_correspondance);

        // update map points
        void update(const utils::Vec3dVector &points, const utils::Vec3d &origin);
        void update(const utils::Vec3dVector &points, const SE3d &pose);

        utils::Vec3dVector pointcloud() const;

        void clear()
        {
            boost::unique_lock<boost::shared_mutex> lock(mutex);
            map.clear();
        }
        bool empty() const
        {
            boost::unique_lock<boost::shared_mutex> lock(mutex);
            return map.empty();
        }

    public:
        // attributes
        tsl::robin_map<utils::Voxel, VoxelBlock, utils::VoxelHash> map;

    private:
        // remove objects far from a given origin
        void remove_points_from_far(const utils::Vec3d &origin);
        // compute voxel index for a given point
        inline utils::Voxel get_vox_index(const utils::Vec3d &point) const
        {
            return utils::Voxel(std::floor(point.x() / vox_size),
                                std::floor(point.y() / vox_size),
                                std::floor(point.z() / vox_size));
        }

        int max_points_per_voxel;
        double max_distance;
        double vox_size;
        int vox_cube;

        // mutex for synchronization.. shared mutex used to improve reading access.
        mutable boost::shared_mutex mutex;
        std::mutex data_mutex;
    };

}
#endif