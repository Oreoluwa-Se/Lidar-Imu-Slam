#include "limu/sensors/lidar/helpers/voxel_hash_map.hpp"
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_reduce.h>
#include <execution>

namespace lidar
{

    void VoxelHashMap::insert(const utils::Vec3d &point)
    {
        // compute index for point
        utils::Voxel vox_idx = get_vox_index(point);
        const auto point_ptr = std::make_shared<utils::Vec3d>(point);
        // locked for writing
        boost::unique_lock<boost::shared_mutex> lock(mutex);

        auto it = map.find(vox_idx);
        if (it != map.end())
        {
            auto &block = const_cast<VoxelBlock &>(it->second);
            block.add_point(point_ptr);
        }
        else
        {
            auto vox_block = VoxelBlock();
            vox_block.num_points = max_points_per_voxel;
            vox_block.add_point(point_ptr);
            map.insert(std::make_pair(vox_idx, vox_block));
        }
    }

    void VoxelHashMap::insert_points(const utils::Vec3dVector &points)
    {
        // use parallel execution policy. -> maybe linearize later on
        tbb::parallel_for_each(
            points.begin(), points.end(),
            [&](const auto &point)
            {
                insert(point);
            });
    }

    VoxelHashMap::VoxelBlock VoxelHashMap::get(const utils::Voxel &vox_idx) const
    {
        // shared lock for reading.
        boost::shared_lock<boost::shared_mutex> lock(mutex);
        auto it = map.find(vox_idx);
        if (it != map.end())
            return it->second;

        return VoxelBlock();
    }

    VoxelHashMap::VoxelBlock VoxelHashMap::get(const utils::Vec3d &point) const
    {
        // shared lock for reading.
        const auto vox_idx = get_vox_index(point);
        boost::shared_lock<boost::shared_mutex> lock(mutex);
        auto it = map.find(vox_idx);
        if (it != map.end())
            return it->second;

        return VoxelBlock();
    }

    std::shared_ptr<const utils::Vec3d> VoxelHashMap::get_closest_neighboor(const utils::Vec3d &point)
    {
        std::shared_ptr<const utils::Vec3d> closest_neighbor = nullptr;
        double closest_distance_sq = std::numeric_limits<double>::max();

        auto search_point = [&](const int i, const int j, const int k)
        {
            utils::Voxel vox(i, j, k);
            const auto vox_block = get(vox);

            // check for if its been found
            for (const auto &neigbor_ptr : vox_block.points)
            {
                const double distance = (*neigbor_ptr - point).squaredNorm();
                if (distance < closest_distance_sq)
                {
                    closest_neighbor = neigbor_ptr;
                    closest_distance_sq = distance;
                }
            }
        };

        // calculate interpolation range
        const int kx = point[0] / vox_size;
        const int ky = point[1] / vox_size;
        const int kz = point[2] / vox_size;

        for (int i = kx - 1; i <= kx + 1; ++i)
        {
            for (int j = ky - 1; j <= ky + 1; ++j)
            {
                for (int k = kz - 1; k <= kz + 1; ++k)
                {
                    search_point(i, j, k);
                }
            }
        }

        return closest_neighbor;
    }

    utils::Vec3_Vec3Tuple VoxelHashMap::get_correspondences(const utils::Vec3dVector &points, double max_correspondance)
    {
        using points_iterator = utils::Vec3dVector::const_iterator;

        const auto paired = tbb::parallel_reduce(
            // Range
            tbb::blocked_range<points_iterator>(points.begin(), points.end()),
            // Initialization
            std::make_tuple(utils::Vec3dVector{}, utils::Vec3dVector{}),
            // First lambda
            [&](const tbb::blocked_range<points_iterator> &range, utils::Vec3_Vec3Tuple local_result)
            {
                auto &src = std::get<0>(local_result);
                auto &targ = std::get<1>(local_result);

                for (auto point_ptr = range.begin(); point_ptr != range.end(); ++point_ptr)
                {
                    std::shared_ptr<const utils::Vec3d> neighbor;
                    {
                        boost::shared_lock<boost::shared_mutex> lock(mutex);
                        neighbor = get_closest_neighboor(*point_ptr);
                    }
                    if (neighbor != nullptr)
                    {
                        auto distance = (*neighbor - *point_ptr).norm();
                        if (distance < max_correspondance)
                        {
                            src.emplace_back(*point_ptr);
                            targ.emplace_back(*neighbor);
                        }
                    }
                }

                return local_result;
            },
            [](utils::Vec3_Vec3Tuple a, const utils::Vec3_Vec3Tuple &b)
            {
                auto &a_source = std::get<0>(a);
                auto &a_target = std::get<1>(a);
                auto &b_source = std::get<0>(b);
                auto &b_target = std::get<1>(b);

                a_source.insert(
                    a_source.end(),
                    std::make_move_iterator(b_source.begin()),
                    std::make_move_iterator(b_source.end()));
                a_target.insert(
                    a_target.end(),
                    std::make_move_iterator(b_target.begin()),
                    std::make_move_iterator(b_target.end()));

                return a;
            });

        return paired;
    }

    void VoxelHashMap::remove_points_from_far(const utils::Vec3d &origin)
    {
        const auto max_dist_sq = max_distance * max_distance;
        // removing voxels
        for (auto it = map.begin(); it != map.end();)
        {
            auto &block = it->second;
            std::lock_guard<std::mutex> lock_mutex(block.mutex);
            const auto &pt = block.points.front();

            if ((*pt - origin).squaredNorm() > max_dist_sq)
            {
                // decrease reference count
                boost::unique_lock<boost::shared_mutex> lock(mutex);
                it = map.erase(it);
            }
            else
                ++it;
        }
    }

    void VoxelHashMap::update(const utils::Vec3dVector &points, const utils::Vec3d &origin)
    {
        insert_points(points);
        remove_points_from_far(origin);
    }

    void VoxelHashMap::update(const utils::Vec3dVector &points, const SE3d &pose)
    {
        auto pts = points;
        utils::transform_points(pose, pts);
        const utils::Vec3d &origin = pose.translation();
        update(pts, origin);
    }

    utils::Vec3dVector VoxelHashMap::pointcloud() const
    {
        utils::Vec3dVector points;

        boost::shared_lock<boost::shared_mutex> lock(mutex);
        points.reserve(max_points_per_voxel * map.size());
        for (const auto &vox_blck_pair : map)
        {
            const auto &v_blck = vox_blck_pair.second;

            std::for_each(
                v_blck.points.cbegin(), v_blck.points.cend(),

                [&](const auto &point)
                { points.emplace_back(*point); });
        }

        return points;
    }
}