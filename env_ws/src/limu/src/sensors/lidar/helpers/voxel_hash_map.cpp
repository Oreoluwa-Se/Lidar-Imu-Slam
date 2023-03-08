#include "limu/sensors/lidar/helpers/voxel_hash_map.hpp"
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <thread>
#include <queue>

namespace lidar
{
    void VoxelHashMap::insert_points(const utils::Vec3dVector &points)
    {
        // concurrent_vector to hold points for each voxel
        tbb::concurrent_vector<std::pair<utils::Voxel, utils::Vec3d>> vox_points;

        // chunck properties
        int num_threads = std::thread::hardware_concurrency();
        int chunk_size = std::max(static_cast<int>(static_cast<double>(points.size() / num_threads)), 1);
        std::vector<utils::Vec3dVector::const_iterator> chunk_starts;
        {
            for (auto it = points.begin(); it < points.end(); it += chunk_size)
            {
                chunk_starts.push_back(it);
            }
            chunk_starts.push_back(points.end());
        }
        // create voxel and store points in parallel
        tbb::parallel_for(
            tbb::blocked_range<size_t>{0, chunk_starts.size() - 1},
            [&](const tbb::blocked_range<size_t> &range)
            {
                for (size_t idx = range.begin(); idx < range.end(); ++idx)
                {
                    auto start = chunk_starts[idx];
                    auto end = chunk_starts[idx + 1];

                    for (auto it = start; it != end; ++it)
                    {
                        const auto &point = *it;
                        const auto voxel = utils::get_vox_index(point, vox_size);
                        vox_points.emplace_back(std::make_pair(voxel, point));
                    }
                }
            });

        // add to map
        boost::unique_lock<boost::shared_mutex> lock(map_mutex);
        for (const auto &vox_point : vox_points)
        {
            const auto &vox = std::get<0>(vox_point);
            const auto &point = std::get<1>(vox_point);

            auto it = map.find(vox);
            if (it == map.end())
                map.emplace(std::make_pair(vox, VoxelBlock(max_points_per_voxel)));

            // non const access
            auto &voxel_block = map[vox];
            voxel_block.add_point(std::make_shared<utils::Vec3d>(point));
        }
    }

    utils::Vec3d VoxelHashMap::get_closest_neighbour(const utils::Vec3d &point)
    {
        // convert the point to voxel
        const auto point_vox = utils::get_vox_index(point, vox_size);

        // check if voxel in map
        boost::shared_lock<boost::shared_mutex> lock(map_mutex);
        const auto it = map.find(point_vox);
        if (it != map.end())
            return *(it->second.get_closest_point(point));

        // restrict voxel range we search for
        auto kx = static_cast<int>(point[0] / vox_size);
        auto ky = static_cast<int>(point[1] / vox_size);
        auto kz = static_cast<int>(point[2] / vox_size);

        // populate voxel map
        std::priority_queue<std::pair<double, const VoxelBlock *>> closest_voxels;
        for (int i = kx - 1; i <= kx + 1; ++i)
        {
            for (int j = ky - 1; j <= ky + 1; ++j)
            {
                for (int k = kz - 1; k <= kz + 1; ++k)
                {
                    const auto voxel = utils::Voxel(i, j, k);
                    if (map.find(voxel) != map.end())
                    {
                        auto dist = (point_vox - voxel).squaredNorm();
                        closest_voxels.emplace(dist, &map[voxel]);
                    }
                }
            }
        }

        if (closest_voxels.empty())
            return utils::Vec3d::Zero();

        return *(closest_voxels.top().second->get_closest_point(point));
    }

    utils::Vec3_Vec3Tuple VoxelHashMap::get_correspondences(const utils::Vec3dVector &points, double max_correspondance)
    {
        utils::Vec3_Vec3Tuple result;
        utils::Vec3dVector source, target;
        tbb::concurrent_vector<utils::Vec3d> src, targ;
        src.reserve(points.size());
        targ.reserve(points.size());

        double max_corresp_sq = max_correspondance * max_correspondance;

        tbb::parallel_for_each(
            points.begin(), points.end(),
            [&](const auto &point)
            {
                const auto found = get_closest_neighbour(point);

                if ((found - point).squaredNorm() < max_corresp_sq)
                {
                    src.emplace_back(point);
                    targ.emplace_back(found);
                }
            });

        std::move(src.begin(), src.end(), std::back_inserter(source));
        std::move(targ.begin(), targ.end(), std::back_inserter(target));
        return {source, target};
    }

    void VoxelHashMap::update(const utils::Vec3dVector &points, const utils::Vec3d &origin)
    {
        insert_points(points);
        remove_points_from_far(origin);
    }

    void VoxelHashMap::update(const utils::Vec3dVector &points, const SE3d &pose)
    {
        auto converted_points = points;
        const utils::Vec3d &origin = pose.translation();
        utils::transform_points(pose, converted_points);
        update(converted_points, origin);
    }

    void VoxelHashMap::remove_points_from_far(const utils::Vec3d &origin)
    {
        const auto max_dist_sq = max_distance * max_distance;
        const auto origin_vox = utils::get_vox_index(origin, vox_size);
        using map_type = tsl::robin_map<utils::Voxel, VoxelBlock, utils::VoxelHash>;
        // boost::upgrade_lock<boost::shared_mutex> lock(map_mutex);
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, map.size()),
            [&](const tbb::blocked_range<size_t> &range)
            {
                boost::shared_lock<boost::shared_mutex> lock(map_mutex);
                for (auto it = std::next(map.begin(), range.begin()); it != std::next(map.begin(), range.end()); ++it)
                {
                    auto &vox = it->first;
                    if ((vox - origin_vox).squaredNorm() > max_dist_sq)
                    {
                        boost::unique_lock<boost::shared_mutex> u_lock(map_mutex);
                        auto &block = map[vox];
                        block.remove_points(origin, max_distance);

                        if (block.empty())
                            map.erase(vox);
                    }
                }
            });
    }

    utils::Vec3dVector VoxelHashMap::pointcloud() const
    {
        utils::Vec3dVector points;

        boost::shared_lock<boost::shared_mutex> lock(map_mutex);
        points.reserve(max_points_per_voxel * map.size());

        tbb::parallel_for_each(
            map.begin(), map.end(),
            [&](const auto &vox_blck_pair)
            {
                const auto block_ptr = vox_blck_pair.second.get_points();
                tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, block_ptr->size()),
                    [&](const tbb::blocked_range<size_t> &r)
                    {
                        for (size_t i = r.begin(); i < r.end(); ++i)
                        {
                            const auto &point = (*block_ptr)[i];
                            points.emplace_back(*point);
                        }
                    });
            });

        return points;
    }

    void VoxelHashMap::clear()
    {
        boost::unique_lock<boost::shared_mutex> lock(map_mutex);
        map.clear();
    }

    bool VoxelHashMap::empty() const
    {
        boost::shared_lock<boost::shared_mutex> lock(map_mutex);
        return map.empty();
    }
}

// tbb::parallel_for_each(
//     local_map.begin(), local_map.end(),
//     [&](const auto &it)
//     {
//         const auto& vox = it.first;
//         if ((vox - origin_vox).squaredNorm() > max_dist_sq)
//         {
//             boost::unique_lock<boost::shared_mutex> lock(map_mutex);
//             auto &block = map[vox];
//             block.remove_points(origin, max_distance);

//             if (block.empty())
//                 map.erase(vox);
//         } });