#include "limu/sensors/lidar/helpers/voxel_hash_map.hpp"
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <functional>
#include <thread>
#include <queue>
#include <sstream>
namespace lidar
{

    void VoxelHashMap::insert_points(const PointsPtrVector &points)
    {
        // concurrent_vector to hold points for each voxel
        tbb::concurrent_vector<std::pair<utils::Voxel, MapPointPtr>> vox_points;

        // chunck properties
        int num_threads = std::thread::hardware_concurrency();
        int chunk_size = std::max(static_cast<int>(static_cast<double>(points.size() / num_threads)), 1);
        std::vector<PointsPtrVector::const_iterator> chunk_starts;
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
        for (const auto &vox_point : vox_points)
        {
            const auto &vox = std::get<0>(vox_point);
            const auto &point = std::get<1>(vox_point);

            auto it = map.find(vox);
            if (it == map.end())
                map.emplace(std::make_pair(vox, VoxelBlock(max_points_per_voxel)));

            // non const access
            auto &voxel_block = map[vox];
            voxel_block.add_point(point);
        }
    }

    PointsVector VoxelHashMap::get_closest_neighbour(const MapPoint &point, double max_corresp, int num_corresp)
    {
        // Validate input parameters
        if (max_corresp <= 0.0)
            throw std::invalid_argument("max_corresp must be greater than 0");

        if (num_corresp <= 0)
            throw std::invalid_argument("num_corresp must be greater than 0");

        boost::unique_lock<boost::shared_mutex> lock(map_mutex);
        // convert the point to voxel
        const auto point_vox = utils::get_vox_index(point.point, vox_size);

        // restrict voxel range we search for
        auto kx = point_vox.x();
        auto ky = point_vox.y();
        auto kz = point_vox.z();
        double corresp_sq = max_corresp * max_corresp;

        using QueueType = std::pair<double, MapPointPtr>;
        std::priority_queue<QueueType, std::vector<QueueType>, std::greater<QueueType>> closest_points_heap;
        for (int i = kx - 1; i <= kx + 1; ++i)
        {
            for (int j = ky - 1; j <= ky + 1; ++j)
            {
                for (int k = kz - 1; k <= kz + 1; ++k)
                {
                    const auto voxel = utils::Voxel(i, j, k);
                    const auto it = map.find(voxel);
                    if (it == map.end())
                        continue;

                    const auto dist = (point_vox - voxel).squaredNorm();
                    if (dist > corresp_sq)
                        continue;

                    const auto closest_points = it->second.get_closest_points(point, max_corresp, num_corresp);
                    for (const auto &closest : closest_points)
                    {
                        const auto p_dist = (point - *closest).squaredNorm();
                        closest_points_heap.emplace(p_dist, closest);
                    }
                }
            }
        }

        // change this snippet to fit the new narrative for  max number of corresponding points
        PointsVector closest_neighbors;
        closest_neighbors.reserve(num_corresp);
        int idx = 0;
        while (!closest_points_heap.empty() && idx != num_corresp)
        {
            closest_neighbors.emplace_back(*(closest_points_heap.top().second));
            closest_points_heap.pop();
            ++idx;
        }

        return closest_neighbors;
    }

    CorrespondenceTuple VoxelHashMap::get_correspondences(
        const PointsPtrVector &points, double max_corresp, int num_corresp)
    {
        tbb::concurrent_vector<std::vector<MapPoint>> matched(points.size());

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, points.size()),
            [&](const tbb::blocked_range<size_t> &range)
            {
                for (size_t i = range.begin(); i != range.end(); ++i)
                {
                    const auto &point = points[i];
                    const auto closest_neighbors = get_closest_neighbour(
                        *point, max_corresp, num_corresp);

                    auto &corresp_for_point = matched.at(i);
                    corresp_for_point.reserve(num_corresp);

                    corresp_for_point.insert(corresp_for_point.end(), closest_neighbors.begin(), closest_neighbors.end());
                }
            });

        const int r_size = static_cast<int>(points.size()) * num_corresp;
        PointsVector target;
        std::vector<int> idx_vec;
        target.reserve(r_size);
        idx_vec.reserve(r_size);

        int idx = 0;
        for (const auto &corresp_for_points : matched)
        {
            if (!corresp_for_points.empty())
            {
                for (const auto &found : corresp_for_points)
                {
                    target.emplace_back(found);
                    idx_vec.emplace_back(idx);
                }
            }

            ++idx;
        }

        return {idx_vec, target};
    }

    void VoxelHashMap::update(const PointsPtrVector &points, const utils::Vec3d &origin)
    {
        insert_points(points);
        remove_points_from_far(origin);
    }

    void VoxelHashMap::update(const PointsPtrVector &points, const SE3d &pose, bool transform_points)
    {
        auto converted_points = points;
        const utils::Vec3d &origin = pose.translation();
        if (transform_points)
            utils::transform_points(pose, converted_points);

        update(converted_points, origin);
    }

    void VoxelHashMap::remove_points_from_far(const utils::Vec3d &origin)
    {
        const auto max_dist_sq = max_distance * max_distance;
        const auto origin_vox = utils::get_vox_index(origin, vox_size);
        using map_type = tsl::robin_map<utils::Voxel, VoxelBlock, utils::VoxelHash>;

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

    PointsVector VoxelHashMap::pointcloud() const
    {
        boost::shared_lock<boost::shared_mutex> lock(map_mutex);
        tbb::concurrent_vector<MapPoint> c_points;

        tbb::parallel_for_each(
            map.begin(), map.end(),
            [&](const auto &vox_blck_pair)
            {
                // std::vector<MapPointPtr>
                const auto block_vec_ptr = vox_blck_pair.second.get_points();
                tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, block_vec_ptr.size()),
                    [&](const tbb::blocked_range<size_t> &r)
                    {
                        for (size_t i = r.begin(); i < r.end(); ++i)
                        {
                            const auto &point = block_vec_ptr[i];
                            c_points.emplace_back(*point);
                        }
                    });
            });

        PointsVector points;
        points.reserve(c_points.size());

        std::transform(
            c_points.begin(), c_points.end(),
            std::back_inserter(points),
            [](auto &&point)
            { return std::move(point); });

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