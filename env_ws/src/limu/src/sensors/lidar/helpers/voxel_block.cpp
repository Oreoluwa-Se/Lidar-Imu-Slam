#include "limu/sensors/lidar/helpers/voxel_block.hpp"

namespace lidar
{
    VoxelBlock::VoxelBlock(const size_t n)
        : max_points(static_cast<size_t>(n)),
          points(std::make_shared<utils::VecEigenPtrVec3d>())
    {
        points->reserve(n);
    }

    // copy constructor
    VoxelBlock::VoxelBlock(const VoxelBlock &other)
        : max_points(other.max_points),
          points(std::make_shared<utils::VecEigenPtrVec3d>())
    {
        boost::shared_lock<boost::shared_mutex> lock(other.mutex);
        std::transform(
            other.points->begin(), other.points->end(),
            std::back_inserter(*points),
            [](const auto &point)
            { return std::make_shared<utils::Vec3d>(*point); });
    }

    // move constructor
    VoxelBlock::VoxelBlock(VoxelBlock &&other)
        : max_points(other.max_points),
          points(std::move(other.points)) {}

    // copy assignment
    VoxelBlock &VoxelBlock::operator=(const VoxelBlock &other)
    {
        if (this != &other)
        {
            boost::shared_lock<boost::shared_mutex> lock1(mutex, boost::defer_lock);
            boost::shared_lock<boost::shared_mutex> lock2(other.mutex, boost::defer_lock);
            std::lock(lock1, lock2);

            max_points = other.max_points;
            points.reset(new utils::VecEigenPtrVec3d);

            std::transform(
                other.points->begin(), other.points->end(),
                std::back_inserter(*points),
                [](const auto &point)
                { return std::make_shared<utils::Vec3d>(*point); });
        }

        return *this;
    }

    // move assignment
    VoxelBlock &VoxelBlock::operator=(VoxelBlock &&other)
    {
        if (this != &other)
        {
            boost::shared_lock<boost::shared_mutex> lock1(mutex, boost::defer_lock);
            boost::shared_lock<boost::shared_mutex> lock2(other.mutex, boost::defer_lock);
            std::lock(lock1, lock2);

            max_points = other.max_points;
            points = std::move(other.points);
        }

        return *this;
    }

    void VoxelBlock::add_point(const PointPointer &point)
    {
        boost::unique_lock<boost::shared_mutex> lock(mutex);
        if (points->size() < static_cast<std::size_t>(max_points))
            points->emplace_back(point);
    }

    Vec3dPointer VoxelBlock::get_points() const
    {
        boost::shared_lock<boost::shared_mutex> lock(mutex);
        return points;
    }

    PointPointer VoxelBlock::get_first_point() const
    {
        boost::shared_lock<boost::shared_mutex> lock(mutex);
        return points->front();
    }

    PointPointer VoxelBlock::get_closest_point(const utils::Vec3d &point) const
    {
        PointPointer closest_point = nullptr;
        double min_dist = std::numeric_limits<double>::max();
        boost::shared_lock<boost::shared_mutex> lock(mutex);

        // Iterate over the points and compute the distance to the given point
        for (const auto &p : *points)
        {
            double dist = (point - *p).squaredNorm();
            if (dist < min_dist)
            {
                closest_point = p;
                min_dist = dist;
            }
        }

        return closest_point;
    }

    void VoxelBlock::remove_points(const utils::Vec3d &origin, double distance)
    {
        auto dist_sq = distance * distance;
        boost::unique_lock<boost::shared_mutex> lock(mutex);
        for (auto it = points->begin(); it != points->end();)
        {
            if ((**it - origin).squaredNorm() > dist_sq)
                it = points->erase(it);
            else
                ++it;
        }
    }
}