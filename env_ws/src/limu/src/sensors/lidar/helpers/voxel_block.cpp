#include "limu/sensors/lidar/helpers/voxel_block.hpp"
#include <functional>
#include <queue>

namespace lidar
{
    VoxelBlock::VoxelBlock(const size_t n)
        : max_points(static_cast<size_t>(n)),
          points(), w_points(), use_weak(false)
    {
        if (!use_weak)
            points.reserve(n);
        else
            w_points.reserve(n);
    }

    VoxelBlock::VoxelBlock(bool use_weak_)
        : max_points(0),
          points(), w_points(), use_weak(use_weak_) {}

    VoxelBlock::VoxelBlock(const size_t n, bool use_weak_)
        : max_points(static_cast<size_t>(n)),
          points(), w_points(), use_weak(use_weak_)
    {
        if (!use_weak)
            points.reserve(n);
        else
            w_points.reserve(n);
    }

    // copy constructor
    VoxelBlock::VoxelBlock(const VoxelBlock &other)
        : max_points(other.max_points),
          points(), w_points(), use_weak(other.use_weak)
    {
        boost::shared_lock<boost::shared_mutex> lock(other.mutex);
        std::transform(
            other.points.begin(), other.points.end(),
            std::back_inserter(points),
            [](const auto &point)
            { return point; });

        std::transform(
            other.w_points.begin(), other.w_points.end(),
            std::back_inserter(w_points),
            [](const auto &point)
            { return point; });
    }

    // move constructor
    VoxelBlock::VoxelBlock(VoxelBlock &&other)
        : max_points(other.max_points),
          points(std::move(other.points)),
          w_points(std::move(other.w_points)),
          use_weak(other.use_weak) {}

    // copy assignment
    VoxelBlock &VoxelBlock::operator=(const VoxelBlock &other)
    {
        if (this != &other)
        {
            boost::unique_lock<boost::shared_mutex> lock1(mutex, boost::defer_lock);
            boost::shared_lock<boost::shared_mutex> lock2(other.mutex, boost::defer_lock);
            std::lock(lock1, lock2);

            max_points = other.max_points;
            use_weak = other.use_weak;

            if (!use_weak)
            {
                points.clear();
                points.reserve(other.points.size());
                std::transform(
                    other.points.begin(), other.points.end(),
                    std::back_inserter(points),
                    [](const auto &point)
                    { return std::make_shared<MapPoint>(*point); });
            }
            else
            {
                w_points.clear();
                w_points.reserve(other.w_points.size());
                for (const auto &w_p : other.w_points)
                {
                    if (auto sharedPtr = w_p.lock())
                        w_points.emplace_back(std::move(w_p));
                }
            }
        }

        return *this;
    }

    // move assignment
    VoxelBlock &VoxelBlock::operator=(VoxelBlock &&other)
    {
        if (this != &other)
        {
            boost::unique_lock<boost::shared_mutex> lock1(mutex, boost::defer_lock);
            boost::shared_lock<boost::shared_mutex> lock2(other.mutex, boost::defer_lock);
            std::lock(lock1, lock2);

            max_points = other.max_points;
            use_weak = other.use_weak;
            points = std::move(other.points);
            w_points = std::move(other.w_points);
        }

        return *this;
    }

    void VoxelBlock::add_point(const MapPointPtr &point)
    {
        boost::unique_lock<boost::shared_mutex> lock(mutex);
        if (max_points > 0)
        {
            if (points.size() < static_cast<std::size_t>(max_points))
            {
                if (!use_weak)
                    points.emplace_back(point);
                else
                    w_points.emplace_back(point);
            }
        }
        else
        {
            if (!use_weak)
                points.emplace_back(point);
            else
                w_points.emplace_back(point);
        }
    }

    PointsPtrVector VoxelBlock::get_points() const
    {
        boost::shared_lock<boost::shared_mutex> lock(mutex);
        if (!use_weak)
            return points;
        else
        {
            PointsPtrVector s_points;
            s_points.reserve(w_points.size());
            for (const auto &point : w_points)
                if (auto ptr = point.lock())
                    s_points.emplace_back(ptr);

            return s_points;
        }
    }

    MapPoint VoxelBlock::get_first_point() const
    {
        boost::shared_lock<boost::shared_mutex> lock(mutex);
        if (!use_weak)
            return *(points.front());
        else
        {
            const auto &point = w_points.front();
            if (auto ptr = point.lock())
                return *ptr;
            else
            {
                MapPoint invalid;
                invalid.point_valid = false;

                return invalid;
            }
        }
    }

    MapPoint VoxelBlock::get_closest_point(const MapPoint &point) const
    {
        if (points.empty())
        {
            MapPoint default_value;
            default_value.is_default = true;
            return default_value; // or throw an exception
        }

        MapPointPtr closest_point = nullptr;
        double min_dist = std::numeric_limits<double>::max();
        boost::shared_lock<boost::shared_mutex> lock(mutex);

        // Iterate over the points and compute the distance to the given point
        if (!use_weak)
        {
            for (const auto &p : points)
            {
                double dist = (point - *p).squaredNorm();
                if (dist < min_dist)
                {
                    closest_point = p;
                    min_dist = dist;
                }
            }
        }
        else
        {
            for (const auto &p : w_points)
            {
                if (auto sp = p.lock())
                {
                    double dist = (point - *sp).squaredNorm();
                    if (dist < min_dist)
                    {
                        closest_point = sp;
                        min_dist = dist;
                    }
                }
            }
        }

        return *closest_point;
    }

    PointsPtrVector VoxelBlock::get_closest_points(const MapPoint &point, double max_corresp, int num_points) const
    {
        boost::shared_lock<boost::shared_mutex> lock(mutex);
        std::vector<std::pair<double, MapPointPtr>> closest_points;
        closest_points.reserve(points.size());

        double corresp_sq = max_corresp * max_corresp;

        for (const auto &p : points)
        {
            double dist = (point - *p).squaredNorm();
            if (dist <= corresp_sq)
                closest_points.emplace_back(dist, p);
        }

        auto cmp = [](const auto &a, const auto &b)
        { return a.first < b.first; };
        std::sort(closest_points.begin(), closest_points.end(), cmp);

        PointsPtrVector points_returned;
        points_returned.reserve(std::min(num_points, static_cast<int>(closest_points.size())));

        for (const auto &cp : closest_points)
        {
            points_returned.emplace_back(cp.second);
            if (points_returned.size() == num_points)
                break;
        }

        return points_returned;
    }

    void VoxelBlock::remove_points(const utils::Vec3d &origin, double distance)
    {
        auto dist_sq = distance * distance;
        boost::unique_lock<boost::shared_mutex> lock(mutex);
        points.erase(
            std::remove_if(
                points.begin(), points.end(),
                [&](const auto point)
                {
                    return (point->point - origin).squaredNorm() > dist_sq;
                }),
            points.end());
    }
}