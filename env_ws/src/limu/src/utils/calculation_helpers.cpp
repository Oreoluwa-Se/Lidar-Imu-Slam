#include "limu/utils/calculation_helpers.hpp"

namespace
{
    sensor_msgs::PointField get_timestamp_field(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        sensor_msgs::PointField ts_field;
        for (const auto &field : msg->fields)
        {
            if ((field.name == "t" || field.name == "timestamp" || field.name == "time"))
                ts_field = field;
        }
        if (!ts_field.count)
        {
            throw std::runtime_error("Field 't', 'timestamp' or 'time' not existing");
        }

        return ts_field;
    }

    std::vector<double> extract_timestamps_from_msg(
        const sensor_msgs::PointCloud2::ConstPtr &msg,
        const sensor_msgs::PointField &field)
    {
        // here we go
        const size_t n_points = msg->height * msg->width;
        std::vector<double> timestamps;
        timestamps.reserve(n_points);

        // timesteps can be unsigned integers
        if (field.name == "t" || field.name == "timestamp")
        {
            // get pointer to field name.
            sensor_msgs::PointCloud2ConstIterator<uint32_t> msg_t(*msg, field.name);
            for (size_t idx = 0; idx < n_points; ++idx, ++msg_t)
                timestamps.emplace_back(static_cast<double>(*msg_t));

            return utils::normalize_timestamps(timestamps);
        }

        // timestamps are floating point values
        sensor_msgs::PointCloud2ConstIterator<double> msg_t(*msg, field.name);
        for (size_t idx = 0; idx < n_points; ++idx, ++msg_t)
            timestamps.emplace_back(static_cast<double>(*msg_t));

        return timestamps;
    }
}

namespace utils
{
    std::vector<double> normalize_timestamps(const std::vector<double> &timestamps)
    {
        const double max_timestamp = *std::max_element(timestamps.cbegin(), timestamps.cend());

        if (max_timestamp < 1.0)
            return timestamps;

        std::vector<double> ts_norm(timestamps.size());
        std::transform(
            timestamps.cbegin(), timestamps.cend(), ts_norm.begin(),
            [&](const auto &ts)
            { return ts / max_timestamp; });

        return ts_norm;
    }

    std::vector<double> get_time_stamps(const sensor_msgs::PointCloud2::ConstPtr &msg,
                                        const sensor_msgs::PointField &field)
    {
        auto ts = [&]()
        {
            if (field.count)
                return field;

            return get_timestamp_field(msg);
        }();

        // extract time!!!
        return extract_timestamps_from_msg(msg, ts);
    }

    utils::Vec3dVector pointcloud2eigen(const utils::PointCloudXYZI &msg)
    {
        utils::Vec3dVector points;
        points.reserve(msg.points.size());

        std::transform(
            msg.points.begin(), msg.points.end(),
            std::back_inserter(points),
            [](const auto &point)
            {
                return utils::Vec3d(point.x, point.y, point.z);
            });

        return points;
    }

    utils::vector<6> delta_pose(const SE3d &first, const SE3d &last)
    {
        return (first.inverse() * last).log();
    }

    double calc_scan_ang_vel(int frame_rate)
    {
        // frame rate in Hz. Returns in seconds
        return double(frame_rate) * (360.0 / 1000.0);
    }

    utils::Mat3d skew_matrix(const utils::Vec3d &vec)
    {
        // clang-format on
        return SO3d::hat(vec);
    }

    SE3d vector6d_to_mat4d(const utils::vector<6> &x)
    {
        return SE3d::exp(x);
    }

    void transform_points(const SE3d &T, utils::Vec3dVector &points)
    {
        if (points.empty())
        {
            std::cout << "[INFO] utils::transform_points the points vector is empty\n";
            return;
        }

        std::for_each(
            points.begin(), points.end(),
            [&](const auto &point)
            {
                return T * point;
            });
    }

    utils::Vec3Tuple get_motion(const SE3d &start_pose, const SE3d &end_pose, double dt)
    {
        // get twist
        const utils::vector<6> twist = utils::delta_pose(start_pose, end_pose) / dt;
        return {twist.head<3>(), twist.tail<3>()};
    }

}