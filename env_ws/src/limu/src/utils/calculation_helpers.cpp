#include "limu/utils/calculation_helpers.hpp"
#include <tbb/parallel_sort.h>

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
        points.resize(msg.points.size());

        std::transform(
            msg.points.begin(), msg.points.end(),
            points.begin(),
            [&](const auto &point)
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

        std::transform(
            points.begin(), points.end(), points.begin(),
            [&](const auto &point)
            { return T * point; });
    }

    void transform_points(const SE3d &T, std::vector<utils::Point> &points)
    {
        if (points.empty())
        {
            std::cout << "[INFO] utils::transform_points the points vector is empty\n";
            return;
        }

        for (auto &point : points)
            point.point = T * point.point;
    }

    void transform_points(const SE3d &T, std::vector<utils::Point::Ptr> &points)
    {
        if (points.empty())
        {
            std::cout << "[INFO] utils::transform_points the points vector is empty\n";
            return;
        }

        for (auto &point : points)
            point->point = T * point->point;
    }

    utils::Vec3Tuple get_motion(const SE3d &start_pose, const SE3d &end_pose, double dt)
    {
        // get twist
        const utils::vector<6> twist = utils::delta_pose(start_pose, end_pose) / dt;
        return {twist.head<3>(), twist.tail<3>()};
    }

    utils::Voxel get_vox_index(const utils::Vec3d &point, double vox_size)
    {
        return utils::Voxel(static_cast<int>(std::round(point.x() / vox_size)),
                            static_cast<int>(std::round(point.y() / vox_size)),
                            static_cast<int>(std::round(point.z() / vox_size)));
    }

    utils::Voxel get_vox_index(const utils::Point &point, double vox_size)
    {
        return utils::Voxel(static_cast<int>(std::round(point.point.x() / vox_size)),
                            static_cast<int>(std::round(point.point.y() / vox_size)),
                            static_cast<int>(std::round(point.point.z() / vox_size)));
    }

    utils::Voxel get_vox_index(const utils::Point::Ptr &point, double vox_size)
    {
        return utils::Voxel(static_cast<int>(std::round(point->point.x() / vox_size)),
                            static_cast<int>(std::round(point->point.y() / vox_size)),
                            static_cast<int>(std::round(point->point.z() / vox_size)));
    }

    utils::Vec3d rotation_matrix_to_euler_angles(const utils::Mat3d &rot)
    {
        double sy = sqrt(rot(0, 0) * rot(0, 0) + rot(1, 0) * rot(1, 0));
        bool singular = sy < 1e-6;
        double x, y, z;
        if (!singular)
        {
            x = atan2(rot(2, 1), rot(2, 2));
            y = atan2(-rot(2, 0), sy);
            z = atan2(rot(1, 0), rot(0, 0));
        }
        else
        {
            x = atan2(-rot(1, 2), rot(1, 1));
            y = atan2(-rot(2, 0), sy);
            z = 0;
        }
        utils::Vec3d ang(x, y, z);
        return ang;
    }

    double calculate_median(const std::vector<double> &sorted_data, size_t start, size_t end)
    {
        size_t n = end - start + 1;
        if (n % 2 == 0)
            return (sorted_data[start + n / 2 - 1] + sorted_data[start + n / 2]) / 2.0;
        else
            return sorted_data[start + n / 2];
    }

    std::vector<double> IQR(const std::vector<double> &data)
    {
        std::vector<double> sorted_data = data;
        tbb::parallel_sort(sorted_data.begin(), sorted_data.end());

        size_t n = sorted_data.size();
        // Calculate the lower quartile (Q1)
        double q1 = calculate_median(sorted_data, 0, n / 2 - 1);

        // Calculate the upper quartile (Q3)
        double q3 = (n % 2 == 0) ? calculate_median(sorted_data, n / 2, n - 1) : calculate_median(sorted_data, n / 2 + 1, n - 1);

        return {q1, q3, q3 - q1};
    }

    double calculate_mad(const std::vector<double> &data)
    {
        std::vector<double> sorted_data = data;
        tbb::parallel_sort(sorted_data.begin(), sorted_data.end());

        size_t n = sorted_data.size();
        double median = calculate_median(sorted_data, 0, n - 1);

        std::vector<double> abs_deviations;
        abs_deviations.reserve(data.size());
        for (const auto &value : data)
            abs_deviations.push_back(std::abs(value - median));

        double mad = calculate_median(abs_deviations, 0, n - 1);

        return mad;
    }
    void vec_2_matrices(const std::vector<double> &R, const std::vector<double> &t, utils::Mat3d &r_mat, utils::Vec3d &t_vec)
    {
        if (R.size() != 9)
            throw std::invalid_argument("Invalid size of input vector R. Expected size: 9.");

        if (t.size() != 3)
            throw std::invalid_argument("Invalid size of input vector t. Expected size: 3 or more.");

        r_mat = Eigen::Matrix3d::Zero();
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
                r_mat(i, j) = R[i * 3 + j];
        }

        utils::rot_mat_norm(r_mat);
        t_vec = utils::Vec3d(t[0], t[1], t[2]);
    }

    void rot_mat_norm(utils::Mat3d &rot_mat)
    {
        for (int i = 0; i < 3; ++i)
        {
            double length = rot_mat.col(i).norm();
            rot_mat.col(i) /= length;
        }
    }
}