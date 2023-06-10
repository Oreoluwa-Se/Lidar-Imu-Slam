#include "limu/sensors/lidar/frame.hpp"
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>
#include <map>

namespace frame
{
    void Lidar::initialize(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex);
        const auto curr_time = msg->header.stamp.toSec();

        if (curr_time < prev_timestamp)
        {
            ROS_ERROR("Lidar Buffer Looped. Clearning buffer");
            buffer.clear();
        }

        msg_holder = std::move(msg);
        prev_timestamp = curr_time;
    }

    void Lidar::sort_clouds(std::vector<utils::Point::Ptr> &points)
    {
        auto compare_curvature = [&](const utils::Point::Ptr &i, const utils::Point::Ptr &j)
        {
            return i->timestamp < j->timestamp;
        };
        std::sort(
            points.begin(), points.end(),
            compare_curvature);
    }

    void Lidar::split_clouds(const std::vector<utils::Point::Ptr> &points)
    {
        std::deque<utils::LidarFrame> frame;
        double prev_ts = -1.0;

        for (const auto &point : points)
        {
            const auto &curr_ts = point->timestamp;
            if (curr_ts != prev_ts)
            {
                frame.emplace_back();
                frame.back().timestamp = curr_ts;
                prev_ts = curr_ts;
            }

            frame.back().pc.emplace_back(point);
        }

        buffer.emplace_back(std::move(frame));
    }

    void Lidar::process_frame()
    {
        std::lock_guard<std::mutex> lock(data_mutex);

        // convert ros message to pcl point cloud
        pcl::PointCloud<LidarPoint>::Ptr cloud(new pcl::PointCloud<LidarPoint>);
        pcl::fromROSMsg(*msg_holder, *cloud);

        // get size of point cloud
        int cloud_size = cloud->size();

        // storage initializer
        std::vector<utils::Point::Ptr> points;
        points.reserve(cloud_size);

        // holders
        std::vector<bool> is_first(config->num_scan_lines, false);
        std::vector<double> yaw_fp(config->num_scan_lines, 0.0);
        std::vector<double> yaw_last(config->num_scan_lines, 0.0);
        std::vector<double> time_last(config->num_scan_lines, 0.0);

        // want to be able to calculate offset time between first scan and last scan from the point cloud
        bool has_offset_time = !cloud->points.empty() && cloud->points.back().timestamp > 0;
        if (!has_offset_time)
        {
            ROS_WARN("Final scan has no offset time. Compute using constant rotation model.");
            std::fill(is_first.begin(), is_first.end(), true);
        }

        double message_time = msg_holder->header.stamp.toSec(); // seconds

        size_t inlier_count = 0;
        // get inliers
        for (size_t i_dx = 0; i_dx < cloud_size; i_dx++)
        {
            const auto &point = cloud->points[i_dx];

            if (isnan(point.x) || isnan(point.y) || isnan(point.z))
                continue;

            // skip points that are out of bounds
            double dist = point.x * point.x + point.y * point.y + point.z * point.z;
            if (!(dist >= blind_sq && dist <= max_sq))
                continue;

            // double x, double y, double z, double intensity, double timestamp, double dt
            double dt = point.timestamp - message_time + 0.1;
            utils::Point::Ptr added_pt = std::make_shared<utils::Point>(
                point.x, point.y, point.z, point.intensity, point.timestamp, dt);
            added_pt->frame_id = frame_idx;

            int layer = point.ring;
            if (!has_offset_time)
            {
                double yaw_angle = atan2(added_pt->y(), added_pt->x()) * 57.2957; // in degrees

                if (is_first[layer])
                {
                    yaw_fp[layer] = yaw_angle;
                    is_first[layer] = false;
                    added_pt->dt = 0.0;
                    yaw_last[layer] = yaw_angle;
                    time_last[layer] = added_pt->dt; // s
                    continue;
                }

                // compute offset time
                double angle_diff = yaw_angle <= yaw_fp[layer] ? (yaw_fp[layer] - yaw_angle) : (yaw_fp[layer] - yaw_angle + angle_limit);
                added_pt->dt = angle_diff / scan_ang_vel;

                if (added_pt->dt < time_last[layer])
                    added_pt->dt += angle_limit / scan_ang_vel;

                yaw_last[layer] = yaw_angle;
                time_last[layer] = added_pt->dt;
            }

            points.emplace_back(added_pt);
        }

        // sort according to increasing curvature.
        sort_clouds(points);

        // partition pointclouds when necessary.
        split_clouds(points);

        frame_idx++;
    }

}