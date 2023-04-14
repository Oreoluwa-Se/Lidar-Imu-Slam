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
            split_buffer.clear();
        }

        msg_holder = std::move(msg);
        prev_timestamp = curr_time;
    }

    void Lidar::sort_clouds(PointCloud::Ptr &surface_cloud)
    {
        auto compare_curvature = [&](const utils::PointNormal &i, const utils::PointNormal &j)
        {
            return i.curvature < j.curvature;
        };
        std::sort(
            surface_cloud->points.begin(), surface_cloud->points.end(),
            compare_curvature);
    }

    void Lidar::split_clouds(const PointCloud::Ptr &surface_cloud, double &message_time)
    {
        double message_time_ms = message_time * 1000; // convert to ms
        double last_frame_end_time = message_time_ms;
        double prev_ts = 0.0, curr_freq = 0.0;

        // each curvature is assigned it's own LidarTSFrame struct
        std::map<double, LidarTSFrame> curvature_keys;

        for (const auto &point : surface_cloud->points)
        {
            const double &cv_time = point.curvature;
            auto it = curvature_keys.find(cv_time);
            if (it == curvature_keys.end())
            {
                curvature_keys.emplace(cv_time, LidarTSFrame());
                auto &frame = curvature_keys[cv_time];
                frame.curvature_time = cv_time;
                frame.acc_time = cv_time + message_time_ms - last_frame_end_time;
                last_frame_end_time += frame.acc_time;

                // calculate the frequency between two timestamps
                curr_freq = (cv_time - prev_ts) / 1000.0; // ms
                frame.freq = 1 / curr_freq;

                prev_ts = cv_time;
            }

            // include point for current timestamp
            curvature_keys[cv_time].pc->push_back(point);
        }

        // updating the deque buffer
        std::transform(
            std::make_move_iterator(curvature_keys.begin()),
            std::make_move_iterator(curvature_keys.end()),
            std::back_inserter(split_buffer),
            [](std::pair<double, LidarTSFrame> &&p)
            { return std::move(p.second); });
    }

    void Lidar::iqr_processing(PointCloud::Ptr &surface_cloud, std::vector<double> &dist)
    {
        const auto &iqr_val = outlier::IQR(dist);
        double low_bound = iqr_val[0] - IQR_TUCHEY * iqr_val[2];
        double high_bound = iqr_val[1] + IQR_TUCHEY * iqr_val[2];

        tbb::concurrent_vector<utils::PointNormal> inliers;
        inliers.reserve(dist.size());

        tbb::parallel_for(
            std::size_t(0), dist.size(),
            [&](std::size_t idx)
            {
                const auto &distance = dist[idx];
                if (distance >= low_bound && distance <= high_bound)
                    inliers.emplace_back(surface_cloud->points[idx]);
            });

        surface_cloud->points.assign(inliers.begin(), inliers.end());
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
        PointCloud::Ptr surface_cloud(new PointCloud());
        surface_cloud->reserve(cloud_size);
        std::vector<double> distance;
        distance.reserve(cloud_size);

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

            // skip points that are out of bounds
            double dist = point.x * point.x + point.y * point.y + point.z * point.z;
            if (!(dist >= blind_sq && dist <= max_sq) || isnan(point.x) || isnan(point.y) || isnan(point.z))
                continue;

            utils::PointNormal added_pt;
            added_pt.normal_x = 0;
            added_pt.normal_y = 0;
            added_pt.normal_z = 0;
            added_pt.x = point.x;
            added_pt.y = point.y;
            added_pt.z = point.z;
            added_pt.intensity = point.intensity;
            // stores time difference between current point and original message time.
            added_pt.curvature = (point.timestamp - message_time + 0.1) * 1000; // ms

            int layer = point.ring;
            if (!has_offset_time)
            {
                double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957; // in degrees

                if (is_first[layer])
                {
                    yaw_fp[layer] = yaw_angle;
                    is_first[layer] = false;
                    added_pt.curvature = 0.0;
                    yaw_last[layer] = yaw_angle;
                    time_last[layer] = added_pt.curvature; // ms
                    continue;
                }

                // compute offset time
                double angle_diff = yaw_angle <= yaw_fp[layer] ? (yaw_fp[layer] - yaw_angle) : (yaw_fp[layer] - yaw_angle + angle_limit);
                added_pt.curvature = angle_diff / scan_ang_vel;

                if (added_pt.curvature < time_last[layer])
                    added_pt.curvature += angle_limit / scan_ang_vel;

                yaw_last[layer] = yaw_angle;
                time_last[layer] = added_pt.curvature;
            }

            surface_cloud->points.emplace_back(added_pt);
            distance.push_back(dist);
        }

        iqr_processing(surface_cloud, distance);

        // sort according to increasing curvature.
        sort_clouds(surface_cloud);

        // partition pointclouds when necessary.
        split_clouds(surface_cloud, message_time);
    }

    void Lidar::set_current_pose_nav(
        const utils::Vec3d &translation, const Eigen::Quaterniond &quat,
        const ros::Time &time, std::string &odom_frame, std::string &child_frame)
    {

        current_pose.header.stamp = time;
        current_pose.header.frame_id = odom_frame;
        current_pose.child_frame_id = child_frame;
        current_pose.transform.rotation.x = quat.x();
        current_pose.transform.rotation.y = quat.y();
        current_pose.transform.rotation.z = quat.z();
        current_pose.transform.rotation.w = quat.w();
        current_pose.transform.translation.x = translation.x();
        current_pose.transform.translation.y = translation.y();
        current_pose.transform.translation.z = translation.z();

        odom_msg.header.stamp = time;
        odom_msg.header.frame_id = odom_frame;
        odom_msg.child_frame_id = child_frame;
        odom_msg.pose.pose.orientation.x = quat.x();
        odom_msg.pose.pose.orientation.y = quat.y();
        odom_msg.pose.pose.orientation.z = quat.z();
        odom_msg.pose.pose.orientation.w = quat.w();
        odom_msg.pose.pose.position.x = translation.x();
        odom_msg.pose.pose.position.y = translation.y();
        odom_msg.pose.pose.position.z = translation.z();
    }
}