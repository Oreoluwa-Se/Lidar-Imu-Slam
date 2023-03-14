#include "limu/sensors/lidar/frame.hpp"
#include <numeric>
namespace
{
    constexpr int MIN_SCAN_COUNT = 20;
}

namespace frame
{
    void Lidar::initialize(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex);
        scan_count++;
        const auto curr_time = msg->header.stamp.toSec();

        if (curr_time < prev_timestamp)
        {
            ROS_ERROR("Lidar Buffer Looped. Clearning buffer");
            processed_buffer.clear();
            timestamps.clear();
            accumulated_segment_time.clear();
        }

        msg_holder = std::move(msg);
        prev_timestamp = curr_time;
    }

    void Lidar::sort_clouds(
        PointCloud::Ptr &surface_cloud, std::vector<double> &valid_timestamp)
    {
        std::vector<int> indices(valid_timestamp.size());
        std::iota(indices.begin(), indices.end(), 0);

        auto compare_curvature = [&](const int i, const int j)
        {
            return surface_cloud->points[i].curvature < surface_cloud->points[j].curvature;
        };
        std::sort(indices.begin(), indices.end(), compare_curvature);

        std::vector<utils::PointNormal> inliers;
        std::vector<double> valid_times_sorted;
        for (const auto idx : indices)
        {
            inliers.emplace_back(surface_cloud->points[idx]);
            valid_times_sorted.emplace_back(valid_timestamp[idx]);
        }

        // replace in original points
        surface_cloud->points.assign(inliers.begin(), inliers.end());
        valid_timestamp = std::move(valid_times_sorted);
    }

    void Lidar::split_clouds(
        PointCloud::Ptr &surface_cloud, std::vector<double> &valid_timestamp, double &message_time)
    {

        double message_time_ms = message_time * 1000; // convert to ms
        double last_frame_end_time = message_time_ms;

        const size_t valid_pcl_size = surface_cloud->points.size();

        // split processed frame
        int valid_num = 0, cut_num = 0;
        const int required_cut_num = (scan_count < MIN_SCAN_COUNT) ? 1 : config->frame_split_num;

        PointCloud::Ptr split_surface(new PointCloud());
        std::vector<double> times;

        for (auto idx = 1; idx < valid_pcl_size; ++idx)
        {
            valid_num++;

            // accumulate time elapsed at each point.
            surface_cloud->points[idx].curvature += message_time_ms - last_frame_end_time;
            split_surface->points.push_back(std::move(surface_cloud->points[idx]));

            times.push_back(std::move(valid_timestamp[idx]));

            if (valid_num == static_cast<int>(((cut_num + 1) * valid_pcl_size / required_cut_num) - 1))
            {
                cut_num++;
                // new pointer
                PointCloud::Ptr temp(new PointCloud());
                *temp = *split_surface;
                // update holders
                processed_buffer.emplace_back(std::move(temp));
                timestamps.push_back(utils::normalize_timestamps(times));

                accumulated_segment_time.push_back(last_frame_end_time / double(1000)); // sec

                // increment or reset parameters
                last_frame_end_time += surface_cloud->points[idx].curvature;
                times.clear();
                split_surface->points.clear();
                split_surface->reserve(valid_pcl_size * 2 / required_cut_num);
                times.reserve(split_surface->points.size());
            }
        }
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
        std::vector<double> valid_timestamp;

        valid_timestamp.reserve(cloud_size);

        // holders
        std::vector<bool> is_first(config->num_scan_lines, false);
        std::vector<double> yaw_fp(config->num_scan_lines, 0.0);
        std::vector<double> yaw_last(config->num_scan_lines, 0.0);
        std::vector<double> time_last(config->num_scan_lines, 0.0);

        const auto extracted_time = utils::get_time_stamps(msg_holder);

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
            valid_timestamp.emplace_back(extracted_time[i_dx]);
        }

        // sort according to increasing curvature.
        sort_clouds(surface_cloud, valid_timestamp);

        // partition pointclouds when necessary.
        split_clouds(surface_cloud, valid_timestamp, message_time);
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