/*Stores general Lidar properties and information*/
#ifndef LIDAR_FRAME_HPP
#define LIDAR_FRAME_HPP

#include <pcl_conversions/pcl_conversions.h>
#include "geometry_msgs/TransformStamped.h"
#include "sensor_msgs/PointCloud2.h"
#include "nav_msgs/Odometry.h"
#include "common.hpp"
#include <ros/ros.h>

struct EIGEN_ALIGN16 LidarPoint
{
    PCL_ADD_POINT4D;
    std::uint8_t intensity;
    std::uint16_t ring;
    double timestamp;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

POINT_CLOUD_REGISTER_POINT_STRUCT(
    LidarPoint,
    (float, x, x)(float, y, y)(float, z, z)(std::uint8_t, intensity, intensity)(std::uint16_t, ring, ring)(double, timestamp, timestamp))

namespace frame
{
    using PointCloud = utils::PointCloudXYZI;
    class Lidar
    {

    public:
        typedef std::shared_ptr<Lidar> Ptr;

        struct ProcessingInfo
        {
            typedef std::shared_ptr<ProcessingInfo> Ptr;

            // lidar information
            double frame_rate;
            double max_range;
            double min_range;
            double min_angle;
            double max_angle;
            int num_scan_lines;
            int frame_split_num;

            // voxel information
            double voxel_size;
            int vox_side_length;
            int max_points_per_voxel;

            // Icp information
            bool deskew;
            double min_motion_th;
            int icp_max_iteration;
            double initial_threshold;
            double estimation_threshold;
        };

        explicit Lidar(ros::NodeHandle &nh)
            : config(std::make_shared<ProcessingInfo>()),
              prev_timestamp(0.0), scan_ang_vel(0.0), scan_count(0),
              Li_init_complete(false), lidar_end_time(0.0)
        {
            nh.param<double>("frame_rate", config->frame_rate, 10.0);
            nh.param<double>("max_range", config->max_range, 100.0);
            nh.param<double>("min_range", config->min_range, 5.0);
            nh.param<double>("min_angle", config->min_angle, 0.0);
            nh.param<double>("max_angle", config->max_angle, 360.0);
            nh.param<int>("num_scan_lines", config->num_scan_lines, 16);
            nh.param<int>("frame_split_num", config->frame_split_num, 1);

            nh.param<double>("voxel_size", config->voxel_size, config->max_range / 100.0);
            nh.param<int>("vox_side_length", config->vox_side_length, 3);
            nh.param<int>("max_points_per_voxel", config->max_points_per_voxel, 10);

            nh.param<bool>("deskew", config->deskew, false);
            nh.param<double>("min_motion_th", config->min_motion_th, 0.1);
            nh.param<int>("icp_max_iteration", config->icp_max_iteration, 500);
            nh.param<double>("initial_threshold", config->initial_threshold, 2.0);
            nh.param<double>("estimation_threshold", config->estimation_threshold, 0.0001);

            scan_ang_vel = utils::calc_scan_ang_vel(config->frame_rate);
            setup();
        }

        void initialize(const sensor_msgs::PointCloud2::ConstPtr &msg);
        void process_frame();

        // accessors
        double return_prev_ts()
        {
            std::unique_lock<std::mutex> lock(data_mutex);
            return prev_timestamp;
        }

        bool buffer_empty()
        {
            std::unique_lock<std::mutex> lock(data_mutex);
            return split_buffer.empty();
        }
        // lidar information

        PointCloud::Ptr get_lidar_buffer_front()
        {
            std::unique_lock<std::mutex> lock(data_mutex);
            return processed_buffer.front();
        }

        PointCloud::Ptr get_split_lidar_buffer_front()
        {
            std::unique_lock<std::mutex> lock(data_mutex);
            const auto &p = split_buffer.front();
            return p.second;
        }

        double get_pc_time_ms()
        {
            std::unique_lock<std::mutex> lock(data_mutex);
            const auto &p = split_buffer.front();
            return p.second->points[0].curvature;
        }

        double get_pc_time_s()
        {
            return get_pc_time_ms() / double(1000);
        }

        double curr_acc_segment_time()
        {
            std::unique_lock<std::mutex> lock(data_mutex);
            const auto &p = split_buffer.front();
            return p.first;
        }

        void pop()
        {
            std::unique_lock<std::mutex> lock(data_mutex);
            split_buffer.pop_front();
        }

        // functions for broadcasting ros tings
        void set_current_pose_nav(
            const utils::Vec3d &pose, const Eigen::Quaterniond &quat,
            const ros::Time &time, std::string &odom_frame, std::string &child_frame);

    public:
        geometry_msgs::TransformStamped current_pose;
        nav_msgs::Odometry odom_msg;
        std::shared_ptr<ProcessingInfo> config;
        double lidar_end_time;

    private:
        // functions
        void setup()
        {
            blind_sq = config->min_range * config->min_range;
            max_sq = config->max_range * config->max_range;
            angle_limit = config->max_angle - config->min_angle;
        }

        void iqr_processing(PointCloud::Ptr &surface_cloud, std::vector<double> &);

        void sort_clouds(PointCloud::Ptr &surface_cloud);

        void split_clouds(const PointCloud::Ptr &surface_cloud, double &message_time);

        // attributes
        sensor_msgs::PointCloud2::ConstPtr msg_holder;
        std::deque<PointCloud::Ptr> processed_buffer;                // holds processed pointcloud from msg.
        std::deque<std::pair<double, PointCloud::Ptr>> split_buffer; // buffer split into towns.

        // class manipulators
        std::mutex data_mutex;
        double prev_timestamp;
        double scan_ang_vel;
        double angle_limit;
        double blind_sq;
        int scan_count;
        double max_sq;
        bool Li_init_complete;
    };
}
#endif