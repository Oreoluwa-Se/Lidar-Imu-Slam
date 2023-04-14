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
    struct LidarTSFrame
    {
        LidarTSFrame()
            : pc(new PointCloud()),
              acc_time(0.0), freq(0.0),
              curvature_time(0.0) {}

        PointCloud::Ptr pc;
        double curvature_time; // represents all curvature point for current set
        double acc_time;       // in milliseconds
        double freq;           // in hz
    };

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
              prev_timestamp(0.0), scan_ang_vel(0.0), lidar_end_time(0.0)
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
            const auto &front_buff = split_buffer.front();
            return front_buff.pc;
        }

        double get_pc_time_ms()
        {
            std::unique_lock<std::mutex> lock(data_mutex);
            const auto &front_buff = split_buffer.front();
            return front_buff.curvature_time;
        }

        double get_pc_time_s()
        {
            return get_pc_time_ms() / double(1000);
        }

        double accumulated_segment_time()
        {
            std::unique_lock<std::mutex> lock(data_mutex);
            const auto &front_buff = split_buffer.front();
            return front_buff.acc_time;
        }

        double get_freq_hz()
        {
            std::unique_lock<std::mutex> lock(data_mutex);
            const auto &front_buff = split_buffer.front();
            return front_buff.freq;
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
        std::deque<LidarTSFrame> split_buffer; // buffer split into towns.

        // class manipulators
        std::mutex data_mutex;
        double prev_timestamp;
        double scan_ang_vel;
        double angle_limit;
        double blind_sq;
        double max_sq;
    };
}
#endif