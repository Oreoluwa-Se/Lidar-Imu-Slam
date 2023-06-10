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
        struct PreProcessInfo
        {
            typedef std::shared_ptr<PreProcessInfo> Ptr;

            // lidar information
            double frame_rate;
            double max_range;
            double min_range;
            double min_angle;
            double max_angle;
            int num_scan_lines;
        };

        explicit Lidar(std::string &params_location)
            : config(std::make_shared<PreProcessInfo>()),
              prev_timestamp(0.0), scan_ang_vel(0.0), lidar_end_time(0.0)
        {
            YAML::Node node = YAML::LoadFile(params_location);

            config->frame_rate = node["scan_preprocessing"]["frame_rate"].as<double>();
            config->max_range = node["scan_preprocessing"]["max_range"].as<double>();
            config->min_range = node["scan_preprocessing"]["min_range"].as<double>();
            config->min_angle = node["scan_preprocessing"]["min_angle"].as<double>();
            config->max_angle = node["scan_preprocessing"]["max_angle"].as<double>();
            config->num_scan_lines = node["scan_preprocessing"]["num_scan_lines"].as<int>();

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
            return buffer.empty();
        }
        // lidar information

        std::deque<utils::LidarFrame> get_lidar_buffer_front()
        {
            std::unique_lock<std::mutex> lock(data_mutex);
            return buffer.front();
        }

        void pop()
        {
            std::unique_lock<std::mutex> lock(data_mutex);
            buffer.pop_front();
        }

    public:
        geometry_msgs::TransformStamped current_pose;
        nav_msgs::Odometry odom_msg;
        std::shared_ptr<PreProcessInfo> config;
        double lidar_end_time;

    private:
        // functions
        void setup()
        {
            blind_sq = config->min_range * config->min_range;
            max_sq = config->max_range * config->max_range;
            angle_limit = config->max_angle - config->min_angle;
        }

        void sort_clouds(std::vector<utils::Point::Ptr> &points);

        void split_clouds(const std::vector<utils::Point::Ptr> &points);

        // attributes
        sensor_msgs::PointCloud2::ConstPtr msg_holder;
        std::deque<std::deque<utils::LidarFrame>> buffer; // buffer split into towns.

        // class manipulators
        std::mutex data_mutex;
        double prev_timestamp;
        double scan_ang_vel;
        double angle_limit;
        double blind_sq;
        double max_sq;
        int frame_idx = 1;
    };
}
#endif