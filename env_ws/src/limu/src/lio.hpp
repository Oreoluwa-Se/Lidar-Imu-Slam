#ifndef ODOM_RUN_HPP
#define ODOM_RUN_HPP

#include "tf2_ros/static_transform_broadcaster.h"
#include <tf2_ros/transform_listener.h>
#include "geometry_msgs/TransformStamped.h"
#include "tf2_ros/transform_broadcaster.h"
#include "limu/sensors/lidar/frame.hpp"
#include "limu/sensors/sync_frame.hpp"
#include "limu/sensors/imu/frame.hpp"
#include "limu/kalman/ekf.hpp"
#include "geometry_msgs/PoseStamped.h"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/Imu.h"
#include "nav_msgs/Path.h"
#include <ros/ros.h>
#include "common.hpp"
#include <ros/timer.h>
#include <future>
#include <thread>

class LIO
{
public:
    struct Trackers
    {
        bool reset_flag{};
        bool exit_flag{};
        bool flag_first_scan{true};
        bool flag_reset{};
        bool first_frame{true};

        double first_lidar_time{};
        double current_time{};
        double time_update_last{};
        double time_predict_last_const{};
        double propagate_time{};
        double solve_time{};
        double update_time{};
        double imu_last_time{};
    };

    explicit LIO(ros::NodeHandle &nh) : tracker(Trackers()), use_frames(false)
    {
        nh.getParam("config_loc", params_config);
        YAML::Node node = YAML::LoadFile(params_config);

        // pointers
        lidar_ptr = std::make_shared<frame::Lidar>(params_config);
        imu_ptr = std::make_shared<frame::Imu>(params_config);
        ekf = std::make_unique<odometry::EKF>(params_config);

        // subscribing to lidar and imu topics.
        lidar_sub = nh.subscribe(
            node["common"]["lidar_topic"].as<std::string>(),
            queue_size, &LIO::lidar_callback, this);
        imu_sub = nh.subscribe(
            node["common"]["imu_topic"].as<std::string>(),
            queue_size, &LIO::imu_callback, this);

        update_freq = node["common"]["update_every_k_ts"].as<int>();

        // publish static transform to connect child frame to baselink
        odom_frame = node["common"]["odom_frame"].as<std::string>();
        child_frame = node["common"]["child_frame"].as<std::string>();
        run_lidar = node["common"]["run_lidar"].as<bool>();

        // initialize publishers
        initialize_transforms(params_config);
        initialize_publishers(nh);

        imu_ptr->enabled = true;
        frontend_thread = std::thread(&LIO::run, this);
    }

    ~LIO()
    {
        frontend_thread.join();
    }

public:
    void run();

private:
    Sophus::SE3d get_transform(const std::string &source_frame, const std::string &target_frame);

    // starter functions
    void publish_transform(const std::string &child_frame, const Sophus::SE3d &transform, tf2_ros::StaticTransformBroadcaster &br);
    void initialize_publishers(ros::NodeHandle &nh);
    void initialize_transforms(std::string &loc);

    void lidar_callback(const sensor_msgs::PointCloud2::ConstPtr &msg);
    void imu_callback(const sensor_msgs::Imu::ConstPtr &msg);

    // sensor packaging systems
    bool sync_packages(frame::LidarImuInit::Ptr &meas);

    // running procedures
    bool kalman_filter_run(const frame::SensorData &data, bool imu_enabled);
    void imu_runner(const frame::SensorData &data, double sf = 1);
    bool lidar_runner(const frame::SensorData &data);

    void publish(int curr_idx = 0);

    void process_frame(const frame::LidarImuInit::Ptr &meas);

    // attributes
    frame::Lidar::Ptr lidar_ptr;
    frame::Imu::Ptr imu_ptr;
    odometry::EKF::Ptr ekf;
    Trackers tracker;

    tf2_ros::TransformBroadcaster tf_broadcaster;

    // subscribe to pointcloud and imu
    ros::Subscriber lidar_sub;
    ros::Subscriber imu_sub;

    ros::Publisher odom_publisher;
    ros::Publisher traj_publisher;
    ros::Publisher frame_publisher;
    ros::Publisher kpoints_publisher;
    ros::Publisher local_map_publisher;

    // message for publishing path
    nav_msgs::Path path_msgs;
    std::string params_config;

    // place holders:
    std::string child_frame;
    std::string odom_frame;
    std::mutex data_mutex;
    int queue_size{1};
    int update_freq;
    bool imu_propagated = false;
    bool use_frames, run_lidar = false;

    std::thread frontend_thread;
};
#endif