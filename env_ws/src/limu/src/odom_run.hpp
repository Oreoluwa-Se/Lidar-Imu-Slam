#ifndef ODOM_RUN_HPP
#define ODOM_RUN_HPP

#include "tf2_ros/static_transform_broadcaster.h"
#include "geometry_msgs/TransformStamped.h"
#include "tf2_ros/transform_broadcaster.h"
#include "limu/sensors/lidar/frame.hpp"
#include "limu/sensors/sync_frame.hpp"
#include "limu/sensors/imu/frame.hpp"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/Imu.h"
#include "nav_msgs/Path.h"
#include <ros/ros.h>

class Odometry
{
public:
    struct Trackers
    {
        Trackers()
            : time_diff_imu_wrt_lidar(0.0), time_lag_IMU_wtr_lidar(0.0),
              lidar_end_time(0.0), move_start_time(0.0),
              timediff_set_flag(false), imu_en(false), data_accum_finished(false),
              data_accum_start(false), exit_flag(false), lidar_pushed(false), reset_flag(false) {}

        double time_diff_imu_wrt_lidar, time_lag_IMU_wtr_lidar, lidar_end_time, move_start_time;
        bool timediff_set_flag, imu_en, data_accum_finished, data_accum_start, exit_flag, lidar_pushed, reset_flag;
    };

    explicit Odometry(ros::NodeHandle &nh)
        : lidar_ptr(std::make_shared<frame::Lidar>(nh)),
          imu_ptr(std::make_shared<frame::Imu>(nh)),
          tracker(Trackers())
    {
        // subscribing to lidar and imu topics.
        std::string lidar_topic, imu_topic;
        nh.param<std::string>("lidar_topic", lidar_topic, "/rslidar_points");
        nh.param<std::string>("imu_topic", imu_topic, "/imu_ned/data");
        lidar_sub = nh.subscribe(lidar_topic, queue_size, &Odometry::lidar_callback, this);
        imu_sub = nh.subscribe(imu_topic, queue_size, &Odometry::imu_callback, this);

        // publish static transform to connect child frame to baselink
        nh.param<std::string>("odom_frame", odom_frame, "odom");
        nh.param<std::string>("child_frame", child_frame, "base_link");
        if (child_frame != "base_link")
        {
            static tf2_ros::StaticTransformBroadcaster br;
            geometry_msgs::TransformStamped baselink_msg;
            baselink_msg.header.stamp = ros::Time::now();
            baselink_msg.transform.translation.x = 0.0;
            baselink_msg.transform.translation.y = 0.0;
            baselink_msg.transform.translation.z = 0.0;
            baselink_msg.transform.rotation.x = 0.0;
            baselink_msg.transform.rotation.y = 0.0;
            baselink_msg.transform.rotation.z = 0.0;
            baselink_msg.transform.rotation.w = 1.0;
            baselink_msg.header.frame_id = child_frame;
            baselink_msg.child_frame_id = "base_link";
            br.sendTransform(baselink_msg);
        }
    }

public:
    void run();

private:
    // functions
    void lidar_callback(const sensor_msgs::PointCloud2::ConstPtr &msg);
    void imu_callback(const sensor_msgs::Imu::ConstPtr &msg);
    bool lidar_process(frame::LidarImuInit::Ptr &meas);

    void estimate_lidar_odometry(frame::LidarImuInit::Ptr &meas);

    // attributes
    frame::Lidar::Ptr lidar_ptr;
    frame::Imu::Ptr imu_ptr;
    Trackers tracker;

    // subscribe to pointcloud and imu
    ros::Subscriber lidar_sub;
    ros::Subscriber imu_sub;
    std::string child_frame;
    std::string odom_frame;
    std::mutex data_mutex;
    int queue_size{1};
};
#endif