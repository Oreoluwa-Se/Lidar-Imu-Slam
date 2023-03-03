#include "odom_run.hpp"

void Odometry::imu_callback(const sensor_msgs::Imu::ConstPtr &msg)
{
    imu_ptr->process_data(msg);

    std::unique_lock<std::mutex> lock(data_mutex);
    imu_ptr->imu_time_compensation(tracker.time_diff_imu_wrt_lidar, tracker.time_lag_IMU_wtr_lidar);
    lock.unlock();

    imu_ptr->update_buffer();
}

void Odometry::lidar_callback(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    lidar_ptr->initialize(msg);

    auto diff_check = imu_ptr->return_prev_ts() - lidar_ptr->return_prev_ts();

    std::unique_lock<std::mutex> lock(data_mutex);
    if (abs(diff_check) > 1.0 && !imu_ptr->buffer_empty() && !tracker.timediff_set_flag)
    {
        tracker.timediff_set_flag = true;
        tracker.time_diff_imu_wrt_lidar = diff_check;
    }
    lock.unlock();

    // split frames when necessary.
    lidar_ptr->process_frame();
}

bool Odometry::lidar_process(frame::LidarImuInit::Ptr &meas)
{
    // For testing lidar process
    if (lidar_ptr->buffer_empty())
        return false;

    if (!tracker.lidar_pushed)
    {
        auto holder = lidar_ptr->get_lidar_buffer_front();
        const auto &processed_frame = std::get<0>(holder);

        if (processed_frame->points.size() <= 1)
        {
            ROS_WARN("Too few input point cloud!\n");
            lidar_ptr->pop();
            return false;
        }

        const auto &orig_frame = std::get<1>(holder);

        // load information into frame
        meas = std::make_shared<frame::LidarImuInit>();
        meas->processed_frame = processed_frame;
        meas->original_frame = orig_frame;
        // ============================ //
        meas->time_buffer = lidar_ptr->get_segment_ts_front();     // normalized time stamps
        meas->lidar_beg_time = lidar_ptr->curr_acc_segment_time(); // uint:s
        tracker.lidar_end_time = meas->lidar_beg_time + meas->processed_frame->points.back().curvature / double(1000);
        tracker.lidar_pushed = true;
    }

    lidar_ptr->pop();
    tracker.lidar_pushed = false;

    return true;
}

void Odometry::estimate_lidar_odometry(frame::LidarImuInit::Ptr &meas)
{
    // estimate lidar odometry

    // publish pose and pointclouds
    return;
}

void Odometry::run()
{
    ros::Rate rate(5000);
    frame::LidarImuInit::Ptr meas = nullptr;

    while (ros::ok())
    {
        if (tracker.exit_flag)
            break;

        ros::spinOnce();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "slam_run");
    ros::NodeHandle nh;
    Odometry odom(nh);
    odom.run();

    return 0;
}