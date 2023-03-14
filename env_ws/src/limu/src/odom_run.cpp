#include "odom_run.hpp"

void Odometry::initialize_publishers(ros::NodeHandle &nh)
{
    // pointcloud publishers
    odom_publisher = nh.advertise<nav_msgs::Odometry>("odometry", queue_size);
    frame_publisher = nh.advertise<sensor_msgs::PointCloud2>("frame", queue_size);
    kpoints_publisher = nh.advertise<sensor_msgs::PointCloud2>("keypoints", queue_size);
    local_map_publisher = nh.advertise<sensor_msgs::PointCloud2>("local_map", queue_size);

    // trajectory publisher
    path_msgs.header.frame_id = odom_frame;
    traj_publisher = nh.advertise<nav_msgs::Path>("trajectory", queue_size);
}

void Odometry::setup_ekf(ros::NodeHandle &nh)
{
    kalman::EKF_PARAMETERS::Ptr p(new kalman::EKF_PARAMETERS);
    nh.param<int>("lidar_pose_trail", p->lidar_pose_trail, 20);
    nh.param<double>("noise_scale", p->noise_scale, 100);
    nh.param<double>("init_pos_noise", p->init_pos_noise, 1e-5);
    nh.param<double>("init_vel_noise", p->init_vel_noise, 0.1);
    nh.param<double>("init_bga_noise", p->init_bga_noise, 1e-3);
    nh.param<double>("init_baa_noise", p->init_baa_noise, 1e-6);
    nh.param<double>("init_bat_noise", p->init_bat_noise, 1e-5);
    // standard deviation from spec sheet
    nh.param<double>("acc_process_noise", p->acc_process_noise, 0.03);
    nh.param<double>("gyro_process_noise", p->gyro_process_noise, 0.00017);
    nh.param<double>("acc_process_noise_rev", p->acc_process_noise_rev, 0.1);
    nh.param<double>("gyro_process_noise_rev", p->gyro_process_noise_rev, 0.1);
    nh.param<double>("init_pos_trail_noise", p->init_pos_trail_noise, 100);
    nh.param<double>("init_ori_trail_noise", p->init_ori_trail_noise, 3.1622776);
    nh.param<double>("init_lidar_imu_time_noise", p->init_lidar_imu_time_noise, 1e-5);

    nh.param<double>("init_ori_noise", p->init_bga_noise, 0.01 * p->init_ori_trail_noise);

    ekf = std::make_unique<kalman::EKF>(p);
}

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
        const auto &processed_frame = lidar_ptr->get_lidar_buffer_front();

        if (processed_frame->points.size() <= 1)
        {
            ROS_WARN("Too few input point cloud!\n");
            lidar_ptr->pop();
            return false;
        }

        // load information into frame
        meas = std::make_shared<frame::LidarImuInit>();
        meas->processed_frame = processed_frame;

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
    // estimate lidar odometry - deskewed, keypoints, pose
    const auto icp_output = icp_ptr->register_frame(*(meas->processed_frame), meas->time_buffer);
    const auto &deskewed = std::get<0>(icp_output);
    const auto &key_points = std::get<1>(icp_output);
    const auto &pose = std::get<2>(icp_output);

    std::cout << "Current pose:" << std::endl;
    std::cout << pose.matrix() << std::endl;

    const utils::Vec3d translation = pose.translation();
    const Eigen::Quaterniond quat = pose.unit_quaternion();

    const auto curr_time = ros::Time::now();
    // broadcast current pose estimate
    lidar_ptr->set_current_pose_nav(translation, quat, curr_time, odom_frame, child_frame);
    tf_broadcaster.sendTransform(lidar_ptr->current_pose);
    odom_publisher.publish(lidar_ptr->odom_msg);

    // path
    geometry_msgs::PoseStamped pose_msg;
    pose_msg.pose = lidar_ptr->odom_msg.pose.pose;
    pose_msg.header = lidar_ptr->odom_msg.header;
    path_msgs.poses.push_back(pose_msg);
    traj_publisher.publish(path_msgs);

    // For debugging purposes
    publish_point_cloud(frame_publisher, curr_time, child_frame, deskewed);
    publish_point_cloud(kpoints_publisher, curr_time, child_frame, key_points);
}

void Odometry::kalman_filter_process(frame::LidarImuInit::Ptr &meas)
{
    if (meas->deskewed.empty())
    {
        tracker.first_lidar_time = meas->lidar_beg_time;
        imu_ptr->first_lidar_time = tracker.first_lidar_time;
        ROS_WARN("Cannot run the Initialization scheme yet.");
    }

    if (imu_ptr->enabled)
        imu_ptr->initialize_propagate_undistort(ekf, meas);
    else
    {
        // undistort points using lidar stuff
        meas->deskewed = icp_ptr->deskew_scan(*(meas->processed_frame), meas->time_buffer);
        ROS_INFO("Motion Compensated with Lidar");
    }
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
        if (!lidar_process(meas))
        {
            rate.sleep();
            continue;
        }

        // packages have been synced.
        if (tracker.reset_flag)
        {
            ROS_WARN("Resetting playback.");
            tracker.reset_flag = false;
            continue;
        }

        // estimate lidar odometry
        estimate_lidar_odometry(meas);

        // reset measurement
        meas.reset();
    }
}

void Odometry::publish_point_cloud(
    ros::Publisher &pub, const ros::Time &time,
    const std::string &frame_id, const utils::Vec3dVector &points)
{
    sensor_msgs::PointCloud2 cloud_msg;
    {
        cloud_msg.header.stamp = time;
        cloud_msg.header.frame_id = frame_id;

        const size_t num_points = points.size();

        // Set the fields
        cloud_msg.fields.resize(3);
        cloud_msg.fields[0].name = "x";
        cloud_msg.fields[0].offset = 0;
        cloud_msg.fields[0].datatype = sensor_msgs::PointField::FLOAT32;
        cloud_msg.fields[0].count = 1;
        cloud_msg.fields[1].name = "y";
        cloud_msg.fields[1].offset = 4;
        cloud_msg.fields[1].datatype = sensor_msgs::PointField::FLOAT32;
        cloud_msg.fields[1].count = 1;
        cloud_msg.fields[2].name = "z";
        cloud_msg.fields[2].offset = 8;
        cloud_msg.fields[2].datatype = sensor_msgs::PointField::FLOAT32;
        cloud_msg.fields[2].count = 1;

        // Set the height and width of the point cloud
        cloud_msg.height = 1;
        cloud_msg.width = num_points;

        // Set the point step (size of each point)
        cloud_msg.point_step = 12; // 3 floats (x,y,z) * 4 bytes/float = 12 bytes

        // Set the data size
        cloud_msg.data.resize(num_points * cloud_msg.point_step);
    }
    // Get a pointer to the data
    float *data_ptr = reinterpret_cast<float *>(&cloud_msg.data[0]);

    // Copy the data from the input vector
    for (const auto &point : points)
    {
        *data_ptr++ = point.x();
        *data_ptr++ = point.y();
        *data_ptr++ = point.z();
    }

    // Set the is_dense flag (assume all points are valid)
    cloud_msg.is_dense = true;

    pub.publish(cloud_msg);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "slam_run");
    ros::NodeHandle nh;
    Odometry odom(nh);
    odom.run();

    return 0;
}
