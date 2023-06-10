#include "lio.hpp"
#include <chrono>
#include <functional>
#include "geometry_msgs/Vector3.h"
#include <cstdlib>
#include "std_msgs/Header.h"

namespace
{
    using hr_clock = std::chrono::high_resolution_clock;

    template <typename Func, typename... Args>
    double duration_meas(Func func, Args &&...args)
    {
        auto start_time = hr_clock::now();
        func(std::forward<Args>(args)...);
        auto end_time = hr_clock::now();

        return std::chrono::duration<double>(end_time - start_time).count();
    }
}

Sophus::SE3d LIO::get_transform(const std::string &source_frame, const std::string &target_frame)
{
    static tf2_ros::Buffer tf_buffer;
    static tf2_ros::TransformListener tf_listener(tf_buffer);

    geometry_msgs::TransformStamped transform;
    bool transform_found = false;

    while (ros::ok())
    {
        try
        {
            transform = tf_buffer.lookupTransform(target_frame, source_frame, ros::Time(0));
            transform_found = true;
            ROS_INFO("Transform Found!");
            break;
        }
        catch (tf2::TransformException &ex)
        {
            ROS_WARN("%s", ex.what());
        }

        // Sleep for a short duration before trying again
        ros::Duration(0.1).sleep();
    }

    if (!transform_found)
    {
        ROS_ERROR("Failed to obtain transform between %s and %s", source_frame.c_str(), target_frame.c_str());
    }

    Eigen::Quaterniond rotation(transform.transform.rotation.w,
                                transform.transform.rotation.x,
                                transform.transform.rotation.y,
                                transform.transform.rotation.z);
    Eigen::Vector3d translation(transform.transform.translation.x,
                                transform.transform.translation.y,
                                transform.transform.translation.z);

    Sophus::SE3d se3(rotation, translation);
    return se3;
}

void LIO::initialize_publishers(ros::NodeHandle &nh)
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

void LIO::publish_transform(
    const std::string &child_frame, const Sophus::SE3d &transform,
    tf2_ros::StaticTransformBroadcaster &br)
{
    geometry_msgs::TransformStamped baselink_msg;
    Eigen::Quaterniond q(transform.rotationMatrix());
    Eigen::Vector3d t(transform.translation());

    baselink_msg.header.stamp = ros::Time::now();
    baselink_msg.transform.translation.x = t.x();
    baselink_msg.transform.translation.y = t.y();
    baselink_msg.transform.translation.z = t.z();
    baselink_msg.transform.rotation.w = q.w();
    baselink_msg.transform.rotation.x = q.x();
    baselink_msg.transform.rotation.y = q.y();
    baselink_msg.transform.rotation.z = q.z();

    baselink_msg.header.frame_id = "base_link";
    baselink_msg.child_frame_id = child_frame;
    br.sendTransform(baselink_msg);
}

void LIO::initialize_transforms(std::string &loc)
{
    YAML::Node node = YAML::LoadFile(params_config);
    static tf2_ros::StaticTransformBroadcaster br;
    Eigen::Matrix3d rot_mat;
    Eigen::Vector3d trans;

    use_frames = node["sensor_locations"]["use_locs"].as<bool>();
    if (use_frames)
    {
        // lidar
        utils::vec_2_matrices(
            extract_rot_mat_values(node["sensor_locations"]["lidar_R"]),
            extract_rot_mat_values(node["sensor_locations"]["lidar_T"]),
            rot_mat, trans);

        Sophus::SE3d lidar_base = Sophus::SE3d(rot_mat, trans);
        publish_transform("lidar_frame", lidar_base, br);

        // imu
        utils::vec_2_matrices(
            extract_rot_mat_values(node["sensor_locations"]["imu_R"]),
            extract_rot_mat_values(node["sensor_locations"]["imu_T"]),
            rot_mat, trans);

        std::cout << rot_mat.matrix() << std::endl;
        Sophus::SE3d imu_base = Sophus::SE3d(rot_mat, trans);
        publish_transform("imu_frame", imu_base, br);

        // other stuff calculate transform
        Sophus::SE3d L_I = get_transform("lidar_frame", "imu_frame");
        ekf->set_extrinsics(L_I.rotationMatrix(), L_I.translation());
    }
    else
    {
        utils::vec_2_matrices(
            extract_rot_mat_values(node["kalman_filter_params"]["extrinsic_R"]),
            extract_rot_mat_values(node["kalman_filter_params"]["extrinsic_T"]),
            rot_mat, trans);
        ekf->set_extrinsics(rot_mat, trans);
    }

    if (child_frame != "base_link")
    {
        // Create the desired transform
        Sophus::SE3d transform(
            Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0),
            Eigen::Vector3d(0.0, 0.0, 0.0));

        // Publish the transform
        publish_transform(child_frame, transform, br);
    }
}
void LIO::imu_callback(const sensor_msgs::Imu::ConstPtr &msg)
{
    if (!msg)
    {
        ROS_WARN("Invalid input data. Skipping processing.");
        return;
    }

    imu_ptr->process_data(msg);
}

void LIO::lidar_callback(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    if (!msg)
    {
        ROS_WARN("Invalid Lidar data. Skipping processing.");
        return;
    }

    lidar_ptr->initialize(msg);

    // split frames when necessary.
    lidar_ptr->process_frame();
}

bool LIO::sync_packages(frame::LidarImuInit::Ptr &meas)
{
    if (lidar_ptr->buffer_empty() || imu_ptr->buffer_empty())
        return false;

    // Collate all lidar and imu data.
    meas = std::make_shared<frame::LidarImuInit>();

    meas->lidar_buffer = lidar_ptr->get_lidar_buffer_front();
    lidar_ptr->pop();

    // update data for this session
    meas->lidar_beg_time = meas->lidar_buffer.front().timestamp;
    meas->lidar_last_time = meas->lidar_buffer.back().timestamp;

    // collect all imu data to end of current timestamp
    if (imu_ptr->enabled)
        imu_ptr->collect_data(meas->lidar_last_time, meas->imu_buffer);

    imu_ptr->process_package(meas);
    meas->merge_sensor_data();
    return true;
}

void LIO::run()
{
    ros::Rate rate(10);
    while (ros::ok())
    {
        frame::LidarImuInit::Ptr meas = nullptr;

        if (tracker.exit_flag)
            continue;

        if (!sync_packages(meas))
            continue;

        if (tracker.flag_reset)
        {
            ROS_WARN("[WARN] Resetting robag playback.");
            imu_ptr->reset();
            tracker.flag_reset = false;

            continue;
        }

        process_frame(meas);
        rate.sleep();
    }
}

void LIO::process_frame(const frame::LidarImuInit::Ptr &meas)
{
    // looping through all available data in current measurement
    int idx = 0;

    if (!ekf->orientation_init)
        ekf->initialize_orientation(imu_ptr->enabled, imu_ptr->mean_acc);

    ROS_INFO("Processing Frame: %d\n", (ekf->frame_num + 1));
    for (const auto &data : meas->data)
    {
        if (tracker.first_frame)
        {
            tracker.first_frame = false;
            tracker.time_update_last = data.timestamp;
            tracker.time_predict_last_const = data.timestamp;
        }
        ROS_INFO("Pre KF run");
        if (kalman_filter_run(data, imu_ptr->enabled))
            publish(idx);
        ROS_INFO("Post publish or not");
        idx++;
        ekf->pose_trail_tracker();
    }

    ekf->frame_num++;
}

void LIO::imu_runner(const frame::SensorData &data, double sf)
{
    const auto &val = data.imu_data;

    // start using imu af
    if (ekf->use_imu_as_input)
    {
        // read values
        ekf->update_imu_reading(val->gyro, val->acc, sf);

        double dt_cov = data.timestamp - tracker.time_update_last;
        if (dt_cov > 0.0)
        {
            tracker.time_update_last = data.timestamp;
            tracker.propagate_time += duration_meas(
                [&]()
                { ekf->predict(dt_cov, false, true); });
            imu_propagated = true;
        }

        // make prediction
        double dt = data.timestamp - tracker.time_predict_last_const;
        ekf->predict(dt, true, false);
        tracker.time_predict_last_const = data.timestamp;
        return;
    }

    // make prediction -> this estimates gyro and acc
    double dt = data.timestamp - tracker.time_predict_last_const;
    ekf->predict(dt, true, false);
    tracker.time_predict_last_const = data.timestamp;

    // update imu values
    ekf->update_imu_reading(val->gyro, val->acc);

    // propagate state based on predicted imu data
    double dt_cov = data.timestamp - tracker.time_update_last;
    if (dt_cov > 0.0)
    {
        tracker.time_update_last = data.timestamp;
        tracker.propagate_time += duration_meas(
            [&]()
            { ekf->predict(dt_cov, false, true); });

        // update the imu values.
        tracker.solve_time += duration_meas(
            [&]()
            { ekf->update_h_model_IMU_output(sf); });

        imu_propagated = true;
    }
}

bool LIO::lidar_runner(const frame::SensorData &data)
{
    // if we have lidar data then we need to do the other checks.
    if (!ekf->init_map)
    {
        // initialize the map
        if (ekf->initialize_map(data))
        {
            ROS_INFO("[INFO] ........ Map Initialized ........");
            return true;
        }

        return false;
    }

    double dt = data.timestamp - tracker.time_predict_last_const;
    if (!imu_propagated)
    {
        double dt_cov = data.timestamp - tracker.time_update_last;
        if (dt_cov > 0.0)
        {
            tracker.propagate_time += duration_meas(
                [&]()
                { ekf->predict(dt_cov, false, true); });

            tracker.time_update_last = data.timestamp;
        }
    }

    // make a state prediction
    ekf->predict(dt, true, false);
    tracker.time_predict_last_const = data.timestamp;
    if (!ekf->update_h_model(data))
        return false;

    return true;
}

bool LIO::kalman_filter_run(const frame::SensorData &data, bool imu_enabled)
{
    bool converged = false;

    if (data.m_type == frame::SensorData::Type::Imu && imu_enabled)
        imu_runner(data);
    else
        imu_propagated = false;

    if (data.m_type == frame::SensorData::Type::Lidar && run_lidar)
        converged = lidar_runner(data);

    return converged;
}

void LIO::publish(int curr_idx)
{
    std_msgs::Header msg;
    msg.stamp = ros::Time::now();

    // publish path
    const auto pose = ekf->icp_ptr->poses_().back();

    // Convert from Eigen to ROS types
    const Eigen::Vector3d t_curr = pose.translation();
    const Eigen::Quaterniond q_curr = pose.unit_quaternion();

    // broadcast tf which is the imu frame
    geometry_msgs::TransformStamped transform_msg;
    transform_msg.header.stamp = msg.stamp;
    transform_msg.header.frame_id = odom_frame;
    transform_msg.child_frame_id = child_frame;
    transform_msg.transform.rotation.x = q_curr.x();
    transform_msg.transform.rotation.y = q_curr.y();
    transform_msg.transform.rotation.z = q_curr.z();
    transform_msg.transform.rotation.w = q_curr.w();
    transform_msg.transform.translation.x = t_curr.x();
    transform_msg.transform.translation.y = t_curr.y();
    transform_msg.transform.translation.z = t_curr.z();
    tf_broadcaster.sendTransform(transform_msg);

    // publish odometry msg
    nav_msgs::Odometry odom_msg;
    odom_msg.header.stamp = msg.stamp;
    odom_msg.header.frame_id = odom_frame;
    odom_msg.child_frame_id = child_frame;
    odom_msg.pose.pose.orientation.x = q_curr.x();
    odom_msg.pose.pose.orientation.y = q_curr.y();
    odom_msg.pose.pose.orientation.z = q_curr.z();
    odom_msg.pose.pose.orientation.w = q_curr.w();
    odom_msg.pose.pose.position.x = t_curr.x();
    odom_msg.pose.pose.position.y = t_curr.y();
    odom_msg.pose.pose.position.z = t_curr.z();
    odom_publisher.publish(odom_msg);

    // publish imu frame
    geometry_msgs::PoseStamped pose_msg;
    pose_msg.pose = odom_msg.pose.pose;
    pose_msg.header = odom_msg.header;
    {
        path_msgs.poses.clear();
        const std::vector<Sophus::SE3d> &ekfPoses = ekf->icp_ptr->poses_();
        for (const auto &pose_ : ekfPoses)
        {
            // Create a new PoseStamped message
            geometry_msgs::PoseStamped poseStamped;
            poseStamped.header = odom_msg.header;

            // Convert the Sophus::SE3d pose to geometry_msgs::Pose
            Eigen::Quaterniond quat(pose_.rotationMatrix());
            poseStamped.pose.orientation.x = quat.x();
            poseStamped.pose.orientation.y = quat.y();
            poseStamped.pose.orientation.z = quat.z();
            poseStamped.pose.orientation.w = quat.w();
            poseStamped.pose.position.x = pose_.translation().x();
            poseStamped.pose.position.y = pose_.translation().y();
            poseStamped.pose.position.z = pose_.translation().z();

            // Add the converted pose to the path_msgs.poses vector
            path_msgs.poses.push_back(poseStamped);
        }

        traj_publisher.publish(path_msgs);
    }

    // publish points in current timestamp.
    if (use_frames)
    {
        std_msgs::Header pf_frame = msg;
        pf_frame.frame_id = child_frame;
        frame_publisher.publish(utils::eigen_to_pc(ekf->lidar_point_cloud, pf_frame));
    }

    // publish current processed point clouds.
    if (curr_idx % update_freq == 0 || curr_idx == 0)
    {
        std_msgs::Header local_map_header = msg;
        local_map_header.frame_id = odom_frame;
        local_map_publisher.publish(utils::eigen_to_pc(ekf->icp_ptr->local_map_(), local_map_header));
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "slam_run");
    ros::NodeHandle nh("~");
    LIO odom(nh);

    ros::spin();

    return 0;
}
