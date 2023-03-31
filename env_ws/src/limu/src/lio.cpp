#include "lio.hpp"
#include <chrono>
#include <functional>
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

void LIO::initialize_publishers(ros::NodeHandle &nh)
{
    // pointcloud publishers
    odom_publisher = nh.advertise<nav_msgs::Odometry>("LIO", queue_size);
    frame_publisher = nh.advertise<sensor_msgs::PointCloud2>("frame", queue_size);
    kpoints_publisher = nh.advertise<sensor_msgs::PointCloud2>("keypoints", queue_size);
    local_map_publisher = nh.advertise<sensor_msgs::PointCloud2>("local_map", queue_size);

    // trajectory publisher
    path_msgs.header.frame_id = odom_frame;
    traj_publisher = nh.advertise<nav_msgs::Path>("trajectory", queue_size);
}

void LIO::setup_ekf(ros::NodeHandle &nh)
{
    odometry::PARAMETERS::Ptr p(new odometry::PARAMETERS);
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

    double init_ori_trail_noise;
    nh.param<double>("init_ori_trail_noise", init_ori_trail_noise, 3.1622776);
    nh.param<double>("init_ori_noise", p->init_bga_noise, 0.01 * init_ori_trail_noise);

    // from imu spec sheet
    nh.param<double>("satu_gyro", p->satu_gyro, 35.0);
    nh.param<double>("satu_acc", p->satu_acc, 3.0);

    ekf = std::make_unique<odometry::EKF>(lidar_ptr->config, p);
}

void LIO::imu_callback(const sensor_msgs::Imu::ConstPtr &msg)
{
    imu_ptr->process_data(msg);

    std::unique_lock<std::mutex> lock(data_mutex);
    imu_ptr->imu_time_compensation(tracker.time_diff_imu_wrt_lidar, tracker.time_lag_IMU_wtr_lidar);
    lock.unlock();

    imu_ptr->update_buffer();
}

void LIO::lidar_callback(const sensor_msgs::PointCloud2::ConstPtr &msg)
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

bool LIO::lidar_process(frame::LidarImuInit::Ptr &meas)
{
    // For testing lidar process
    if (lidar_ptr->buffer_empty())
        return false;

    const auto &processed_frame = lidar_ptr->get_split_lidar_buffer_front();

    if (processed_frame->points.size() < 1)
    {
        ROS_WARN("Not enough points in pointcloud...");
        lidar_ptr->pop();
        return false;
    }

    // load information into frame
    meas = std::make_shared<frame::LidarImuInit>();
    meas->lidar_beg_time = lidar_ptr->get_pc_time_s();
    meas->lidar_last_time = (lidar_ptr->get_pc_time_ms() + lidar_ptr->curr_acc_segment_time()) / double(1000);
    meas->processed_frame = processed_frame;
    lidar_ptr->pop();

    return true;
}

bool LIO::imu_process(frame::LidarImuInit::Ptr &meas)
{
    if (!tracker.init_map)
        meas->imu_buffer.emplace_back(imu_ptr->imu_last_ptr);

    meas->imu_buffer.shrink_to_fit();
    double imu_time = imu_ptr->get_front_time();
    while (!imu_ptr->buffer_empty() && imu_time < meas->lidar_last_time)
    {
        imu_time = imu_ptr->get_front_time();
        if (imu_time > meas->lidar_last_time)
            break;

        meas->imu_buffer.emplace_back(imu_ptr->buffer_front());
        imu_ptr->imu_last = imu_ptr->imu_next;
        imu_ptr->imu_last_ptr = imu_ptr->buffer_front();
        imu_ptr->imu_next = *(imu_ptr->imu_last_ptr);
        imu_ptr->pop();
    }

    return true;
}

bool LIO::sync_packages(frame::LidarImuInit::Ptr &meas)
{
    // using lidar information only
    if (!imu_ptr->enabled)
        return lidar_process(meas);

    if (lidar_ptr->buffer_empty() || imu_ptr->buffer_empty())
        return false;

    if (!tracker.lidar_pushed)
    {
        if (!lidar_process(meas))
            return false;

        tracker.lidar_pushed = true;
    }

    if (imu_ptr->get_prev_timestamp() < lidar_ptr->lidar_end_time)
        return false;

    /*... Push imu data ...*/
    if (imu_ptr->need_init || !tracker.init_map)
    {
        imu_process(meas);
    }

    tracker.lidar_pushed = false;
    return true;
}

void LIO::publish_init_map(const utils::Vec3dVector &map_points)
{
    return;
}
void LIO::run()
{
    ros::Rate rate(5000);
    frame::LidarImuInit::Ptr meas = nullptr;

    while (ros::ok())
    {
        if (tracker.exit_flag)
            break;

        ros::spinOnce();
        if (!sync_packages(meas))
        {
            rate.sleep();
            continue;
        }

        if (tracker.flag_first_scan)
        {
            tracker.first_lidar_time = meas->lidar_beg_time;
            tracker.flag_first_scan = false;
            ROS_INFO("[INFO] First Lidar Time initializzed");
        }

        if (tracker.flag_reset)
        {
            ROS_WARN("[WARN] Resetting robag playback.");
            imu_ptr->reset();
            tracker.flag_reset = false;
            meas.reset();
            continue;
        }

        if (!ekf->map_def())
        {
            ekf->h_model_input(data, meas);
            ROS_INFO("[INFO] ........ Map Initialized ........");

            if (imu_ptr->enabled)
            {
                while (meas->lidar_beg_time < imu_ptr->imu_next.header.stamp.toSec())
                    imu_ptr->recycle();
            }
            else
            {
                (ekf->inp_state.grav << 0.0, 0.0, -gravity).finished();
                (ekf->out_state.grav << 0.0, 0.0, -gravity).finished();
                (ekf->out_state.imu_acc << 0.0, 0.0, gravity).finished();
            }

            // publish_init_map(ekf->icp_ptr->local_map_());
            meas.reset();
            continue;
        }

        bool enabled = imu_ptr->enabled;

        /** Data accumulation begins here **/
        if (!enabled && !tracker.data_accum_start && ekf->get_lidar_pos().norm() > 0.05)
        {
            ROS_INFO("[INFO] Data Accumulation for Robot Initialization Begins.");
            tracker.data_accum_start = true;
            tracker.move_start_time = lidar_ptr->lidar_end_time;
        }

        /*** L_I initialization ***/
        if (!enabled && !tracker.data_accum_finished && tracker.data_accum_start)
        {
            continue;
        }

        // const auto &point = meas->processed_frame->points[0];
        // current_time = point.curvature / 1000.0 + meas->lidar_beg_time;
        // if (!ekf->use_imu_as_input)
        // {

        //     bool propagated = false;

        //     if (tracker.first_frame)
        //     {
        //         if (enabled)
        //         {
        //             // align imu reading with current lidar measurement
        //             while (current_time > imu_ptr->imu_next.header.stamp.toSec())
        //                 imu_ptr->recycle();
        //         }

        //         tracker.first_frame = false;
        //         time_update_last = current_time;
        //         time_predict_last_const = current_time;
        //     }

        //     if (enabled)
        //     {
        //         while (current_time > imu_ptr->imu_next.header.stamp.toSec())
        //         {
        //             const auto &imu_next = imu_ptr->imu_next;
        //             ekf->update_imu_reading(imu_next.angular_velocity, imu_next.linear_acceleration);

        //             // update the covariance
        //             imu_ptr->recycle();

        //             double dt = imu_ptr->imu_next.header.stamp.toSec() - time_predict_last_const;
        //             ekf->out_state.predict(ekf->zero, ekf->zero, dt, true, false);
        //             time_predict_last_const = imu_ptr->imu_last.header.stamp.toSec();

        //             // covariance propagation
        //             double dt_cov = time_predict_last_const - time_update_last;
        //             if (dt_cov > 0.0)
        //             {
        //                 propagated = true;
        //                 time_update_last = time_predict_last_const;
        //                 double pg_time = duration_meas(
        //                     [&]()
        //                     {
        //                         ekf->out_state.predict(ekf->zero, ekf->zero, dt_cov, false, true);
        //                     });
        //                 propagate_time += pg_time;

        //                 // update model based on imu readings
        //                 double s_time = duration_meas(
        //                     [&]()
        //                     {
        //                         ekf->update_h_model_IMU_output(meas);
        //                     });
        //                 solve_time += s_time;
        //             }
        //         }
        //     }

        //     double dt = current_time - time_predict_last_const;
        //     if (!propagated)
        //     {
        //         double dt_cov = time_predict_last_const - time_update_last;
        //         if (dt_cov > 0.0)
        //         {
        //             time_update_last = time_predict_last_const;
        //             double pg_time = duration_meas(
        //                 [&]()
        //                 {
        //                     ekf->out_state.predict(ekf->zero, ekf->zero, dt_cov, false, true);
        //                 });
        //             propagate_time += pg_time;
        //         }
        //     }

        //     ekf->out_state.predict(ekf->zero, ekf->zero, dt, true, false);

        //     if (!ekf->update_h_model_output(meas))
        //     {
        //         meas.reset();
        //         continue;
        //     }
        // }
        // else
        // {
        //     if (tracker.first_frame)
        //     {
        //         while (current_time > imu_ptr->imu_next.header.stamp.toSec())
        //             imu_ptr->recycle();

        //         tracker.first_frame = false;
        //         time_update_last = current_time;
        //         time_predict_last_const = current_time;
        //     }

        //     bool propagated = false;
        //     while (current_time > imu_ptr->imu_next.header.stamp.toSec())
        //     {
        //         const auto &imu_last = imu_ptr->imu_last;
        //         double sf = gravity / meas->get_mean_acc_norm();
        //         ekf->update_imu_reading(imu_last.angular_velocity, imu_last.linear_acceleration, sf);

        //         // scale with graivty
        //         double dt = imu_last.header.stamp.toSec() - time_predict_last_const;
        //         double dt_cov = time_predict_last_const - time_update_last;
        //         if (dt_cov > 0.0)
        //         {
        //             propagated = true;
        //             double pg_time = duration_meas(
        //                 [&]()
        //                 {
        //                     ekf->inp_state.predict(ekf->ang_vel_read, ekf->acc_vel_read, dt_cov, false, true);
        //                 });
        //             propagate_time += pg_time;
        //         }

        //         ekf->inp_state.predict(ekf->ang_vel_read, ekf->acc_vel_read, dt, true, false);
        //         time_update_last = time_predict_last_const;
        //     }

        //     double dt = current_time - time_predict_last_const;
        //     if (!propagated)
        //     {
        //         double dt_cov = current_time - time_update_last;
        //         if (dt_cov > 0.0)
        //         {
        //             time_update_last = time_predict_last_const;
        //             double pg_time = duration_meas(
        //                 [&]()
        //                 {
        //                     ekf->inp_state.predict(ekf->ang_vel_read, ekf->acc_vel_read, dt_cov, false, true);
        //                 });
        //             propagate_time += pg_time;
        //         }
        //     }
        //     ekf->inp_state.predict(ekf->ang_vel_read, ekf->acc_vel_read, dt, true, false);

        //     if (!ekf->update_h_model_input(meas))
        //     {
        //         meas.reset();
        //         continue;
        //     }
        // }

        // // update the map -> contains position and translation information
        // auto start_time = hr_clock::now();
        // const auto tup_val = ekf->update_map();
        // auto end_time = hr_clock::now();
        // update_time += std::chrono::duration<double>(end_time - start_time).count();

        // PUBLISHING ODOMETRY TO BE DONE HERE
        meas.reset();
    }
}

void LIO::publish_point_cloud(
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
    LIO odom(nh);
    odom.run();

    return 0;
}
