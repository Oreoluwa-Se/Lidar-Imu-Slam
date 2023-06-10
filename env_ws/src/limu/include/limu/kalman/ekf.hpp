#ifndef ODOM_EKF_HPP
#define ODOM_EKF_HPP

#include "limu/sensors/sync_frame.hpp"
#include "limu/sensors/imu/frame.hpp"
#include "limu/sensors/lidar/icp.hpp"
#include <Eigen/SparseCore>
#include <Eigen/Cholesky>
#include "common.hpp"
#include "states.hpp"
#include "helper.hpp"

namespace odometry
{
    using MapPoint = utils::Point;
    using MapPointPtr = MapPoint::Ptr;
    using PointsVector = std::vector<MapPoint>;
    using PointsPtrVector = std::vector<MapPointPtr>;
    using CorrespondenceTuple = std::tuple<std::vector<int>, PointsVector>;
    using PointsVectorTuple = std::tuple<PointsVector, PointsVector>;

    template <typename T>
    struct dyn_share_modified
    {
        bool valid, converge;
        T M_noise;
        Eigen::Matrix<T, Eigen::Dynamic, 1> z;
        Eigen::Matrix<T, Eigen::Dynamic, 1> weight;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> h_x;
        Eigen::Matrix<T, 6, 1> z_IMU;
        Eigen::Matrix<T, 6, 1> R_IMU;

        std::mutex mutex;
        bool satu_check[6];
    };

    class EKF
    {
    public:
        typedef std::unique_ptr<EKF> Ptr;

        explicit EKF(const std::string &params_location)
            : ang_vel_read(Eigen::Vector3d::Zero()),
              acc_vel_read(Eigen::Vector3d::Zero()),
              orientation_init(false), init_map(false),
              frame_num(0), sigma(0.0), dx_norm(0.0),
              threshold(0.0), key_tracker(0), all_tracker(0),
              max_imu_iter(0), use_outliers(false)
        {
            initialize_params(params_location);

            // initialize classes
            state = std::make_shared<State>(params_location);
            icp_ptr = std::make_shared<lidar::KissICP>(params_location);
        }

        void initialize_params(const std::string &loc)
        {
            YAML::Node node = YAML::LoadFile(loc);

            use_imu_as_input = node["common"]["use_imu_as_input"].as<bool>();
            verbose = node["common"]["verbose"].as<bool>();
            print_matrices = node["common"]["print_kf_matrices"].as<bool>();
            grav = node["common"]["gravity"].as<double>();
            frames_to_keep = node["common"]["pose_trail"].as<int>();
            pose_storage = node["common"]["pose_storage_loc"].as<std::string>();
            all_pose_storage = node["common"]["all_pose_storage_loc"].as<std::string>();
            store_all = node["common"]["store_all_coordinates"].as<bool>();
            use_outliers = node["common"]["use_outliers"].as<bool>();

            est_extrinsic = node["kalman_filter_params"]["est_extrinsic"].as<bool>();
            laser_point_cov = node["kalman_filter_params"]["lidar_measurement_noise"].as<double>();
            imu_acc_meas_cov = node["kalman_filter_params"]["imu_acc_measurement_noise"].as<double>();
            imu_gyro_meas_cov = node["kalman_filter_params"]["imu_gyro_measurement_noise"].as<double>();
            satu_acc = node["kalman_filter_params"]["measurement_covariance"]["satu_acc"].as<double>();
            satu_gyro = node["kalman_filter_params"]["measurement_covariance"]["satu_gyro"].as<double>();
            max_imu_iter = node["kalman_filter_params"]["max_imu_iterations"].as<int>();

            num_corresp = node["icp_params"]["num_correspondences"].as<int>();
            max_icp_iter = node["icp_params"]["max_iteration"].as<int>();
            estimation_threshold = node["icp_params"]["estimation_threshold"].as<double>();

            tracker = PointHelper(est_extrinsic);
        }

        void reset_parameters()
        {
            // Holds information about initial guess for all readings in current frames
            tracker.update_pose(state);
            sigma = icp_ptr->ICP_setup();
            threshold = sigma / 3.0;
        }

        void set_extrinsics(const Eigen::Matrix3d &Lidar_R_wrt_IMU, const Eigen::Vector3d &Lidar_T_wrt_IMU)
        {
            tracker.Lidar_R_wrt_IMU = Lidar_R_wrt_IMU;
            tracker.Lidar_T_wrt_IMU = Lidar_T_wrt_IMU;

            // push into respective states
            state->nominal_state.offset_R_L_I = utils::rmat2quat(Lidar_R_wrt_IMU);
            state->nominal_state.offset_T_L_I = Lidar_T_wrt_IMU;
        }

        void update_imu_reading(const Eigen::Vector3d &imu_gyro, const Eigen::Vector3d &imu_acc, double sf = 1)
        {
            ang_vel_read = imu_gyro;
            acc_vel_read = sf * imu_acc;
        }

        bool has_moved()
        {
            return icp_ptr->has_moved();
        }

        void initialize_orientation(bool imu_enabled, Eigen::Vector3d mean_acc)
        {
            (state->nominal_state.grav << 0.0, 0.0, -grav).finished();
            state->nominal_state.imu_acc = -1.0 * state->nominal_state.grav;

            state->initialize_orientation(mean_acc);
            orientation_init = true;
        }

        bool initialize_map(const frame::SensorData &data);
        bool update_h_model(const frame::SensorData &data);
        bool update_h_model_IMU_output(double gravity_sf = 1.0);
        void update_map(PointsPtrVector &points);
        void predict(double dt, bool predict_state, bool prop_cov);
        void pose_trail_tracker();

    public:
        // attributes
        std::vector<Sophus::SE3d> key_poses, all_pose;
        bool est_extrinsic, use_imu_as_input, orientation_init, init_map;
        Eigen::Vector3d ang_vel_read, acc_vel_read;
        Eigen::Vector3d zero = Eigen::Vector3d::Zero();
        PointsPtrVector lidar_point_cloud;
        lidar::KissICP::Ptr icp_ptr;
        State::Ptr state = nullptr;
        int frame_num, num_corresp;
        double grav;

    private:
        struct PointHelper
        {
            PointHelper() {}
            explicit PointHelper(bool est) : est_extrinsic(est)
            {
                matched.resize(2);
            }

            void initial_transform(const Sophus::SE3d &init_guess)
            {
                // points in imu coordinates
                points_imu_coord = orig_point_lidar;
                utils::transform_points(T_L_I, points_imu_coord);

                // convert base point to imu-world coordinates
                points_imu_world_coord = points_imu_coord;
                utils::transform_points(init_guess, points_imu_world_coord);
                T_I_W = init_guess;
                T_total = T_I_W * T_L_I;
            }

            void transform_points(State::Ptr &state)
            {
                // update pose
                update_pose(state);

                /*................. Get new imu frame locations .......... */
                points_imu_coord = orig_point_lidar;
                utils::transform_points(T_L_I, points_imu_coord);

                /*................. Get new world locations .......... */
                points_imu_world_coord = points_imu_coord;
                utils::transform_points(T_I_W, points_imu_world_coord);
            }

            void clear()
            {
                orig_point_lidar.clear();
                points_imu_world_coord.clear();
                points_imu_coord.clear();
                map_points.clear();
                matched.clear();
                first_matched = false;
            }

            void update_init(State::Ptr &state)
            {
                T_I_W_init = state->get_IMU_pose();
                if (est_extrinsic)
                    T_L_I_init = state->get_L_I_pose();
                else
                    T_L_I_init = Sophus::SE3d(Eigen::Quaterniond(Lidar_R_wrt_IMU), Lidar_T_wrt_IMU);

                T_total_init = T_I_W_init * T_L_I_init;
            }

            void update_pose(State::Ptr &state)
            {
                T_I_W = state->get_IMU_pose();

                if (est_extrinsic)
                    T_L_I = state->get_L_I_pose();
                else
                    T_L_I = Sophus::SE3d(Eigen::Quaterniond(Lidar_R_wrt_IMU), Lidar_T_wrt_IMU);

                T_total = T_I_W * T_L_I;
            }

            void curr_info()
            {
                std::cout << "-----------------------------------" << std::endl;
                std::cout << "\nNew Lidar_Imu Pose :\n"
                          << T_L_I.matrix() << std::endl;
                std::cout << "-----------------------------------" << std::endl;
                std::cout << "\nNew Imu_world Pose :\n"
                          << T_I_W.matrix() << std::endl;
                std::cout << "-----------------------------------" << std::endl;
            }

            void print_match_info()
            {
                bool src_empty = std::get<0>(matched[0]).empty() || std::get<1>(matched[0]).empty();
                bool targ_empty = std::get<0>(matched[1]).empty() || std::get<1>(matched[1]).empty();

                if (!src_empty || !targ_empty)
                {
                    std::cout << "-----------------------------------" << std::endl;
                    std::cout << "Matched points at first step" << std::endl;
                    utils::printMatchedPoints(std::get<0>(matched[0]), std::get<1>(matched[0]));
                    std::cout << "Matched points at final step" << std::endl;
                    utils::printMatchedPoints(std::get<0>(matched[1]), std::get<1>(matched[1]));
                }
            }

            // track current transform
            Sophus::SE3d T_L_I, T_I_W, T_total;
            Sophus::SE3d T_L_I_init, T_I_W_init, T_total_init;

            PointsPtrVector orig_point_lidar;
            PointsPtrVector points_imu_world_coord;
            PointsPtrVector points_imu_coord;
            std::vector<PointsVectorTuple> matched;

            // the map points are in the lidar frame. Need to convert before putting in map
            PointsPtrVector map_points;

            // extrinsics
            Eigen::Matrix3d Lidar_R_wrt_IMU = Eigen::Matrix3d::Identity();
            Eigen::Vector3d Lidar_T_wrt_IMU = Eigen::Vector3d::Zero();

            bool est_extrinsic;
            bool first_matched = false;
        };

        void h_model_IMU_output(dyn_share_modified<double> &data, double gravity_sf);
        void calculate_h_lidar(dyn_share_modified<double> &data, int idx);
        bool h_model_lidar(dyn_share_modified<double> &data);
        double align_points(dyn_share_modified<double> &data);
        double kf_propagation(const Eigen::MatrixXd &h_x, const Eigen::MatrixXd &iter_props,
                              const Eigen::Matrix<double, -1, 1> &z, const Eigen::MatrixXd &noise,
                              bool is_imu);
        void outlier_removal(const std::vector<MapPoint> &targ, const std::vector<int> &returned_idxs,
                             std::vector<MapPoint> &targ_calc, std::vector<int> &tracked_idx);
        void print_imu(const State::Ptr &state);

        // attributes
        double laser_point_cov, imu_acc_meas_cov, imu_gyro_meas_cov, satu_acc, satu_gyro, sigma, threshold, dx_norm;
        int key_tracker, all_tracker, frames_to_keep, max_imu_iter;
        bool verbose, print_matrices, store_all, use_outliers;
        std::string pose_storage, all_pose_storage;
        double max_icp_iter, estimation_threshold;
        PointsPtrVector initial_frames;
        std::vector<int> tracked_idx; // tracking matched points
        PointHelper tracker;
    };
};
#endif
