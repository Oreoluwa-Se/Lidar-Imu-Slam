#ifndef TYPES_HPP
#define TYPES_HPP

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <functional>
#include <vector>

namespace utils
{
    // Type definitions
    typedef Eigen::Vector3i Voxel;
    typedef Eigen::Vector3d Vec3d;
    typedef Eigen::Matrix3d Mat3d;
    typedef std::tuple<Vec3d, Vec3d> Vec3Tuple;
    typedef std::vector<utils::Vec3d> Vec3dVector;
    typedef std::vector<std::shared_ptr<utils::Vec3d>> VecEigenPtrVec3d;
    typedef std::tuple<Vec3dVector, VecEigenPtrVec3d> Vec3_VecPtrTuple; // not used currently
    typedef std::tuple<Vec3dVector, Vec3dVector> Vec3_Vec3Tuple;

    // generic matrix type.
    template <int Rows = Eigen::Dynamic, int Cols = Rows, bool UseRowMajor = false, typename T = double>
    using matrix = typename std::conditional<
        Rows != 1 && Cols != 1,
        Eigen::Matrix<T, Rows, Cols, UseRowMajor ? Eigen::RowMajor : Eigen::ColMajor>,
        Eigen::Matrix<T, Rows, Cols>>::type;

    // generic vector type.
    template <int Dim = Eigen::Dynamic, bool RowVec = false, typename T = double>
    using vector = typename std::conditional<RowVec, matrix<1, Dim, false, T>, matrix<Dim, 1, false, T>>::type;

    // pcl point types
    typedef pcl::PointXYZINormal PointNormal;
    typedef pcl::PointXYZRGB PointRGB;
    typedef pcl::PointCloud<PointNormal> PointCloudXYZI;
    typedef pcl::PointCloud<PointRGB> PointCloudXYZRGB;
    typedef std::tuple<PointCloudXYZI, PointCloudXYZI> PointCloudXYZITuple;
    typedef std::tuple<PointCloudXYZI::Ptr, PointCloudXYZI::Ptr> PointCloudXYZIPtrTuple;

    struct VoxelHash
    {
        size_t operator()(const Voxel &voxel) const
        {
            const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
            return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349663 ^ vec[2] * 83492791);
        }
    };

    struct ImuData
    {
        typedef std::shared_ptr<ImuData> Ptr;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        /**
         * @brief Default constructor.
         */
        ImuData() = default;

        /**
         * @brief Constructor that initializes the IMU data with given values.
         *
         * @param acc_ The accelerometer data.
         * @param gyro_ The gyroscope data.
         * @param ts The timestamp of the IMU data.
         */
        ImuData(const utils::Vec3d &acc_, const utils::Vec3d &gyro_, double ts)
            : acc(acc_), gyro(gyro_), timestamp(ts) {}

        /**
         * @brief Copy constructor.
         *
         * @param other The IMU data object to be copied.
         */
        ImuData(const ImuData &other)
            : timestamp(other.timestamp),
              acc(other.acc),
              gyro(other.gyro) {}

        /**
         * @brief Move constructor.
         *
         * @param other The IMU data object to be moved.
         */
        ImuData(ImuData &&other) noexcept
            : timestamp(std::move(other.timestamp)),
              acc(std::move(other.acc)),
              gyro(std::move(other.gyro)) {}

        /**
         * @brief Copy assignment operator.
         *
         * @param other The IMU data object to be copied.
         * @return Reference to the assigned object.
         */
        ImuData &operator=(const ImuData &other)
        {
            if (this != &other)
            {
                timestamp = other.timestamp;
                acc = other.acc;
                gyro = other.gyro;
            }
            return *this;
        }

        /**
         * @brief Move assignment operator.
         *
         * @param other The IMU data object to be moved.
         * @return Reference to the assigned object.
         */
        ImuData &operator=(ImuData &&other) noexcept
        {
            if (this != &other)
            {
                timestamp = std::move(other.timestamp);
                acc = std::move(other.acc);
                gyro = std::move(other.gyro);
            }
            return *this;
        }

        double timestamp;  /**< The timestamp of the IMU data. */
        utils::Vec3d acc;  /**< The accelerometer data. */
        utils::Vec3d gyro; /**< The gyroscope data. */
    };

    /**
     * @brief Point struct represents a point in 3D space with additional attributes.
     */
    struct Point
    {
        typedef std::shared_ptr<Point> Ptr;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        /**
         * @brief Default constructor. Initializes all members to default values.
         */
        Point()
            : point(utils::Vec3d::Zero()), intensity(0.0), timestamp(0.0), dt(0.0) {}

        /**
         * @brief Constructor with a 3D point.
         * @param p The 3D point coordinates.
         */
        explicit Point(const utils::Vec3d &p)
            : point(p), intensity(0.0), timestamp(0.0), dt(0.0) {}

        /**
         * @brief Constructor with individual coordinates, intensity, timestamp, and dt.
         * @param x The X coordinate.
         * @param y The Y coordinate.
         * @param z The Z coordinate.
         * @param intensity The intensity value.
         * @param timestamp The timestamp value (in seconds).
         * @param dt The time difference value (in seconds).
         */
        Point(double x, double y, double z, double intensity, double timestamp, double dt)
            : point(utils::Vec3d(x, y, z)), intensity(intensity), timestamp(timestamp), dt(dt) {}

        /**
         * @brief Constructor with a 3D point, intensity, timestamp, and dt.
         * @param point The 3D point coordinates.
         * @param intensity The intensity value.
         * @param timestamp The timestamp value (in seconds).
         * @param dt The time difference value (in seconds).
         */
        Point(const utils::Vec3d &point, double intensity, double timestamp, double dt)
            : point(point), intensity(intensity), timestamp(timestamp), dt(dt) {}

        /**
         * @brief Copy constructor.
         * @param other The Point object to be copied.
         */
        Point(const Point &other)
            : point(other.point), intensity(other.intensity), timestamp(other.timestamp), dt(other.dt) {}

        /**
         * @brief Move constructor.
         * @param other The Point object to be moved.
         */
        Point(Point &&other) noexcept
            : point(std::move(other.point)), intensity(other.intensity), timestamp(other.timestamp), dt(other.dt) {}

        /**
         * @brief Assignment operator.
         * @param other The Point object to be assigned.
         * @return Reference to the updated Point object.
         */
        Point &operator=(const Point &other)
        {
            if (this != &other)
            {
                point = other.point;
                intensity = other.intensity;
                timestamp = other.timestamp;
                dt = other.dt;
            }
            return *this;
        }

        /**
         * @brief Move assignment operator.
         * @param other The Point object to be moved.
         * @return Reference to the updated Point object.
         */
        Point &operator=(Point &&other) noexcept
        {
            if (this != &other)
            {
                point = std::move(other.point);
                intensity = other.intensity;
                timestamp = other.timestamp;
                dt = other.dt;
            }
            return *this;
        }

        /**
         * @brief Addition operator.
         * @param other The Point object to be added.
         * @return The result of adding the current Point object with the other Point object.
         */
        Point operator+(const Point &other) const
        {
            Point result(*this);
            result.point += other.point;
            return result;
        }

        /**
         * @brief Subtraction operator.
         * @param other The Point object to be subtracted.
         * @return The result of subtracting the current Point object by the other Point object.
         */
        Point operator-(const Point &other) const
        {
            Point result(*this);
            result.point -= other.point;
            return result;
        }

        /**
         * @brief Calculate the squared norm of the point.
         * @return The squared norm.
         */
        double squaredNorm() const { return point.squaredNorm(); }

        /**
         * @brief Calculate the norm of the point.
         * @return The norm.
         */
        double norm() const { return point.norm(); }

        /**
         * @brief Get the X coordinate of the point.
         * @return The X coordinate.
         */
        double x() const { return point.x(); }

        /**
         * @brief Get the Y coordinate of the point.
         * @return The Y coordinate.
         */
        double y() const { return point.y(); }

        /**
         * @brief Get the Z coordinate of the point.
         * @return The Z coordinate.
         */
        double z() const { return point.z(); }

        // attributes
        utils::Vec3d point; /**< The 3D point coordinates. */
        double intensity;   /**< The intensity value. */
        double timestamp;   /**< The timestamp value (in seconds). */
        double dt;          /**< The time difference value (in seconds). */
        int frame_id = 0;   /**< Frame index. */
        bool point_valid = true;
        bool is_default = false;
    };

    struct LidarFrame
    {
        LidarFrame()
            : pc(),
              timestamp(0.0) {}

        std::vector<Point::Ptr> pc;
        double timestamp;
    };
}
#endif