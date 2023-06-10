#ifndef SYNC_FRAME_HPP
#define SYNC_FRAME_HPP

#include "sensor_msgs/Imu.h"
#include "common.hpp"
#include "tbb/parallel_for.h"
#include "tbb/concurrent_vector.h"
#include "limu/sensors/lidar/frame.hpp"

namespace frame
{
    struct SensorData
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        enum class Type
        {
            Lidar,
            Imu
        };

        explicit SensorData(utils::LidarFrame &&data)
            : m_type(Type::Lidar),
              timestamp(data.timestamp),
              lidar_data(std::move(data)) {}

        explicit SensorData(utils::ImuData::Ptr &&p)
            : m_type(Type::Imu),
              timestamp(p->timestamp),
              imu_data(std::move(p)) {}

        void print() const
        {
            if (m_type == Type::Lidar)
            {
                std::cout << "Lidar Data:" << std::endl;
                std::cout << "Timestamp: ";
                utils::print_double(timestamp);
                std::cout << "Number of points: " << static_cast<int>(lidar_data.pc.size()) << std::endl;
            }
            else if (m_type == Type::Imu)
            {
                std::cout << "IMU Data:" << std::endl;
                std::cout << "Timestamp: ";
                utils::print_double(timestamp);
            }
        }

        Type m_type;
        double timestamp;
        utils::LidarFrame lidar_data;
        utils::ImuData::Ptr imu_data;
    };

    struct LidarImuInit
    {
        typedef std::shared_ptr<LidarImuInit> Ptr;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        void merge_sensor_data()
        {
            // reserve memory for data
            data.reserve(lidar_buffer.size() + imu_buffer.size());

            // insert all imu data
            for (auto &&imu : imu_buffer)
                data.emplace_back(std::move(imu));

            // insert all lidar data
            for (auto &&l_data : lidar_buffer)
                data.emplace_back(std::move(l_data));

            // Sort by timestamp
            std::sort(
                data.begin(), data.end(),
                [](const SensorData &a, const SensorData &b)
                { return a.timestamp < b.timestamp; });
        }

        // storage buffers
        std::deque<utils::ImuData::Ptr> imu_buffer;
        std::deque<utils::LidarFrame> lidar_buffer;
        std::vector<SensorData> data;

        double lidar_beg_time = 0.0;
        double lidar_last_time = 0.0;
        utils::Vec3d mean_acc;
    };
}

#endif
