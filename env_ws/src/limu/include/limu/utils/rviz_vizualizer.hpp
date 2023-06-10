#ifndef RVIZ_VIZUALIZER_HPP
#define RVIZ_VIZUALIZER_HPP

#include "types.hpp"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/point_cloud2_iterator.h"
#include "std_msgs/Header.h"

namespace utils
{
    using PointCloud2 = sensor_msgs::PointCloud2;
    using PointField = sensor_msgs::PointField;
    using Header = std_msgs::Header;

    inline void fill_pc2_xyz(const utils::Vec3dVector &points, PointCloud2 &msg)
    {
        sensor_msgs::PointCloud2Iterator<float> msg_x(msg, "x");
        sensor_msgs::PointCloud2Iterator<float> msg_y(msg, "y");
        sensor_msgs::PointCloud2Iterator<float> msg_z(msg, "z");

        for (size_t idx = 0; idx < points.size(); idx++, ++msg_x, ++msg_y, ++msg_z)
        {
            const utils::Vec3d &point = points[idx];
            *msg_x = point.x();
            *msg_y = point.y();
            *msg_z = point.z();
        }
    }

    inline void fill_pc2_xyz(const std::vector<utils::Point> &points, PointCloud2 &msg)
    {
        sensor_msgs::PointCloud2Iterator<float> msg_x(msg, "x");
        sensor_msgs::PointCloud2Iterator<float> msg_y(msg, "y");
        sensor_msgs::PointCloud2Iterator<float> msg_z(msg, "z");
        sensor_msgs::PointCloud2Iterator<float> msg_intensity(msg, "intensity");

        for (size_t idx = 0; idx < points.size(); idx++, ++msg_x, ++msg_y, ++msg_z, ++msg_intensity)
        {
            const auto &point = points[idx];
            *msg_x = point.x();
            *msg_y = point.y();
            *msg_z = point.z();

            // Set the intensity value
            *msg_intensity = point.intensity;
        }
    }

    inline void fill_pc2_xyz(const std::vector<utils::Point::Ptr> &points, PointCloud2 &msg)
    {
        sensor_msgs::PointCloud2Iterator<float> msg_x(msg, "x");
        sensor_msgs::PointCloud2Iterator<float> msg_y(msg, "y");
        sensor_msgs::PointCloud2Iterator<float> msg_z(msg, "z");
        sensor_msgs::PointCloud2Iterator<float> msg_intensity(msg, "intensity");

        for (size_t idx = 0; idx < points.size(); idx++, ++msg_x, ++msg_y, ++msg_z, ++msg_intensity)
        {
            const auto &point = points[idx];
            *msg_x = point->x();
            *msg_y = point->y();
            *msg_z = point->z();

            // Set the intensity value
            *msg_intensity = point->intensity;
        }
    }

    inline PointCloud2 create_pc2_msg(const size_t n_points, const Header &header, bool ts = false, bool intensity = false)
    {
        PointCloud2 msg;
        sensor_msgs::PointCloud2Modifier mod(msg);
        msg.header = header;
        msg.fields.clear();

        int offset = 0;
        offset = addPointField(msg, "x", 1, PointField::FLOAT32, offset);
        offset = addPointField(msg, "y", 1, PointField::FLOAT32, offset);
        offset = addPointField(msg, "z", 1, PointField::FLOAT32, offset);
        offset += sizeOfPointField(PointField::FLOAT32);

        if (ts)
        {
            offset = addPointField(msg, "time", 1, PointField::FLOAT64, offset);
            offset += sizeOfPointField(PointField::FLOAT64);
        }

        if (intensity)
        {
            offset = addPointField(msg, "intensity", 1, PointField::FLOAT32, offset);
            offset += sizeOfPointField(PointField::FLOAT32);
        }

        // resize point cloud accordingly
        msg.point_step = offset;
        msg.row_step = msg.width * msg.point_step;
        msg.data.resize(msg.height * msg.row_step);
        mod.resize(n_points);

        return msg;
    }

    inline PointCloud2 eigen_to_pc(const utils::Vec3dVector &points, const Header &header)
    {
        PointCloud2 msg = create_pc2_msg(points.size(), header);
        fill_pc2_xyz(points, msg);

        return msg;
    }

    inline PointCloud2 eigen_to_pc(const std::vector<utils::Point> &points, const Header &header)
    {
        PointCloud2 msg = create_pc2_msg(points.size(), header, false, true);
        fill_pc2_xyz(points, msg);

        return msg;
    }

    inline PointCloud2 eigen_to_pc(const std::vector<utils::Point::Ptr> &points, const Header &header)
    {
        PointCloud2 msg = create_pc2_msg(points.size(), header, false, true);
        fill_pc2_xyz(points, msg);

        return msg;
    }

}
#endif