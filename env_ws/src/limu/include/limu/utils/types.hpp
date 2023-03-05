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
    { // slightly different from previous.
        size_t operator()(const Voxel &vox) const
        {
            // Using the FNV-1a hash
            constexpr uint32_t offset_basis = 0x811c9dc5;
            constexpr uint32_t fnv_prime = 0x01000193;

            size_t hash = offset_basis;
            const uint8_t *byte_ptr = reinterpret_cast<const uint8_t *>(&vox);
            const uint8_t *end_ptr = byte_ptr + sizeof(Voxel);

            while (byte_ptr < end_ptr)
            {
                hash ^= *byte_ptr++;
                hash ^= fnv_prime;
            }
            return hash;
        }
    };
}
#endif