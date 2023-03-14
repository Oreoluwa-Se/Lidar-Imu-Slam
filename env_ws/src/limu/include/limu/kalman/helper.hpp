#ifndef HELPER_HPP
#define HELPER_HPP
#include "common.hpp"

namespace utils
{
    // Convert a quaternion to a rotation matrix
    // Input: q - a quaternion as a 4-element vector (w, x, y, z)
    // Output: a 3x3 rotation matrix
    template <typename T>
    inline Eigen::Matrix3d quat2rmat(const T &q)
    {
        Eigen::Quaterniond quat(q);
        Eigen::Matrix3d R = quat.toRotationMatrix();

        return R;
    }

    inline std::tuple<Eigen::Matrix3d, std::unique_ptr<Eigen::Matrix3d[]>> extract_rot_dr(const Eigen::Vector4d &q)
    {
        std::unique_ptr<Eigen::Matrix3d[]> dR(new Eigen::Matrix3d[4]);

        // use perturbation to calculate derivative
        for (int i = 0; i < 4; i++)
        {
            Eigen::Vector4d dq = Eigen::Vector4d::Zero();
            dq(i) = 1.0;
            Eigen::Matrix3d dRi = quat2rmat(dq) - quat2rmat(q);
            dR[i] = dRi;
        }

        return {quat2rmat(q), std::move(dR)};
    }

    inline Eigen::Matrix3d ang_vel_to_rmat(const Eigen::Vector3d &ang_vel, double dt)
    {
        Eigen::AngleAxisd aa(dt * ang_vel.norm(), ang_vel.normalized());

        return aa.toRotationMatrix();
    }

}

#endif