#ifndef EKF_HELPER_HPP
#define EKF_HELPER_HPP

#include "common.hpp"
#include <unordered_map>
#include <tbb/parallel_for.h>

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

    inline Eigen::Quaterniond rmat2quat(const Eigen::Matrix3d &mat)
    {
        Eigen::Quaterniond quat(mat);
        quat.normalize();
        return quat;
    }
    inline Eigen::Matrix3d extract_rot_dr(const Eigen::Vector4d &q, Eigen::Matrix3d (&dR)[4])
    {
        // use perturbation to calculate derivative
        for (int i = 0; i < 4; i++)
        {
            Eigen::Vector4d dq = Eigen::Vector4d::Zero();
            dq(i) = 1.0;
            Eigen::Matrix3d dRi = quat2rmat(dq) - quat2rmat(q);
            dR[i] = dRi;
        }

        return quat2rmat(q);
    }

    inline Eigen::Matrix3d ang_vel_to_rmat(const Eigen::Vector3d &ang_vel, double dt)
    {
        Eigen::AngleAxisd aa(dt * ang_vel.norm(), ang_vel.normalized());

        return aa.toRotationMatrix();
    }

    inline Eigen::Vector3d quat_mult_point(const Eigen::Vector4d &quat_v, const Eigen::Vector3d &p)
    {
        Eigen::Quaterniond q(quat_v);
        Eigen::Quaterniond q_inv = q.conjugate();
        Eigen::Quaterniond p_quat(0, p(0), p(1), p(2));
        Eigen::Quaterniond p_rotated = q * p_quat * q_inv;
        return p_rotated.vec();
    }

    inline Eigen::Vector4d quat_mult_point_q(const Eigen::Vector4d &quat_v, const Eigen::Vector3d &p)
    {
        Eigen::Quaterniond q(quat_v);
        Eigen::Quaterniond q_inv = q.conjugate();
        Eigen::Quaterniond p_quat(0, p(0), p(1), p(2));
        Eigen::Quaterniond p_rotated = q * p_quat * q_inv;
        return Eigen::Vector4d(p_rotated.w(), p_rotated.x(), p_rotated.y(), p_rotated.z());
    }

    inline Eigen::Quaterniond quat_mult_norm(const Eigen::Quaterniond &q1, const Eigen::Quaterniond &q2)
    {
        Eigen::Quaterniond q = q1 * q2;
        q.normalize();
        return q;
    }

    inline Eigen::Vector4d rot_norm(const Eigen::Vector4d &v)
    {
        Eigen::Quaterniond q(v(0), v(1), v(2), v(3));
        q.normalize();
        return q.coeffs();
    }
    inline Eigen::Quaterniond quat_mult_norm(const Eigen::Vector4d &v1, const Eigen::Vector4d &v2)
    {
        Eigen::Quaterniond q1(v1);
        Eigen::Quaterniond q2(v2);

        return quat_mult_norm(q1, q2);
    }

    inline Eigen::Matrix4d ohm(const Eigen::Vector3d &vec)
    {
        Eigen::Matrix4d S;
        (S << 0, -vec[0], -vec[1], -vec[2],
         vec[0], 0, -vec[2], vec[1],
         vec[1], vec[2], 0, -vec[0],
         vec[2], -vec[1], vec[0], 0)
            .finished();

        return S;
    }

    inline Eigen::Vector4d ohm_to_vec(const Eigen::Matrix4d &mat)
    {
        Eigen::Vector4d _vec(0, mat(1, 0), mat(2, 0), mat(3, 0));

        return _vec;
    }

    inline Eigen::Vector3d quat_to_euler(const Eigen::Vector4d &v1)
    {
        Eigen::Quaterniond q(v1.normalized());
        Eigen::Vector3d euler;

        double sinr_cosp = 2.0 * (q.w() * q.x() + q.y() * q.z());
        double cosr_cosp = 1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y());
        euler(0) = std::atan2(sinr_cosp, cosr_cosp);

        double sinp = 2.0 * (q.w() * q.y() - q.z() * q.x());
        if (std::abs(sinp) >= 1.0)
        {
            euler(1) = std::copysign(M_PI / 2.0, sinp);
        }
        else
        {
            euler(1) = std::asin(sinp);
        }

        double siny_cosp = 2.0 * (q.w() * q.z() + q.x() * q.y());
        double cosy_cosp = 1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
        euler(2) = std::atan2(siny_cosp, cosy_cosp);

        return euler;
    }

    inline Eigen::Matrix4d left_quat2mat(const Eigen::Quaterniond &q)
    {
        Eigen::Matrix4d mat;
        mat << q.w(), -q.x(), -q.y(), -q.z(),
            q.x(), q.w(), -q.z(), q.y(),
            q.y(), q.z(), q.w(), -q.x(),
            q.z(), -q.y(), q.x(), q.w();
        return mat;
    }

    inline Eigen::Matrix4d right_quat2mat(const Eigen::Quaterniond &q)
    {
        Eigen::Matrix4d mat;
        mat << q.w(), -q.x(), -q.y(), -q.z(),
            q.x(), q.w(), q.z(), -q.y(),
            q.y(), -q.z(), q.w(), q.x(),
            q.z(), q.y(), -q.x(), q.w();
        return mat;
    }

}

#endif