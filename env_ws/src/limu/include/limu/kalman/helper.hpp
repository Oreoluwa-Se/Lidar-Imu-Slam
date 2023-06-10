#ifndef EKF_HELPER_HPP
#define EKF_HELPER_HPP

#include "common.hpp"
#include <unordered_map>
#include <tbb/parallel_for.h>
#include <iomanip>

namespace utils
{
    inline Eigen::Vector4d quat2vec(const Eigen::Quaterniond &v)
    {
        Eigen::Vector4d q;
        q << v.w(), v.x(), v.y(), v.z();
        q.normalize();
        return q;
    }

    inline Eigen::Quaterniond vec2quat(const Eigen::Vector4d &v)
    {
        return Eigen::Quaterniond(v(0), v(1), v(2), v(3)).normalized();
    }

    inline Sophus::SE3d get_pose(const Eigen::Vector4d &quat_v, const Eigen::Vector3d &t)
    {
        return Sophus::SE3d(utils::vec2quat(quat_v), t);
    }

    inline Eigen::Vector4d rmat2quat(const Eigen::Matrix3d &R)
    {
        Eigen::Quaterniond q(R);
        Eigen::Vector4d q_v;
        // Set the order of coefficients to [w, x, y, z]
        q_v << q.w(), q.x(), q.y(), q.z();

        return q_v;
    }

    inline Eigen::Matrix3d quat2rmat(const Eigen::Vector4d &qvec)
    {
        Eigen::Quaterniond q = utils::vec2quat(qvec);
        return q.toRotationMatrix();
    }

    inline Eigen::Vector3d quat_to_euler(const Eigen::Vector4d &v1)
    {
        Eigen::Quaterniond q = utils::vec2quat(v1);
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

    inline Eigen::Vector3d rmat_to_euler(const Eigen::Matrix3d &R)
    {
        Eigen::Vector4d q = rmat2quat(R);

        return quat_to_euler(q);
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

    inline Eigen::Matrix<double, 3, 4> jacobian_wrt_quat(const Eigen::Vector4d &quat_v, const Eigen::Vector3d &p)
    {
        /* General function for calculation Jacobian wrt quaternion */
        const double w = quat_v[0];
        const Eigen::Vector3d v(quat_v[1], quat_v[2], quat_v[3]);

        const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

        Eigen::Matrix<double, 3, 4> int_mat;

        // quaternion solutions
        int_mat.block<3, 1>(0, 0) = 2 * (w * p + v.cross(p));
        int_mat.block<3, 3>(0, 1) = 2 * (v.transpose() * p * I + v * p.transpose() - p * v.transpose() - w * utils::skew_matrix(p));

        return int_mat;
    }

    // convert vector 4d orientation into rotation matrix
    inline Eigen::Matrix3d vec4d_to_rmat(const Eigen::Vector4d &vec)
    {
        Eigen::Matrix3d rot_m = utils::quat2rmat(vec);
        utils::rot_mat_norm(rot_m);

        return rot_m;
    }

    inline Eigen::Vector4d quat_left_multiply(const Eigen::Vector4d &vec, const Eigen::Vector4d &prev)
    {
        // used for quaternion quaternion multiplication
        Eigen::Matrix4d S;

        (S << vec[0], -vec[1], -vec[2], -vec[3],
         vec[1], vec[0], -vec[3], vec[2],
         vec[2], vec[3], vec[0], -vec[1],
         vec[3], -vec[2], vec[1], vec[0])
            .finished();

        Eigen::Vector4d r = S * prev;
        r.normalize();

        return r;
    }

    inline Eigen::Vector4d quat_right_multiply(const Eigen::Vector4d &vec, const Eigen::Vector4d &prev)
    {
        // used for quaternion quaternion multiplication
        Eigen::Matrix4d S;

        // represents the skewed quaternions
        (S << vec[0], -vec[1], -vec[2], -vec[3],
         vec[1], vec[0], vec[3], -vec[2],
         vec[2], -vec[3], vec[0], vec[1],
         vec[3], vec[2], -vec[1], vec[0])
            .finished();

        Eigen::Vector4d r = S * prev;
        r.normalize();

        return r;
    }

    inline Eigen::Vector4d dquat_left_multiply(const Eigen::Vector3d &vec, const Eigen::Vector4d &prev, double dt = 1)
    {
        // used when we are incrementing by dtheta
        Eigen::Matrix4d S;

        (S << 0, -vec[0], -vec[1], -vec[2],
         vec[0], 0, -vec[2], vec[1],
         vec[1], vec[2], 0, -vec[0],
         vec[2], -vec[1], vec[0], 0)
            .finished();

        S *= 0.5 * dt;
        Eigen::Matrix4d A = S.exp();
        Eigen::Vector4d r = A * prev;
        r.normalize();

        return r;
    }

    // inline Eigen::Matrix3d extract_ori_ori(const Eigen::Vector3d &vec, double dt = 1)
    // {
    //     // used when we are incrementing by dtheta
    //     Eigen::Matrix4d S;

    //     (S << 0, -vec[0], -vec[1], -vec[2],
    //      vec[0], 0, -vec[2], vec[1],
    //      vec[1], vec[2], 0, -vec[0],
    //      vec[2], -vec[1], vec[0], 0)
    //         .finished();

    //     S *= 0.5 * dt;
    //     Eigen::Matrix4d A = S.exp();
    //     Eigen::Matrix3d rot_m = A.block(0, 0, 3, 3);
    //     utils::rot_mat_norm(rot_m);

    //     return rot_m;
    // }

    inline Eigen::Matrix3d extract_ori_ori(const Eigen::Vector3d &vec, double dt = 1)
    {
        Eigen::Matrix3d rot_m = utils::skew_matrix(vec) * dt;
        rot_m = rot_m.exp();

        // utils::rot_mat_norm(rot_m);

        return rot_m;
    }

    inline Eigen::Vector4d dquat_right_multiply(const Eigen::Vector3d &vec, const Eigen::Vector4d &prev, double dt = 1)
    {
        // used when we are incrementing by dtheta
        Eigen::Matrix4d S;

        (S << 0, -vec[0], -vec[1], -vec[2],
         vec[0], 0, vec[2], -vec[1],
         vec[1], -vec[2], 0, vec[0],
         vec[2], vec[1], -vec[0], 0)
            .finished();

        S *= 0.5 * dt;
        Eigen::Matrix4d A = S.exp();
        Eigen::Vector4d r = A * prev;
        r.normalize();

        return r;
    }

    // derviative of true state orientation wrt error state.
    inline Eigen::MatrixXd ori_ts_jacobian(const Eigen::Vector4d &rot)
    {

        Eigen::MatrixXd rotation_matrix(4, 3);
        rotation_matrix << -rot[1], -rot[2], -rot[3],
            rot[0], rot[3], -rot[2],
            -rot[3], rot[0], rot[1],
            rot[2], -rot[1], rot[0];

        return 0.5 * rotation_matrix;
    }

    inline Eigen::Matrix3d rot_mat_frm_vec(const Eigen::Vector3d &rot_vec)
    {
        double angle = rot_vec.norm();
        Eigen::Vector3d axis = rot_vec.normalized();

        double c = cos(angle);
        double s = sin(angle);
        double t = 1 - c;

        Eigen::Matrix3d cpm;
        (cpm << 0, -axis.z(), axis.y(),
         axis.z(), 0, -axis.x(),
         -axis.y(), axis.x(), 0)
            .finished();

        Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity() + (s * cpm) + (t * cpm * cpm);
        utils::rot_mat_norm(rotation);

        return rotation;
    }

}

#endif