// #include "limu/kalman/ekf.hpp"
// #include "limu/kalman/helper.hpp"
// #include <cassert>
// #include <tbb/parallel_for.h>

// namespace kalman
// {
//     void EKF::predict(StatesGroup &m, double dt)
//     {
//         // reset matrices
//         dydx.setIdentity();
//         dydq.setZero();

//         // random walk bias for gyro
//         if (params->gyro_process_noise > 0.0)
//         {
//             const double qc = utils::square(params->gyro_process_noise);
//             const double theta = params->gyro_process_noise_rev;
//             Q.block(Q_BGA_DRIFT, Q_BGA_DRIFT, 3, 3).setIdentity(3, 3) *= params->noise_scale * qc;

//             if (theta > 0.0)
//                 Q.block(Q_BGA_DRIFT, Q_BGA_DRIFT, 3, 3) *= (1 - exp(-2 * dt * theta)) / (2 * theta);
//         }

//         // random walk bias for accelerometer
//         if (params->acc_process_noise > 0.0)
//         {
//             const double qc = utils::square(params->acc_process_noise);
//             const double theta = params->acc_process_noise_rev;
//             Q.block(Q_BAA_DRIFT, Q_BAA_DRIFT, 3, 3).setIdentity(3, 3) *= params->noise_scale * qc;

//             if (theta > 0.0)
//                 Q.block(Q_BAA_DRIFT, Q_BAA_DRIFT, 3, 3) *= (1 - exp(-2 * dt * theta)) / (2 * theta);
//         }

//         // quatertion rotation exterior quaternion
//         Eigen::Matrix3d A = calculate_S(m.imu_gyro, m, -dt);

//         Eigen::Matrix3d dR[4];
//         // Eigen::Matrix3d R = utils::extract_rot_dr(A * m., dR);
//     }

//     Eigen::Matrix4d EKF::calculate_S(const Eigen::Vector3d &xg, const StatesGroup &m, const double dt)
//     {
//         const Eigen::Vector3d w = xg - m.gyro_bias;
//         Eigen::Matrix4d S;
//         (S << 0, -w[0], -w[1], -w[2],
//          w[0], 0, -w[2], w[1],
//          w[1], w[2], 0, -w[0],
//          w[2], -w[1], w[0], 0)
//             .finished();

//         S *= dt / 2;

//         return S.exp();
//     }
// }