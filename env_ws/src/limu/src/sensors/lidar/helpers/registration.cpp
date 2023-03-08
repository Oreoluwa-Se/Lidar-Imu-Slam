#include "limu/sensors/lidar/helpers/registration.hpp"
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <cmath>

namespace
{
    struct ResultTuple
    {
        ResultTuple()
            : JTJ(utils::matrix<6, 6>::Zero()),
              JTr(utils::vector<6>::Zero()){};

        ResultTuple(const ResultTuple &other)
        {
            this->JTJ = other.JTJ;
            this->JTr = other.JTr;
        };

        ResultTuple operator+(const ResultTuple &other)
        {
            this->JTJ += other.JTJ;
            this->JTr += other.JTr;
            return *this;
        }

        utils::matrix<6, 6> JTJ;
        utils::vector<6> JTr;
    };

    double percentage_difference(double value1, double value2)
    {
        double avg = (value1 + value2) / 2;
        double diff = std::abs(value1 - value2);
        return (diff / avg) * 100;
    }

}

namespace lidar
{
    SE3d align_clouds(const utils::Vec3dVector &source, const utils::Vec3dVector &target, double th)
    {
        // compute jacobian and residuals
        auto compute_jacobian_residual = [&](auto idx)
        {
            const utils::Vec3d residual = source[idx] - target[idx];
            const auto skew_source = utils::skew_matrix(source[idx]);

            return std::make_tuple(
                (utils::matrix<3, 6>() << utils::Mat3d::Identity(), -skew_source).finished(),
                residual);
        };

        // defining weight function - Same as EQ(13)
        auto weight = [&](double res_sq)
        { return utils::square(th) / utils::square(th + res_sq); };

        const ResultTuple jac = tbb::parallel_reduce(
            // Range
            tbb::blocked_range<size_t>{0, source.size()},
            // Identity
            ResultTuple(),
            // 1st lambda: Parallel computation
            [&](const tbb::blocked_range<size_t> &r, ResultTuple J)
            {
                for (size_t i = r.begin(); i < r.end(); ++i)
                {
                    const auto result = compute_jacobian_residual(i);
                    const auto &J_r = std::get<0>(result);
                    const auto &res = std::get<1>(result);
                    const double w = weight(res.squaredNorm());

                    J.JTJ.noalias() += J_r.transpose() * w * J_r;
                    J.JTr.noalias() += J_r.transpose() * w * res;
                }

                return J;
            },
            // 2nd lambda
            [](ResultTuple a, const ResultTuple &b) -> ResultTuple
            {
                return a + b;
            });

        // solve for twist
        const auto &JTJ = jac.JTJ;
        const auto &JTr = jac.JTr;
        const utils::vector<6> x = JTJ.ldlt().solve(-JTr);
        return utils::vector6d_to_mat4d(x);
    }

    SE3d ICP(
        VoxelHashMap &local_map, const utils::Vec3dVector &points,
        const SE3d &init_guess, const double max_corresp_dist, const double kernel,
        const int &icp_max_iteration, const double &est_threshold)
    {
        if (local_map.empty())
            return init_guess;

        utils::Vec3dVector source = points;
        utils::transform_points(init_guess, source);

        // ICP-Loop
        SE3d T_icp = SE3d();

        for (int j = 0; j < icp_max_iteration; ++j)
        {
            // Equation(11)
            const auto result = local_map.get_correspondences(source, max_corresp_dist);
            const auto &src = std::get<0>(result);
            const auto &target = std::get<1>(result);

            // Equation (12) - For longer runs we alternate IQR method and without IQR
            auto estimate = align_clouds(src, target, kernel);

            // transform points based on current estimation
            utils::transform_points(estimate, source);

            // update iterations
            T_icp = estimate * T_icp;

            if (estimate.log().norm() < est_threshold)
                break;
        }

        // return updated pose
        return T_icp * init_guess;
    }

}