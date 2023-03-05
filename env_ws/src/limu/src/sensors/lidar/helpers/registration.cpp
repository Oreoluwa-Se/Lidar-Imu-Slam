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
    SE3d align_clouds_IQR(
        const utils::Vec3_Vec3Tuple &points, double th)
    {
        const auto &source = std::get<0>(points);
        const auto &target = std::get<1>(points);

        // calculate residual and squarednorm
        std::vector<utils::Vec3d> residual(source.size());
        std::vector<double> residual_sq_norm(source.size());

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, source.size()),
            [&](const tbb::blocked_range<size_t> &range)
            {
                for (size_t i = range.begin(); i != range.end(); ++i)
                {
                    const auto res = source[i] - target[i];
                    residual_sq_norm[i] = res.squaredNorm();
                    residual[i] = std::move(res);
                }
            });

        // use iqr to calculate bounds
        const auto &iqr_val = outlier::IQR(residual_sq_norm);
        double low_bound = iqr_val[0] - IQR_TUCHEY * iqr_val[2];
        double high_bound = iqr_val[1] + IQR_TUCHEY * iqr_val[2];

        // compute jacobian and residuals
        auto compute_jacobian = [&](auto idx)
        {
            const auto skew_source = utils::skew_matrix(source[idx]);

            return (utils::matrix<3, 6>() << utils::Mat3d::Identity(), -skew_source).finished();
        };

        // defining weight function
        auto weight = [&](double idx)
        {
            const auto &res_sq = residual_sq_norm[idx];
            const auto cost = utils::square(th) / utils::square(th + res_sq);
            if (res_sq >= low_bound && res_sq <= high_bound)
                return cost;

            return 0.5 * cost;
        };

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
                    const auto J_r = compute_jacobian(i);
                    const auto &res = residual[i];
                    const double w = weight(i);

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

    SE3d align_clouds(
        const utils::Vec3_Vec3Tuple &points, double th)
    {
        const auto &source = std::get<0>(points);
        const auto &target = std::get<1>(points);

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
        const SE3d &init_guess, const double &max_corresp_dist, const double &kernel,
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
            // Equation(11) returns: std::tuple<Vec3dVector, VecEigenPtrVec3d> {source, target}
            const auto result = local_map.get_correspondences(source, max_corresp_dist);
            const auto &src = std::get<0>(result);
            const auto &target = std::get<1>(result);

            // Equation (12) - For longer runs we alternate IQR method and without IQR
            // auto estimate = (j % 2 == 0) ? align_clouds_IQR(result, kernel) : align_clouds(result, kernel);
            auto estimate = align_clouds(result, kernel);

            // transform points based on current estimation
            utils::transform_points(estimate, source);

            // update iterations
            T_icp = estimate * T_icp;

            auto calc_error = estimate.log().norm();
            if (calc_error < est_threshold)
                break;

            // // difference is within 50 percent.
            // if (percentage_difference(est_threshold, calc_error) < 0)
            //     break;
        }

        // return updated pose
        return T_icp * init_guess;
    }

}