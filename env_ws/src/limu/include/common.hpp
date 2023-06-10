#ifndef COMMON_HPP
#define COMMON_HPP

#include "limu/utils/calculation_helpers.hpp"
#include <boost/thread/shared_mutex.hpp>
#include "limu/utils/rviz_vizualizer.hpp"
#include "limu/utils/print_helpers.hpp"
#include "limu/utils/types.hpp"
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <utility>
#include <fstream>
#include <string>
#include <deque>
#include <mutex>
#include <tuple>

// define constants
#define PI_M 3.14159265358979
#define IQR_TUCHEY 1.5
#define MAD_THRESH_VAL 3.0

// yaml exractors -> should just make this a template
inline void set_double_param(double &param, const YAML::Node &value)
{
    param = value.as<double>();
}

inline void set_bool_param(bool &param, const YAML::Node &value)
{
    param = value.as<bool>();
}

inline void set_int_param(int &param, const YAML::Node &value)
{
    param = value.as<int>();
}

inline std::vector<double> extract_rot_mat_values(const YAML::Node &imuRNode)
{
    std::vector<double> rot_values;
    for (const auto &row : imuRNode)
    {
        for (const auto &value : row)
        {
            rot_values.push_back(value.as<double>());
        }
    }

    return rot_values;
}
#endif