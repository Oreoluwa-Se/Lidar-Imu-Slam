#ifndef COMMON_HPP
#define COMMON_HPP

#include "limu/utils/calculation_helpers.hpp"
#include "limu/utils/types.hpp"
#include <deque>
#include <mutex>
#include <string>

// define constants
#define PI_M 3.14159265358979
#define IQR_TUCHEY 1.25
#define gravity 9.81

namespace outlier
{

    // for calculating median in IQR
    template <typename itr>
    inline typename std::iterator_traits<itr>::value_type median(itr begin, itr end)
    {
        const int size = std::distance(begin, end);
        const int half = size / 2;
        itr mid = begin + half;
        // partially sorts vector. middle element becomes the median element
        std::nth_element(begin, mid, end);
        // handle even case
        if (size % 2 == 0)
        { // used because the vector isn't fully sorted
            itr p_mid = std::max_element(begin, mid);
            return (*p_mid + *mid) / 2.0;
        }

        return *mid;
    }
    // IQR stuff
    template <typename T>
    inline std::vector<double> IQR(std::vector<T> &inp)
    {
        std::vector<T> a(inp.begin(), inp.end());
        const int n = a.size();
        std::sort(a.begin(), a.end());

        if (n == 1)
        {
            return {0, a[0], a[0]};
        }

        // median of lower and upper halves
        const int half = n / 2;
        const T q1 = median(a.begin(), a.begin() + half);
        const T q3 = median(a.begin() + half + n % 2, a.end());

        const double iqr = static_cast<double>(q3 - q1);

        return {static_cast<double>(q1), static_cast<double>(q3), iqr};
    }
}
#endif