#ifndef PRINT_HELPERS_HPP
#define PRINT_HELPERS_HPP

#include "calculation_helpers.hpp"
#include <sys/stat.h>
#include <filesystem>
#include <string>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>

namespace utils
{
    inline void print_double(double value)
    {
        std::cout << std::fixed << std::setprecision(5) << value << std::endl;
    }

    inline void print_vector(const std::vector<utils::PointNormal, Eigen::aligned_allocator<utils::PointNormal>> &vec)
    {
        std::cout << "Eigen::Vector3d Vector: " << std::endl;
        std::cout << "------------------------" << std::endl;
        for (int i = 0; i < vec.size(); ++i)
        {
            std::cout << vec[i].x << ", " << vec[i].y << ", " << vec[i].z << std::endl;
        }
        std::cout << "------------------------" << std::endl;
    }

    inline void print_vector(const std::vector<utils::Point> &vec)
    {
        std::cout << "Eigen::Vector3d Vector: " << std::endl;
        std::cout << "------------------------" << std::endl;
        for (int i = 0; i < vec.size(); ++i)
        {
            std::cout << vec[i].x() << ", " << vec[i].y() << ", " << vec[i].z() << std::endl;
        }
        std::cout << "------------------------" << std::endl;
    }

    inline void print_vector(const std::vector<utils::Point::Ptr> &vec)
    {
        std::cout << "Eigen::Vector3d Vector: " << std::endl;
        std::cout << "------------------------" << std::endl;
        for (int i = 0; i < vec.size(); ++i)
        {
            std::cout << vec[i]->x() << ", " << vec[i]->y() << ", " << vec[i]->z() << std::endl;
        }
        std::cout << "------------------------" << std::endl;
    }

    inline void print_vector(const std::vector<utils::Vec3d> &vec)
    {
        std::cout << "Eigen::Vector3d Vector: " << std::endl;
        std::cout << "------------------------" << std::endl;
        for (int i = 0; i < vec.size(); ++i)
        {
            std::cout << vec[i][0] << ", " << vec[i][1] << ", " << vec[i][2] << std::endl;
        }
        std::cout << "------------------------" << std::endl;
    }

    inline void printEigenMatrix(const Eigen::MatrixXd &matrix, const std::string &tableName)
    {
        std::cout << "Table Name: " << tableName << std::endl;
        std::cout << "Dimensions: " << matrix.rows() << " x " << matrix.cols() << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        for (int i = 0; i < matrix.rows(); ++i)
        {
            for (int j = 0; j < matrix.cols(); ++j)
            {
                std::cout << std::setw(10) << std::setprecision(4) << matrix(i, j) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "----------------------------------------" << std::endl;
    }

    inline void printMatchedPoints(const std::vector<utils::Point> &src, const std::vector<utils::Point> &targ)
    {
        std::cout << "Matched Points" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << std::setw(15) << "Source X" << std::setw(15) << "Source Y" << std::setw(15) << "Source Z";
        std::cout << std::setw(15) << "Target X" << std::setw(15) << "Target Y" << std::setw(15) << "Target Z" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        for (int i = 0; i < src.size(); ++i)
        {
            std::cout << std::setw(15) << std::setprecision(4) << src[i].x() << std::setw(15) << std::setprecision(4) << src[i].y() << std::setw(15) << std::setprecision(4) << src[i].z();
            std::cout << std::setw(15) << std::setprecision(4) << targ[i].x() << std::setw(15) << std::setprecision(4) << targ[i].y() << std::setw(15) << std::setprecision(4) << targ[i].z() << std::endl;
        }
        std::cout << "----------------------------------------" << std::endl;
    }

    inline void printMatchedPoints(const std::vector<utils::Point::Ptr> &src, const std::vector<utils::Point::Ptr> &targ)
    {
        std::cout << "Matched Points" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << std::setw(15) << "Source X" << std::setw(15) << "Source Y" << std::setw(15) << "Source Z";
        std::cout << std::setw(15) << "Target X" << std::setw(15) << "Target Y" << std::setw(15) << "Target Z" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        for (int i = 0; i < src.size(); ++i)
        {
            std::cout << std::setw(15) << std::setprecision(4) << src[i]->x() << std::setw(15) << std::setprecision(4) << src[i]->y() << std::setw(15) << std::setprecision(4) << src[i]->z();
            std::cout << std::setw(15) << std::setprecision(4) << targ[i]->x() << std::setw(15) << std::setprecision(4) << targ[i]->y() << std::setw(15) << std::setprecision(4) << targ[i]->z() << std::endl;
        }
        std::cout << "----------------------------------------" << std::endl;
    }

    inline void printMatchedPoints(const std::vector<Eigen::Vector3d> &src, const std::vector<Eigen::Vector3d> &targ)
    {
        std::cout << "Matched Points" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << std::setw(15) << "Source X" << std::setw(15) << "Source Y" << std::setw(15) << "Source Z";
        std::cout << std::setw(15) << "Target X" << std::setw(15) << "Target Y" << std::setw(15) << "Target Z" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        for (int i = 0; i < src.size(); ++i)
        {
            std::cout << std::setw(15) << std::setprecision(4) << src[i][0] << std::setw(15) << std::setprecision(4) << src[i][1] << std::setw(15) << std::setprecision(4) << src[i][2];
            std::cout << std::setw(15) << std::setprecision(4) << targ[i][0] << std::setw(15) << std::setprecision(4) << targ[i][1] << std::setw(15) << std::setprecision(4) << targ[i][2] << std::endl;
        }
        std::cout << "----------------------------------------" << std::endl;
    }

    inline void Points(const std::vector<Eigen::Vector3d> &src)
    {
        std::cout << "Tracker points" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << std::setw(15) << "Source X" << std::setw(15) << "Source Y" << std::setw(15) << "Source Z" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        for (int i = 0; i < src.size(); ++i)
        {
            std::cout << std::setw(15) << std::setprecision(4) << src[i][0] << std::setw(15) << std::setprecision(4) << src[i][1] << std::setw(15) << std::setprecision(4) << src[i][2] << std::endl;
        }
        std::cout << "----------------------------------------" << std::endl;
    }

    inline void Points(const std::vector<utils::Point::Ptr> &src)
    {
        std::cout << "Tracker points" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << std::setw(15) << "Source X" << std::setw(15) << "Source Y" << std::setw(15) << "Source Z" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        for (int i = 0; i < src.size(); ++i)
        {
            std::cout << std::setw(15) << std::setprecision(4) << src[i]->x() << std::setw(15) << std::setprecision(4) << src[i]->y() << std::setw(15) << std::setprecision(4) << src[i]->z() << std::endl;
        }
        std::cout << "----------------------------------------" << std::endl;
    }

    inline void Points(const std::vector<utils::Point> &src)
    {
        std::cout << "Tracker points" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << std::setw(15) << "Source X" << std::setw(15) << "Source Y" << std::setw(15) << "Source Z" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        for (int i = 0; i < src.size(); ++i)
        {
            std::cout << std::setw(15) << std::setprecision(4) << src[i].x() << std::setw(15) << std::setprecision(4) << src[i].y() << std::setw(15) << std::setprecision(4) << src[i].z() << std::endl;
        }
        std::cout << "----------------------------------------" << std::endl;
    }

}
#endif