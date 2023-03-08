#ifndef HASH_MAP_TEST
#define HASH_MAP_TEST

#include "limu/sensors/lidar/helpers/voxel_hash_map.hpp"
#include "limu/utils/types.hpp"
#include <cassert>

void basic_test()
{
    // Create some test points
    utils::Vec3dVector points = {
        {1.0, 2.0, 3.0},
        {2.0, 3.0, 4.0},
        {3.0, 4.0, 5.0},
        {4.0, 5.0, 6.0},
        {5.0, 6.0, 7.0},
        {6.0, 7.0, 8.0},
        {7.0, 8.0, 9.0},
        {8.0, 9.0, 10.0},
        {9.0, 10.0, 11.0},
        {10.0, 11.0, 12.0}};

    // Insert the points into the map
    lidar::VoxelHashMap map_data(1, 1, 200);
    map_data.insert_points(points);

    // Verify that the points have been correctly inserted
    auto &map = map_data.map;
    assert(map.size() == 10);
    for (const auto &entry : map)
    {
        const auto &voxel = std::get<0>(entry);
        const auto &block = std::get<1>(entry);
        assert(block.get_points()->size() == 1);
        const auto &point = (*block.get_points())[0];
        std::cout << "\t(" << point->x() << "," << point->y() << "," << point->z() << ")" << std::endl;

        // assert((*block.get_points())[0]->isApprox(points[voxel(0)]));
    }

    // Print out the map for inspection
    for (const auto &entry : map)
    {
        const auto &voxel = std::get<0>(entry);
        const auto &block = std::get<1>(entry);
        std::cout << "Voxel " << voxel(0) << "," << voxel(1) << "," << voxel(2) << ": " << std::endl;
        for (const auto &point : *block.get_points())
        {
            std::cout << "\t(" << point->x() << "," << point->y() << "," << point->z() << ")" << std::endl;
        }
    }
}
void test_insert_points()
{
    // Create some test points
    std::vector<utils::Vec3d> points;
    for (double x = -10; x <= 10; x += 1.1)
    {
        for (double y = -10; y <= 10; y += 1.1)
        {
            for (double z = -10; z <= 10; z += 1.1)
            {
                points.emplace_back(x, y, z);
            }
        }
    }

    // Add some more points that fall on specific voxels
    points.emplace_back(-0.5, -0.5, -0.5);
    points.emplace_back(-0.5, -0.5, 0.5);
    points.emplace_back(-0.5, 0.5, -0.5);
    points.emplace_back(-0.5, 0.5, 0.5);
    points.emplace_back(0.5, -0.5, -0.5);
    points.emplace_back(0.5, -0.5, 0.5);
    points.emplace_back(0.5, 0.5, -0.5);
    points.emplace_back(0.5, 0.5, 0.5);
    points.emplace_back(1.6, 0.5, -1.1);
    points.emplace_back(-1.6, -0.5, 1.1);

    // Insert the points into the map
    lidar::VoxelHashMap map_data(1, 1, 200);
    map_data.insert_points(points);

    // Verify that the points have been correctly inserted
    for (const auto &kv : map_data.map)
    {
        const auto &voxel = std::get<0>(kv);
        const auto &voxel_block = std::get<1>(kv);
        const auto &points_ = *(voxel_block.get_points());

        std::cout << "Voxel: " << voxel.transpose() << "\nPoints: ";
        for (const auto &point : points_)
        {
            std::cout << "(" << point->x() << ", " << point->y() << ", " << point->z() << ") ";
        }
        std::cout << std::endl;
    }

    std::cout << "End of testing." << std::endl;
}

void test_closest_neighbor()
{
    // Create a VoxelHashMap
    // utils::Vec3d vox_size(1, 1, 1);
    lidar::VoxelHashMap map(1, 1, 200);

    // Create some points to add to the map
    utils::Vec3d p1(0, 0, 0);
    utils::Vec3d p2(2, 2, 2);
    utils::Vec3d p3(4, 4, 4);

    // Add the points to the map
    map.insert_points({p1, p2, p3});

    // Test with a point that is already in the map
    std::weak_ptr<utils::Vec3d> result1 = map.get_closest_neighbour(p1);
    assert(result1.lock() != nullptr);
    std::cout << "result1: " << (p1 - *result1.lock()).norm() << std::endl;
    assert((p1 - *result1.lock()).norm() == 0);

    // Test with a point that is not in the map
    utils::Vec3d p4(1, 1, 1);
    std::weak_ptr<utils::Vec3d> result2 = map.get_closest_neighbour(p4);
    assert(result2.lock() != nullptr);
    std::cout << "result2: " << (p4 - *result2.lock()).norm() << std::endl;
    assert(std::sqrt(3) == (p4 - *result2.lock()).norm());
}

void test_correspondences()
{
    lidar::VoxelHashMap map(1, 0.2, 1000);
    const double max_correspondance = 0.2;
    const int num_points = 1000;
    std::vector<utils::Vec3d> points(num_points);
    for (int i = 0; i < num_points; i++)
    {
        points[i] = utils::Vec3d(1.0 * rand() / RAND_MAX, 1.0 * rand() / RAND_MAX, 1.0 * rand() / RAND_MAX);
    }

    map.insert_points(points);
    std::cout << "points inserted " << std::endl;

    // compute correspondences
    auto correspondences = map.get_correspondences(points, max_correspondance);

    int num_correspondences = std::min(std::get<0>(correspondences).size(), std::get<1>(correspondences).size());
    std::cout << "correspondences complete. Number: " << num_correspondences << std::endl;
    // check that correspondences are close enough
    for (int i = 0; i < num_correspondences; i++)
    {
        double distance = (std::get<0>(correspondences)[i] - std::get<1>(correspondences)[i]).norm();
        if (distance > max_correspondance)
        {
            std::cout << "Test failed: correspondence distance too large\n";
            return;
        }
    }

    // check number of correspondences
    if (num_correspondences != num_points)
    {
        std::cout << "Test failed: incorrect number of correspondences\n";
        return;
    }

    std::cout << "All tests passed\n";
}

void test_correspondences2(int size)
{
    // Generate random points
    const int n_points = 100000;
    const double max_distance = 0.5;
    std::vector<utils::Vec3d> points(n_points);
    for (auto &p : points)
    {
        p = utils::Vec3d::Random() * max_distance;
    }

    // Compute correspondences
    const double max_correspondence = 0.1;
    lidar::VoxelHashMap map(1, 1, size);
    map.insert_points(points);
    const auto correspondences = map.get_correspondences(points, max_correspondence);

    // Verify that each point has at least one correspondence
    for (int i = 0; i < n_points; ++i)
    {
        const auto &src = points[i];
        bool found_correspondence = false;
        for (int j = 0; j < std::get<0>(correspondences).size(); ++j)
        {
            const auto &tgt = std::get<1>(correspondences)[j];
            if ((tgt - src).squaredNorm() < max_correspondence * max_correspondence)
            {
                found_correspondence = true;
                break;
            }
        }
        if (!found_correspondence)
        {
            std::cerr << "Point " << i << " has no correspondences" << std::endl;
            return;
        }
    }
    std::cout << "All points have at least one correspondence" << std::endl;
}

void test_remove_points_from_far(double distance = 5.0)
{
    lidar::VoxelHashMap map(0.1, distance, 100);
    std::vector<utils::Vec3d> num_points;
    for (size_t i = 0; i < 1000; ++i)
    {
        const auto pt = utils::Vec3d::Random() * 100;
        num_points.emplace_back(pt);
    }

    map.insert_points(num_points);
    // Set an origin point from which to remove points that are too far
    const auto origin = utils::Vec3d::Zero();

    // Call the remove_points_from_far method
    map.remove_points_from_far(origin);

    // Check that the removed points are indeed farther than max_dist from the origin
    for (auto it = map.map.begin(); it != map.map.end(); it++)
    {
        const auto &vox_block = it->second;
        for (const auto &pt : *vox_block.points)
        {
            const auto dist = (*pt - origin).norm();
            if (dist > distance)
            {

                std::cout << "Test failed: Point: ";
                std::cout << "(" << pt->x() << ", " << pt->y() << ", " << pt->z() << ") ";
                std::cout << " is too far from origin\n";
                return;
            }
        }
    }

    std::cout << "Test passed: All removed points are farther than max_dist from origin\n";
}

void run_tests()
{
    ROS_WARN("Testing insertions");
    basic_test();
    ROS_WARN("\nTesting insertions");
    test_insert_points();
    ROS_WARN("\nTesting closest");
    test_closest_neighbor();
    ROS_WARN("\nTesting correspondences");
    test_correspondences();
    ROS_WARN("\nTesting correspondences 2 - size 100");
    test_correspondences2(100);
    ROS_WARN("\nTesting correspondences 3 - size 50");
    test_correspondences2(50);
    ROS_WARN("\nTesting correspondences 4 - size 10");
    test_correspondences2(10);
    ROS_WARN("\nTesting correspondences 5 - size 60");
    test_correspondences2(60);
    ROS_WARN("\nTesting correspondences 6 - size 1000");
    test_correspondences2(1000);
    ROS_WARN("\nRemove points from far");
    test_remove_points_from_far(5.0);

    ROS_WARN("\nEnd of tests.");
}
#endif
