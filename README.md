# Lidar-Imu-Slam
Lidar-Imu-Slam is an algorithm that combines Lidar and Imu sensors to localize and generate a map of the environment. The system utilizes a combination of ICP and Kalman Filtering process in the odometry phase, and Non-Linear optimization for optimizing the Map.

## Completed Tasks
- Reworked to the Kiss-Icp VoxelHashMap:
  - 3d points use smart pointer access.
  - TBB library used to parallelize operations inserting, searching, and updating functions.
- Process Imu and Lidar in a unified manner without having to deskew the Lidar frame:
  - Basic tests show updates in estimated pose speed.
  - Introduced outlier pipeline during the lidar preprocessing step. It will be tested upon completion to see if it's beneficial. Current signs on lidar only method looks promising.

- Included multiplicative bias to state to help correct accelerometer error. Similar to HYBVIO paper. 

- Following the Point-LIO paper, I'll implement an Iterative Kalman filter. The system will predict the imu readings as output values, which should help filter noise and allow for better estimates.
  - Differences:
    - Replace point-to-plane estimation with Kiss-ICP point-to-point method. This should reduce the required computation.
    - Points will be processed by timestamps. Point-to-Point Icp is a little sensitive
    - Bigger state variables.

## Targets
- Verify F_x and F_w matrices
- Include Lidar_IMU_Init package
- Finalize running algorithm