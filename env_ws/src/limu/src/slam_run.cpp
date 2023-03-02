int main(int argc, char **argv)
{
    ros::init(argc, argv, "slam_run");
    ros::NodeHandle nh;
    SensorProcess lp(nh);
    lp.program_run();

    return 0;
}