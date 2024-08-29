# Launching your graph based mapping for Ass2

- Launch Gazebo Simulation:

`roslaunch robot_gazebo platform_building.launch full_laser_fov:=true tf_base_link:=base_link_g2o`

You can set the desired tf tree branch for testing your implementation by setting the `tf_base_link` argument: `base_link` sets the robot onto the odometry tf (default), `base_link_real` sets the robot onto the ground truth, and `base_link_g2o` sets the odom onto your estimated position from your graph based mapping assignment. 

You can test your implementation with two different laser scanners by setting the `full_laser_fov` argument: if set to `true`, a 360° laser is used for the assignment. If set to `false`, a standard 180° laser is used.
  

- Start your mapping implementation:

`rosrun g2o_based_mapping g2o_based_mapping`

- Launch RViz for ROS visualization:

`roslaunch robot_gazebo rviz.launch`

- Launch Keyboard Control for moving around ([Check here for control keys](http://wiki.ros.org/teleop_twist_keyboard)):

`roslaunch robot_control teleop_keyboard.launch`

