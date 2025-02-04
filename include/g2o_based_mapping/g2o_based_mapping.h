#ifndef G2O_BASED_MAPPING_H
#define G2O_BASED_MAPPING_H

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>

#include <ros/ros.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/OccupancyGrid.h>

#include <sensor_msgs/LaserScan.h>
#include <boost/thread.hpp>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/Int8.h>

#include <Eigen/Dense>
#include <Eigen/LU>
#include "g2o/core/sparse_optimizer.h"
#include "g2o/types/slam2d/se2.h"
#include "g2o/types/data/raw_laser.h"

class G2oBasedMapping
{
public:
    G2oBasedMapping(ros::NodeHandle n);

    void updateOdometry(nav_msgs::Odometry odometry);
    void updateLaserScan(sensor_msgs::LaserScan landmarks);
    void init(double x, double y, double theta);
    void odometryCallback(const nav_msgs::Odometry::ConstPtr& msg);
    void laserscanCallback(const sensor_msgs::LaserScan::ConstPtr& msg);
    void initialposeCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg);
    void publishMapThread();
    
private:

    boost::mutex* data_mutex_;
    nav_msgs::Odometry last_odometry;
    tf::TransformBroadcaster pose_tf_broadcaster;
    tf::TransformListener* tf_listener_;

    ros::NodeHandle nh_;

    Eigen::MatrixXd x;
    Eigen::MatrixXd last_add_pose;
    sensor_msgs::LaserScan last_scan_added;

    std::vector<int> laser_vertex_ids_;                 // List to store laser vertex IDs
    std::map<int, sensor_msgs::LaserScan> laser_scans_; // Map laser_scans_ to store laser scans by vertex ID


    Eigen::Matrix3d odom_noise_;
    Eigen::Matrix3d laser_noise_;
    Eigen::Matrix2d landmark_noise_;

    g2o::SparseOptimizer graph_;
    g2o::HyperGraph::VertexSet vertex_set_;
    g2o::HyperGraph::EdgeSet edge_set_;
    g2o::LaserParameters* laser_params_;

    bool first_vert_;
    bool valid_;
    bool first_opt_;
    std::vector<int> seen_landmarks_;
    std::vector<int> robot_pose_ids_;
    std::vector<std::pair<int, int> > robot_landmark_edge_ids_;
    std::vector<std::pair<int, int> > laser_edge_ids_;
    int min_to_optimize_;
    int last_id_;
    int vertex_c = 0;
    bool robot_pose_set;
    bool reset_;
    nav_msgs::OccupancyGrid graph_map;

    // adds odom vertex
    // takes global position and a unique id, first only needed in init
    void addOdomVertex(double x, double y, double theta, int id, bool first = false);
    
    // adds laser vertex
    // takes global position, a laser scan and a unique id, first only needed in init
    void addLaserVertex(double x, double y, double theta, sensor_msgs::LaserScan scan, int id, bool first = false);

    // adds edge between the vertices based on a 2D transformation
    void addLaserEdge(int id1, int id2, double x, double y, double yaw, Eigen::Matrix3d noise);

    // adds edge between the vertices
    // takes two ids of odom vertices
    void addOdomEdge(int id1, int id2);
    
    // adds landmark vertex
    // takes global postion of the landmark and the corresponding id
    void addLandmarkVertex(double x, double y, int id);
  
    // adds landmark edge
    // takes odom vertex and landmark id and the global position of the landmark
    void addLandmarkEdge(int id1, int id2, double x, double y);
    
    // optimize graph
    void optimizeGraph();
    
    // resets the robot position to the odom vertex with the given id
    // takes id of odom vertex 
    void setRobotToVertex(int id);

    void publishMap();

    // helper to bridge g2o with ros
    sensor_msgs::LaserScan rawLasertoLaserScanMsg(g2o::RawLaser rawlaser);

    void visualizeLaserScans();
    void visualizeRobotPoses();
    void visualizeLandmarks();
    void visualizeEdges();
    void visualizeOldLandmarks();
    void updateLocalization();
    
    
    //std::vector<Eigen::Vector2d> laserScanToPoints(const sensor_msgs::LaserScan& scan);
    Eigen::MatrixXd laserScanToPoints(const sensor_msgs::LaserScan& scan);
    Eigen::Vector2d computeCentroid(const Eigen::MatrixXd& points);
    Eigen::Matrix3d displacement(Eigen::MatrixXd source_points, Eigen::MatrixXd target_points);
    float dist(const Eigen::Vector2d &source, const Eigen::Vector2d &target);
    Eigen::Vector3d ICP(const sensor_msgs::LaserScan& last_scan_added, const sensor_msgs::LaserScan& scan);
    void detectLoopClosureICP(const sensor_msgs::LaserScan& current_scan, int current_vertex_id);
    sensor_msgs::LaserScan getScan(int vertex_id);
    sensor_msgs::LaserScan ScanToBaseFrame(const sensor_msgs::LaserScan& scan, const tf::StampedTransform& transform);

    //void detect3SgimaMethod(int current_id);

    
};

template<typename to, typename from>to arg_cast(from const &x) {
  std::stringstream os;
  to ret;

  os << x;
  os >> ret;

  return ret;
}

#endif // G2O_BASED_MAPPING_H
