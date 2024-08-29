#include <g2o_based_mapping/g2o_based_mapping.h>

#include <Eigen/StdVector>
#include <Eigen/Eigen>
#include <numeric>
#include <thread>

#include "g2o/types/slam2d/vertex_point_xy.h"
#include "g2o/types/slam2d/edge_se2_pointxy.h"
#include "g2o/types/slam2d/vertex_se2.h"
#include "g2o/types/slam2d/edge_se2.h"

#include "g2o/core/block_solver.h"
#include "g2o/core/factory.h"
#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/solvers/csparse/linear_solver_csparse.h"

#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/core/hyper_dijkstra.h>
#include <g2o/core/batch_stats.h>
#include <g2o/core/estimate_propagator.h>

#include "nabo/nabo.h"
#include <fstream>

using namespace std;
using namespace ros;



G2oBasedMapping::G2oBasedMapping(ros::NodeHandle n){

    data_mutex_ = new boost::mutex();
    tf_listener_ = new tf::TransformListener(n);
    nh_ = n;

    x = Eigen::MatrixXd::Zero(3,1);
    last_add_pose = Eigen::MatrixXd::Zero(3,1);

    typedef g2o::LinearSolverCSparse<g2o::BlockSolverX::PoseMatrixType> SlamLinearSolver;

    auto linearSolver = g2o::make_unique<SlamLinearSolver>();
    linearSolver->setBlockOrdering(false);
    //std::unique_ptr<g2o::Solver> solver(new g2o::BlockSolverX(linearSolver));
    g2o::OptimizationAlgorithmGaussNewton *algorithm = new g2o::OptimizationAlgorithmGaussNewton(g2o::make_unique<g2o::BlockSolverX>(std::move(linearSolver)));
    graph_.setAlgorithm(algorithm);

    // Use standard dimensions for the map publishing, as we already know the actual map size. 
    // In real unknown environments, the map size has to be dynamic!
    graph_map.header.frame_id = "map";
    graph_map.header.stamp = ros::Time::now();
    graph_map.info.map_load_time = ros::Time::now();
    graph_map.info.resolution = 0.05;
    graph_map.info.width = 560;
    graph_map.info.height = 360;
    geometry_msgs::Pose origin;
    origin.position.x = -19.5;
    origin.position.y = -9.0;
    origin.orientation.w = 1.0;
    graph_map.info.origin = origin;

    laser_params_ = 0;

    init(10,28,0);

    // TODO
    // find appropriate parameters
    // Putting a high noise regarding the rotation values since the alg is more accurate 
    // regarding the x,y movements and less performing when it comes to yaw values
    double x_noise = 0.1;
    double y_noise = 0.1;
    double rot_noise = 0.01;    //rad
    double landmark_x_noise = 1;
    double landmark_y_noise = 1;

    odom_noise_.fill(0.);
    odom_noise_(0, 0) = 1/(x_noise*x_noise);
    odom_noise_(1, 1) = 1/(y_noise*y_noise);
    odom_noise_(2, 2) = 1/(rot_noise*rot_noise);

    laser_noise_.fill(0.);
    laser_noise_(0, 0) = 0.1;
    laser_noise_(1, 1) = 0.1;
    laser_noise_(2, 2) = 0.01;

    landmark_noise_.fill(0.);
    landmark_noise_(0, 0) = 1/(landmark_x_noise*landmark_x_noise);
    landmark_noise_(1, 1) = 1/(landmark_y_noise*landmark_y_noise);
}

void G2oBasedMapping::updateOdometry(nav_msgs::Odometry odometry){

    if (reset_){
        last_odometry = odometry;
        updateLocalization();
        
        init(odometry.pose.pose.position.x, odometry.pose.pose.position.y, tf::getYaw(odometry.pose.pose.orientation));
        
        reset_ = false;
        valid_ = false;
        return;
    }

    // 1.1 - Implement Odometry Update
    // Extract the current pose from the odometry message
    double curr_x = odometry.pose.pose.position.x;
    double curr_y = odometry.pose.pose.position.y;
    double curr_t = tf::getYaw(odometry.pose.pose.orientation);

    // Extract the last pose from the last odometry message
    double last_x = last_odometry.pose.pose.position.x;
    double last_y = last_odometry.pose.pose.position.y;
    double last_t = tf::getYaw(last_odometry.pose.pose.orientation);

    // Compute the difference in position and orientation
    double delta_x = curr_x - last_x;
    double delta_y = curr_y - last_y;
    double delta_t = curr_t - last_t;

    // Keep track of the odometry updates to the robot position

    // Normalize the orientation to keep it within the range of -pi to pi.
    delta_t = atan2(sin(delta_t), cos(delta_t));
    
    // Store or use the deltas for graph construction
    // Compute the odometry increment in the robot's local frame
    double delta_trans = sqrt(delta_x * delta_x + delta_y * delta_y);
    double delta_rot1  = atan2(delta_y, delta_x) - last_t;
    double delta_rot2  = delta_t - delta_rot1;

    // Normalize the rotation increments
    delta_rot1 = atan2(sin(delta_rot1), cos(delta_rot1));
    delta_rot2 = atan2(sin(delta_rot2), cos(delta_rot2));

    // Update the pose estimate (x)
    x(0) += delta_trans * cos(x(2) + delta_rot1);
    x(1) += delta_trans * sin(x(2) + delta_rot1);
    
    double angle = atan2(sin(delta_rot1 + delta_rot2), cos(delta_rot1 + delta_rot2));
    x(2) += angle;
    
    // Keep track of the odometry updates to the robot position
    /*
    x(0) += delta_x;
    x(1) += delta_y;
    x(2) += delta_t;
    */

    // Normalize the updated orientation to keep it within the range of -pi to pi.
    x(2) = atan2(sin(x(2)), cos(x(2)));
    //cout << x << endl;

    // global variable last_odometry contains the last odometry position estimation (ROS Odometry Messasge)
    // local variable odometry contains the current odometry position estimation (ROS Odometry Messasge)
    // local variable x holds your position (Eigen vector of size 3 [x,y,theta])
    last_odometry = odometry;
}

// Convert laser scan to 2D points
Eigen::MatrixXd G2oBasedMapping::laserScanToPoints(const sensor_msgs::LaserScan& scan) {

    // Create an Eigen::MatrixXd with 2 cols and as many rows as valid points
    Eigen::MatrixXd points(scan.ranges.size(), 2);
    
    
    for (int i = 0; i < scan.ranges.size(); ++i) {
        double range = scan.ranges[i];
        if (range < scan.range_max && range > scan.range_min) {
            double angle = scan.angle_min + i * scan.angle_increment;
            points(i, 0) = range * cos(angle);  // Store the point at row i, col 0
            points(i, 1) = range * sin(angle);  // Store the point at row i, col 1
        }else{
            points(i, 0) = std::numeric_limits<double>::quiet_NaN();
            points(i, 1) = std::numeric_limits<double>::quiet_NaN();
        }
    }
    return points;


    // Consider only one point each 'point_skip', to speed up computation
    /*int point_skip = 4;
    for (int i = 0; i < scan.ranges.size(); i+=point_skip) {
        double range = scan.ranges[i];
        if (range < scan.range_max && range > scan.range_min) {
            double angle = scan.angle_min + i * scan.angle_increment;
            points(i, 0) = range * cos(angle);  // Store the point at row i, col 0
            points(i, 1) = range * sin(angle);  // Store the point at row i, col 1
        }   
    }
    return points;*/
}

// Returns the centroid of a set of points
Eigen::Vector2d G2oBasedMapping::computeCentroid(const Eigen::MatrixXd& points) {
    Eigen::Vector2d centroid(0, 0);
    int it = points.rows();

    for (int i=0; i<it; i++) {
        // Retreive a syb-matrix of size 1x2 starting from row i and col 0
        centroid += points.block<1,2>(i,0);
    }
    centroid /= it;
    return centroid;
}

// Compute the Eucledian distance between 2 points
float G2oBasedMapping::dist(const Eigen::Vector2d &source, const Eigen::Vector2d &target) {
    return sqrt((source[0] - target[0]) * (source[0] - target[0]) + (source[1] - target[1]) * (source[1] - target[1]));
}

// Compute 1 iteration of the ICP algorithm, compute the displacement between source and target set
// returning the translation/rotation values
Eigen::Matrix3d G2oBasedMapping::displacement(Eigen::MatrixXd source_points, 
                                              Eigen::MatrixXd target_points){

    Eigen::Matrix3d T = Eigen::MatrixXd::Identity(3,3);
    Eigen::Vector2d source_centroid(0,0);
    Eigen::Vector2d target_centroid(0,0);

    Eigen::MatrixXd s_mat = source_points;
    Eigen::MatrixXd t_mat = target_points;
    int it = s_mat.rows();

    // Compute the centroids of both sets
    source_centroid = computeCentroid(source_points);   // mu_x
    target_centroid = computeCentroid(target_points);   // mu_p

    // Subtract the centroids to both sets
    for(int i=0; i<it; ++i){
        s_mat.block<1,2>(i,0) = source_points.block<1,2>(i,0) - source_centroid.transpose();    // X' = {x__i'}
        t_mat.block<1,2>(i,0) = target_points.block<1,2>(i,0) - target_centroid.transpose();    // P' = {p_i'}
    }

    // Compute cross-covariance matrix Mat 'W'
    Eigen::MatrixXd W = s_mat.transpose()*t_mat;    // W = sum_i^n(x__i' * p_i'^T)
    Eigen::MatrixXd U, V, R;
    Eigen::VectorXd t;

    Eigen::JacobiSVD<Eigen::Matrix2d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    U = svd.matrixU();
    V = svd.matrixV();

    R = V * U.transpose();      // R = V * U^T
    if(R.determinant() < 0){
        V.col(1) *= -1;
        R = V * U.transpose();
    }

    t = target_centroid - R*source_centroid;    //T = mu_x - R*mu_p

    T.block<2,2>(0,0) = R;
    T.block<2,1>(0,2) = t;

    return T;
}

// Compute the ICP 
Eigen::Vector3d G2oBasedMapping::ICP(const sensor_msgs::LaserScan& last_scan_added, const sensor_msgs::LaserScan& scan){
    cout << "Computing ICP displacement" << endl;
    int maxIt     = 30;
    int tolerance = 0.0002;

    // Convert laser scan to 2D points
    Eigen::MatrixXd source_points = laserScanToPoints(last_scan_added);
    Eigen::MatrixXd target_points = laserScanToPoints(scan);

    int it = source_points.rows();
    double last_err = 0, meanE = 0;
    // Allocating space
    vector<int>    idxs;
    vector<float> dists;
    Eigen::MatrixXd source  = Eigen::MatrixXd::Ones(3, it);
    Eigen::MatrixXd source2 = Eigen::MatrixXd::Ones(2, it);
    Eigen::MatrixXd target  = source;
    Eigen::Matrix3d T       = Eigen::MatrixXd::Identity(3,3);
    Eigen::MatrixXd targ_o  = Eigen::MatrixXd::Ones(2, it);  //target set ordered

    // Computing the matrices in order to use the Nabo lib
    for(int i=0; i<it; i++){
        source.block<2, 1>(0,i) = source_points.block<1,2>(i,0).transpose();
        source2.block<2,1>(0,i) = source_points.block<1,2>(i,0).transpose();
        target.block<2, 1>(0,i) = target_points.block<1,2>(i,0).transpose();
    }
    // Nabo implementation
    // Build the k-d tree index for the target points
    Nabo::NNSearchD* nns = Nabo::NNSearchD::createKDTreeLinearHeap(target);
    Eigen::VectorXi indices(1);
    Eigen::VectorXd distances(1);

    // Compute maxIt iteration to find the optimal displacement 
    for(int i=0; i<maxIt; ++i){

        Eigen::MatrixXd set = source2.transpose();
        int it_src = set.rows();
        int it_trg = target.cols();
        // IMPROVED implementation using kd-tree method 
          for(int i=0; i<it_src; ++i){
            // Perform the nearest neighbor search for the current source point
            nns->knn(set.row(i).transpose(), indices, distances, 1);
            idxs.push_back(indices[0]);
            dists.push_back(sqrt(distances[0])); 
        }

        // Compute the set of the distances and indices of the closest neighbor
        // for each point storing its distance and the index (brute force)
        /*for(int ii=0; ii<it_src; ++ii){
            Eigen::Vector2d source_vec = set.row(ii).transpose();
            float min_dist = numeric_limits<float>::max();
            int min_idx    = 0;

            // For a point in the source set find the closest in the target set
            for(int j=0; j<it_trg; j++){
                Eigen::Vector2d target_vec = target.block<2,1>(0,j);
                float d = dist(source_vec, target_vec);

                if(d < min_dist){
                    min_dist = d;
                    min_idx = j;
                }
            }
            // Save the info in the proper arrays
            dists.push_back(min_dist);
            idxs.push_back(min_idx); 
        }*/

        // Order the target points according to the previous idx set
        for(int j=0; j<dists.size(); ++j){
            targ_o.block<2,1>(0,j) = target.block<2,1>(0,idxs[j]);
        }
        
        // Compute the displacement 
        T = displacement(source2.transpose(), targ_o.transpose());

        // Apply the transformation
        source = T * source;
        for(int j=0; j<it; ++j){
            source2.block<2,1>(0,j) = source.block<2,1>(0,j);
        }

        // Compute iterations until the error drops below a treshold
        // Compute the squared Euclidean error
        meanE = 0.0;
        for (double dist : dists) {
            meanE += dist * dist;
        }
        meanE /= dists.size();

        if (abs(last_err - meanE) < tolerance) break;

        last_err = meanE;
        dists.clear();
        idxs.clear();
    }
    // Compute the optimal T
    T = displacement(source_points, source2.transpose());

    Eigen::Matrix2d R = T.block<2,2>(0,0);
    Eigen::Vector2d t = T.block<2,1>(0,2);
    
    // Extract the angle from the rotation matrix and normalize
    double angle = atan2(R(1, 0), R(0, 0));

    //cout << "Translation: " << t[0] << " " << t[1] <<  "\nRot:\t " << angle << endl;
    return Eigen::Vector3d (t[0], t[1], angle);
}


// The algorithm selects the candidate nodes in the past as the
// ones whose 3σ marginal covariances contains the current
// robot pose. These covariances can be obtained as the
// diagonal blocks of the inverse of a reduced Hessian H red .
// H red is obtained from H by removing rows and the
// columns of the newly inserted robot pose. H red is the
// information matrix of all the
/*void G2oBasedMapping::detect3SgimaMethod(int current_id) {
    cout << "Computing detect3SgimaMethod" << endl;

    // 3 DOF (x, y, theta)
    int dim = graph_.vertices().size() * 3; 
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dim, dim);

    int reduced_dim = dim - 3;  // 3 DOF
    Eigen::MatrixXd H_red = Eigen::MatrixXd::Zero(reduced_dim, reduced_dim);

    for (auto& edge : graph_.edges()) {
        auto* e = dynamic_cast<g2o::EdgeSE2*>(edge);
        if (!e) continue; // Ensure edge is of type EdgeSE2

        // Compute error
        e->computeError();

        // Retrieve the error vector
        Eigen::VectorXd e_ij = Eigen::Map<Eigen::VectorXd>(e->errorData(), e->dimension());

        // Retrieve the vertices
        g2o::OptimizableGraph::Vertex* vi = static_cast<g2o::OptimizableGraph::Vertex*>(e->vertex(0));
        g2o::OptimizableGraph::Vertex* vj = static_cast<g2o::OptimizableGraph::Vertex*>(e->vertex(1));

        // Ensure the vertices are of type VertexSE2
        auto* vi_se2 = dynamic_cast<g2o::VertexSE2*>(vi);
        auto* vj_se2 = dynamic_cast<g2o::VertexSE2*>(vj);
        if (!vi_se2 || !vj_se2) continue;

        // Retrieve the information matrix (omega)
        Eigen::Matrix3d omega = e->information();

        // Compute Jacobians of the error function
        e->linearizeOplus();    // THIS GIVES SEGMENTATION FAULT
        Eigen::MatrixXd J_i = e->jacobianOplusXi(); // retrieve after computation
        Eigen::MatrixXd J_j = e->jacobianOplusXj(); // retrieve after computation

        // Indices for the vertices in the Hessian matrix
        int i_idx = vi->hessianIndex() * 3;
        int j_idx = vj->hessianIndex() * 3;

        // Hessian block updates
        // compute the contribution of this constraint to the linear system
        Eigen::Matrix<double, 3, 3> H_ii = J_i.transpose() * omega * J_i;
        Eigen::Matrix<double, 3, 3> H_ij = J_i.transpose() * omega * J_j;
        Eigen::Matrix<double, 3, 3> H_jj = J_j.transpose() * omega * J_j;

        // Add contributions to the global Hessian matrix
        H.block<3, 3>(i_idx, i_idx) += H_ii;
        H.block<3, 3>(i_idx, j_idx) += H_ij;
        H.block<3, 3>(j_idx, i_idx) += H_ij.transpose();        
        H.block<3, 3>(j_idx, j_idx) += H_jj;

        //cout << "Error: " << e_ij.transpose() << endl;
        //cout << "J - i: \n" << J_i << endl;
        //cout << "J - j: \n" << J_j << endl;

        // compute the coefficient vector
        Eigen::VectorXd& b_i = J_i.transpose() * omega * e_ij;
        Eigen::VectorXd& b_j = J_j.transpose() * omega * e_ij;

        // Compute the reduced Hessian matrix
        int dim = H.rows();
        int reduced_dim = dim - 3;  // 3 DOF
        Eigen::MatrixXd H_red = Eigen::MatrixXd::Zero(reduced_dim, reduced_dim);

        // Copy blocks except for the current pose block - (not tried) 
        for (int i = 0; i < current_id; ++i) {
            for (int j = 0; j < current_id; ++j) {
                H_red.block<3, 3>(i * 3, j * 3) = H.block<3, 3>(i * 3, j * 3);
            }
        }
        for (int i = current_id + 1; i < dim / 3; ++i) {
            for (int j = current_id + 1; j < dim / 3; ++j) {
                H_red.block<3, 3>((i - 1) * 3, (j - 1) * 3) = H.block<3, 3>(i * 3, j * 3);
            }
        }
    }

    // H_red = reduced Hessian
    Eigen::MatrixXd sigma = H_red.inverse();

    // Finds the best matching vertex using the 3σ covariance criterion.
    g2o::OptimizableGraph::Vertex* best_match = nullptr;   
    // Retrieeve the current pose relative to the current_id
    g2o::OptimizableGraph::Vertex* currentV   = graph_.vertex(current_id); 
    vector<double>  current_pose;
    currentV->getEstimateData(current_pose);
    Eigen::VectorXd current_pose = static_cast<Eigen::VectorXd>(current_pose);

    Eigen::Vector3d displacement;
    double min_d = 0.2; // TRESH

    // Loop trought the vertices that are within the 3sigma 
    for (auto& v : graph_.vertices()) {
        auto* vertex = static_cast<g2o::OptimizableGraph::Vertex*>(v.second);

        // Retrieving the vertex pose to be able to compare
        vector<double> vertex_pose;
        vertex->getEstimateData(vertex_pose);
        Eigen::VectorXd vertex_pose = static_cast<Eigen::VectorXd>(vertex_pose);
        
        // Compute the difference bwt the 2 poses
        Eigen::VectorXd diff = vertex_pose - current_pose;

        // Computing the Maahalanobis distance
        double mahalanobis_dist = diff.transpose() * sigma.inverse() * diff;
        if (mahalanobis_dist <= 9) {    // 3σ corresponds to 9 in squared Mahalanobis distance

            displacement = ICP(getScan(vertex->id()), getScan(current_id));
            if (displacement.norm() < min_d) {
                min_d = displacement.norm();
                best_match = vertex;
            }
        }
    }
    // If best match found, add the loop-closure constraint and optimize the graph
    if (best_match) {
        addLaserEdge(current_id, best_match->id(), displacement(0), displacement(1), displacement(2), laser_noise_);
        optimizeGraph();
    }
}*/

// Implemented method due to above's failure
// Loop trough all the Laser vertices stored, in a reverse order (with the assumption that the loop-closure is more
// likely to happen with one of the latter scans). 
void G2oBasedMapping::detectLoopClosureICP(const sensor_msgs::LaserScan& current_scan, int current_vertex_id) {
    // Treshold for which the best scan is selected (to refine)
    double TRESH_D = 0.4;   // 40cm
    double TRESH_R = 0.3;   // 0.3 rad

    g2o::VertexSE2* best_match_vertex = nullptr;
    // Threshold to find the best-match scan
    double min_d = numeric_limits<double>::max();
    double min_r = numeric_limits<double>::max();

    // Loop through all laser vertices sorted in LIFO logic to find the ICP values that match the condition
    // If the condition are matched, then i found a loop closure
    for (auto it = laser_vertex_ids_.rbegin(); it != laser_vertex_ids_.rend(); ++it) {
        
        int vertex_id = *it;
        if (vertex_id != current_vertex_id) {
            g2o::VertexSE2* vertex = dynamic_cast<g2o::VertexSE2*>(graph_.vertex(vertex_id));
            if (vertex) {   // 
                sensor_msgs::LaserScan stored_scan = getScan(vertex_id);

                // Compute the displacement using ICP
                Eigen::Vector3d d_scans = ICP(current_scan, stored_scan);

                // Check if the displacement is below the threshold
                double d_d = d_scans.head<2>().norm();
                double d_r = fabs(d_scans(2));

                cout << "DD: " << d_d << "  - DR: " << d_r << endl;

                // Checks if the displacement between scans is below tresholds and if that is 
                // better than the previous one found (if 'break' is not used)
                bool cond = d_d < TRESH_D && d_r < TRESH_R && d_d < min_d   && d_r <= min_r;
                if (cond) {
                    min_d = d_d;
                    min_r = d_r;
                    best_match_vertex = vertex;

                    // To speed-up computation (since LIFO logic is implemented), it's possible to 
                    // break after finding one scan that match the tresholds (refine better if 'break' is used)
                    break;  // Remove the break to have a slower, but more accurate implementation
                }
            }
        }
    }
    // If a match is found, add an edge and optimize the graph
    if (best_match_vertex) {
        // Convert laser scan to 2D points
        Eigen::Vector3d d = ICP(current_scan, getScan(best_match_vertex->id()));
        addLaserEdge(current_vertex_id, best_match_vertex->id(), d(0), d(1), d(2), laser_noise_);
        
        // Optimize the graph
        optimizeGraph();
    }
}

// Function to get stored scan for a given vertex id 
sensor_msgs::LaserScan G2oBasedMapping::getScan(int vertex_id) {
    return laser_scans_[vertex_id];
}

// Return a scan msg containing points from scan transformed according to 'transform'
sensor_msgs::LaserScan G2oBasedMapping::ScanToBaseFrame(const sensor_msgs::LaserScan& scan, const tf::StampedTransform& transform){
    
    sensor_msgs::LaserScan tf_scan = scan;
    for(int i=0; i<scan.ranges.size(); ++i){

        // Compute the original point
        double angle = scan.angle_min + i * scan.angle_increment;
        double range = scan.ranges[i];

        if(range >= scan.range_min && range <= scan.range_max){ // If it s an actual measurement

            tf::Vector3 point(range * cos(angle), range * sin(angle), 0.0);
            tf::Vector3 tf_point = transform * point;

            // compute the new values
            double new_range = sqrt(pow(tf_point.x(), 2) + pow(tf_point.y(), 2));
            //double new_angle = atan2(tf_point.y(), tf_point.x());   // Even if the angles stays the same

            // Update
            tf_scan.ranges[i] = new_range;
        }else{  // Set to inf if the scan is not a valid one
            tf_scan.ranges[i] = numeric_limits<float>::infinity();
        }
    }
    return tf_scan;
}


void G2oBasedMapping::updateLaserScan(sensor_msgs::LaserScan scan){

    // Tranformer between 2 frames in order to avoid anomalies, computing calculation
    // on points refering all to the same frame
    tf::StampedTransform transform;
    static tf::TransformListener listener;

    if (!laser_params_ || graph_.vertices().size() == 0){
        // first laser update
        laser_params_ = new g2o::LaserParameters(scan.ranges.size(), scan.angle_min, scan.angle_increment, scan.range_max);
        addLaserVertex(x(0), x(1), x(2), scan, last_id_, true);
        
        laser_scans_[last_id_] = scan;
        laser_vertex_ids_.push_back(last_id_);

        last_scan_added = scan;
        last_add_pose   = x;
        return;
    }

    // 1.2 - Building up a Graph
    // Whenever the robot moves more than 0.7 meters or rotates more than 0.5 radians
    // from the last vertex's pose, it adds to the graph a new LasreVertex each 2/3
    // OdomVertices (to refine the vertex frequency).

    double TRESH_D = 0.7;   // mt
    double TRESH_T = 0.5;   // rad
    double d = sqrt(pow((x(0) - last_add_pose(0)), 2) + pow((x(1) - last_add_pose(1)), 2)); // Euclidean  dist.
    double d_t  = fabs(x(2) - last_add_pose(2));
    int last_id = robot_pose_ids_.back();
    int current_id  = scan.header.seq;
    
    if(d >= TRESH_D || d_t >= TRESH_T){     // We need to add a vertex
        if(vertex_c < 4 || first_vert_){    // Add OdomVertex

            addOdomVertex(x(0), x(1), x(2), current_id, first_vert_);   // Add vertex
            addOdomEdge(last_id, current_id);                               // Add edge
            
            last_add_pose   = x;
            first_vert_ = false;
        }else{  // Add LaserVertex each 3rd vertex
            // Lookup the transform from /front_laser/scan to /base_link in order to cope with the
            // laser sensor offset
            try {
                listener.waitForTransform("/base_link", "/front_laser_link", ros::Time(0), ros::Duration(1.0));
                listener.lookupTransform( "/base_link", "/front_laser_link", ros::Time(0), transform);
            } catch (tf::TransformException &ex) {
                ROS_ERROR("%s", ex.what());
                return;
            }
            scan = ScanToBaseFrame(scan, transform);
            
            addLaserVertex(x(0), x(1), x(2), scan, current_id);
            laser_scans_[current_id] = scan;
            //cout << "Laser Vertex added\nD: " << d << " - Theta: " << d_t << endl;

            Eigen::Vector3d disp = ICP(last_scan_added, scan);

            // Add the edge between the last LaserVertex and the current one storing the ICP's dipslacement as info
            addLaserEdge(laser_vertex_ids_.back(), current_id, disp(0), disp(1), disp(2), laser_noise_);
            //cout << "-----------\n" << disp.transpose() << endl;
            laser_vertex_ids_.push_back(current_id);

            // Check for loop-closure by looking at the displacement (ICP) comparing last scan with the 3 best 
            // matches among the vertices: found by looking at the 3sigma marginal covariance (in Hred)
            // If the current_scan, compared with the best matches, has displacement < than a tresh, then I 
            // found a loop-closure --> add the edge between current_scan and the best matched scan, adding
            // as edge 'weight' the displacement btw the 2

            // detect3SgimaMethod(current_id); // the Jacobian calculations dont work due to 'e->linearizeOplus() segmentation fault'

            // Simpllified version (brute force): find a match by scanning trought all the LaserVertices, computing the ICP
            // and checking if the latter is within a treshold or the error is below 'tolerance'
            detectLoopClosureICP(scan, current_id);
            
            last_scan_added     = scan;
            last_add_pose       = x;
            vertex_c = 0;
        }
        // Counter to track the number of vertices added since the last laser vertex
        vertex_c++;
    }
    // Keep This - reports your update
    updateLocalization();
    visualizeRobotPoses();
    // Keep This - if you like to visualize your map (collected laser scans in the graph)
    visualizeLaserScans();
}


sensor_msgs::LaserScan G2oBasedMapping::rawLasertoLaserScanMsg(g2o::RawLaser rawlaser){
    sensor_msgs::LaserScan msg;

    msg.header.frame_id = "base_laser_link";
    msg.angle_min = rawlaser.laserParams().firstBeamAngle;
    msg.angle_increment = rawlaser.laserParams().angularStep;
    msg.range_min = 0;
    msg.range_max = rawlaser.laserParams().maxRange;

    vector<double>::const_iterator it = rawlaser.ranges().begin();
    msg.ranges.assign(it, rawlaser.ranges().end());

    //static ros::Publisher pub = nh_.advertise<sensor_msgs::LaserScan>("scan_match", 0);
    //pub.publish(msg);

    return msg;
}


void G2oBasedMapping::visualizeLaserScans()
{
    sensor_msgs::PointCloud graph_cloud;
    graph_cloud.header.frame_id = "map";
    graph_cloud.header.stamp = ros::Time::now();
    double laser_x_trans = 0.38;
    
    for(int j = 0; j < robot_pose_ids_.size(); j++)
    {
        std::vector<double> data;
        graph_.vertex(robot_pose_ids_[j])->getEstimateData(data);

        g2o::OptimizableGraph::Data* d = graph_.vertex(robot_pose_ids_[j])->userData();

        g2o::RawLaser* rawLaser = dynamic_cast<g2o::RawLaser*>(d);
        if (rawLaser)
        {
            float angle = rawLaser->laserParams().firstBeamAngle;
            for(std::vector<double>::const_iterator i = rawLaser->ranges().begin(); i != rawLaser->ranges().end(); i++)
            {
                geometry_msgs::Point32 p;
                float x = *i*cos(angle);
                float y = *i*sin(angle);
                p.x = data[0] + laser_x_trans * std::cos(data[2]) + x*cos(data[2])-y*sin(data[2]);
                p.y = data[1] + laser_x_trans * std::sin(data[2]) + x*sin(data[2])+y*cos(data[2]);
                p.z = 0;
                angle += rawLaser->laserParams().angularStep;
                graph_cloud.points.push_back(p);

            }
        }
    }


    static ros::Publisher pub = nh_.advertise<sensor_msgs::PointCloud>("graph_cloud", 0);
    pub.publish(graph_cloud);
}

void G2oBasedMapping::visualizeRobotPoses()
{

    visualization_msgs::Marker marker;
    visualization_msgs::MarkerArray marker_array;

    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time::now();
    marker.ns = "robot_poses";
    marker.pose.position.z = 0.0;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale.x = 0.2;
    marker.scale.y = 0.2;
    marker.scale.z = 0.2;
    marker.color.a = 0.5;
    marker.color.r = 0.1;
    marker.color.g = 0.1;
    marker.color.b = 0.9;

    for(int j = 0; j < robot_pose_ids_.size(); j++)
    {
        // Sphere Marker
        std::vector<double> data;
        graph_.vertex(robot_pose_ids_[j])->getEstimateData(data);
        marker.pose.position.x = data[0];
        marker.pose.position.y = data[1];
        marker.id = robot_pose_ids_[j];
        marker_array.markers.push_back(marker);
    }

    static ros::Publisher pub = nh_.advertise<visualization_msgs::MarkerArray>("robot_pose_marker", 0);
    pub.publish(marker_array);

    visualizeEdges();
}


void G2oBasedMapping::visualizeLandmarks()
{
    visualization_msgs::Marker marker;
    visualization_msgs::Marker marker_text;;
    visualization_msgs::MarkerArray marker_array;
    visualization_msgs::MarkerArray marker_array_text;

    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time::now();
    marker.pose.position.z = 0.0;
    marker.ns = "observed_fiducials";
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale.x = 0.6;
    marker.scale.y = 0.6;
    marker.scale.z = 0.6;
    marker.color.a = 0.5;
    marker.color.r = 1.0;
    marker.color.g = 0.3;
    marker.color.b = 0.0;
    
    marker_text.header = marker.header;
    marker_text.ns = "observed_fiducials_text";
    marker_text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker_text.action = visualization_msgs::Marker::ADD;
    marker_text.scale.z = 0.6*0.85;
    marker_text.color.a = 0.7;
    marker_text.color.r = 0.0;
    marker_text.color.g = 0.0;
    marker_text.color.b = 0.0;
    
    for(int j = 0; j < seen_landmarks_.size(); j++)
    {
        // Sphere Marker
        std::vector<double> data;
        graph_.vertex(seen_landmarks_[j])->getEstimateData(data);

        marker.pose.position.x = data[0];
        marker.pose.position.y = data[1];
        marker.id = seen_landmarks_[j];
        marker_array.markers.push_back(marker);

        // Text Marker
        marker_text.pose.position = marker.pose.position;
        marker_text.id = seen_landmarks_[j];
        marker_text.text = arg_cast<std::string>(marker_text.id);
        marker_array_text.markers.push_back(marker_text);
    }

    static ros::Publisher pub = nh_.advertise<visualization_msgs::MarkerArray>("fiducials_observed_marker", 0);
    pub.publish(marker_array);

    static ros::Publisher pub2 = nh_.advertise<visualization_msgs::MarkerArray>("fiducials_observed_marker", 0);
    pub2.publish(marker_array_text);

    visualizeEdges();
}


void G2oBasedMapping::visualizeEdges()
{
    visualization_msgs::Marker marker;
    visualization_msgs::MarkerArray marker_array;

    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time::now();
    marker.scale.x = 0.05;
    marker.scale.y = 0.05;
    marker.scale.z = 0.05;
    marker.color.a = 0.5;
    marker.color.r = 0.9;
    marker.color.g = 0.1;
    marker.color.b = 0.1;
    marker.id = 0;
    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.action = visualization_msgs::Marker::ADD;
    marker.ns = "edges";

    geometry_msgs::Point p;
    p.z = 0;

    std::vector<double> data;

    for(int j = 0; j < robot_pose_ids_.size(); j++)
    {
        graph_.vertex(robot_pose_ids_[j])->getEstimateData(data);

        p.x = data[0];
        p.y = data[1];
        marker.points.push_back(p);
    }


    static ros::Publisher pub = nh_.advertise<visualization_msgs::Marker>("graph_edges", 0);
    pub.publish(marker);

    marker.points.clear();
    marker.id = 1;
    marker.type = visualization_msgs::Marker::LINE_LIST;

    for(int j = 0; j < robot_landmark_edge_ids_.size(); j++)
    {
        graph_.vertex(robot_landmark_edge_ids_[j].first)->getEstimateData(data);
        p.x = data[0];
        p.y = data[1];
        marker.points.push_back(p);

        graph_.vertex(robot_landmark_edge_ids_[j].second)->getEstimateData(data);
        p.x = data[0];
        p.y = data[1];
        marker.points.push_back(p);
    }

    for(int j = 0; j < laser_edge_ids_.size(); j++)
    {
        graph_.vertex(laser_edge_ids_[j].first)->getEstimateData(data);
        p.x = data[0];
        p.y = data[1];
        marker.points.push_back(p);

        graph_.vertex(laser_edge_ids_[j].second)->getEstimateData(data);
        p.x = data[0];
        p.y = data[1];
        marker.points.push_back(p);
    }



    pub.publish(marker);
}

void G2oBasedMapping::visualizeOldLandmarks()
{
    visualization_msgs::Marker marker;
    visualization_msgs::Marker marker_text;;
    visualization_msgs::MarkerArray marker_array;
    visualization_msgs::MarkerArray marker_array_text;

    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time::now();
    marker.pose.position.z = 0.0;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale.x = 0.6;
    marker.scale.y = 0.6;
    marker.scale.z = 0.6;
    marker.color.a = 0.5;
    marker.color.r = 0.1;
    marker.color.g = 0.1;
    marker.color.b = 0.8;
    marker.ns = "old_observed_fiducials";

    marker_text.header = marker.header;
    marker_text.pose.position = marker.pose.position;
    marker_text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker_text.action = visualization_msgs::Marker::ADD;
    marker_text.scale.z = 0.6*0.85;
    marker_text.color.a = 0.7;
    marker_text.color.r = 0.0;
    marker_text.color.g = 0.0;
    marker_text.color.b = 0.0;
    marker_text.ns = "old_observed_fiducials_text";

    for(int j = 0; j < seen_landmarks_.size(); j++)
    {
        // Sphere Marker
        std::vector<double> data;
        graph_.vertex(seen_landmarks_[j])->getEstimateData(data);

        marker.pose.position.x = data[0];
        marker.pose.position.y = data[1];
        marker.id = seen_landmarks_[j];
        marker_array.markers.push_back(marker);

        // Text Marker
        marker_text.id = seen_landmarks_[j];
        marker_text.text = arg_cast<std::string>(marker_text.id);
        marker_array_text.markers.push_back(marker_text);
    }

    static ros::Publisher pub = nh_.advertise<visualization_msgs::MarkerArray>("old_fiducials_observed_marker", 0);
    pub.publish(marker_array);

    static ros::Publisher pub2 = nh_.advertise<visualization_msgs::MarkerArray>("old_fiducials_observed_marker", 0);
    pub2.publish(marker_array_text);
}


void G2oBasedMapping::init(double x, double y, double theta)
{
    this->x(0,0) = x;
    this->x(1,0) = y;
    this->x(2,0) = theta;

    graph_.clear();
    edge_set_.clear();
    vertex_set_.clear();
    seen_landmarks_.clear();
    robot_pose_ids_.clear();
    robot_landmark_edge_ids_.clear();
    laser_edge_ids_.clear();
    min_to_optimize_ = 4;
    last_id_ = 30;
    valid_ = false;
    reset_ = true;
    robot_pose_set = true;
    first_opt_  = true;
    first_vert_ = true;
    // Counter to track the number of vertices added since the last laser vertex
    vertex_c    = 0;
    visualizeOldLandmarks();
    visualizeLandmarks();
    visualizeRobotPoses();
    visualizeEdges();
}

void G2oBasedMapping::updateLocalization()
{
    tf::Transform transform;
    transform.setOrigin( tf::Vector3(x(0,0), x(1,0), 0.0) );
    transform.setRotation( tf::createQuaternionFromRPY(0 , 0, x(2,0)) );
    pose_tf_broadcaster.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map", "base_link_g2o"));
}

void G2oBasedMapping::laserscanCallback(const sensor_msgs::LaserScan::ConstPtr& msg)
{
    data_mutex_->lock();
    updateLaserScan(*msg);
    data_mutex_->unlock();
}

void G2oBasedMapping::odometryCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
    data_mutex_->lock();
    updateOdometry(*msg);
    data_mutex_->unlock();
}

void G2oBasedMapping::initialposeCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg)
{
    double x, y, theta;
    data_mutex_->lock();
    x = msg->pose.pose.position.x;
    y = msg->pose.pose.position.y;
    theta =  tf::getYaw(msg->pose.pose.orientation);
    ROS_INFO("initalPoseCallback x=%f, y=%f, theta=%f", x, y, theta);
    init(x, y, theta);
    data_mutex_->unlock();
}

void G2oBasedMapping::addOdomVertex(double x, double y, double theta, int id, bool first)
{
    g2o::SE2 pose(x, y, theta);
    g2o::VertexSE2* vertex = new g2o::VertexSE2;
    vertex->setId(id);
    vertex->setEstimate(pose);
    graph_.addVertex(vertex);
    vertex_set_.insert(vertex);
    robot_pose_ids_.push_back(id);
    if(first)
        vertex->setFixed(true);
}

void G2oBasedMapping::addLaserVertex(double x, double y, double theta, sensor_msgs::LaserScan scan, int id, bool first)
{
    g2o::SE2 pose(x, y, theta);
    g2o::VertexSE2* vertex = new g2o::VertexSE2;
    vertex->setId(id);
    vertex->setEstimate(pose);
    g2o::RawLaser * rl = new g2o::RawLaser();
    rl->setLaserParams(*laser_params_);
    std::vector<double> r;
    std::vector<float>::iterator it = scan.ranges.begin();
    r.assign(it, scan.ranges.end());
    rl->setRanges(r);
    vertex->addUserData(rl);
    graph_.addVertex(vertex);
    vertex_set_.insert(vertex);
    robot_pose_ids_.push_back(id);
    if(first)
        vertex->setFixed(true);
}



void G2oBasedMapping::addLaserEdge(int id1, int id2, double x, double y, double yaw, Eigen::Matrix3d noise)
{
    g2o::EdgeSE2* edge = new g2o::EdgeSE2;
    edge->vertices()[0] = graph_.vertex(id1);
    edge->vertices()[1] = graph_.vertex(id2);
    edge->setMeasurement(g2o::SE2(x,y,yaw));
    edge->setInformation(noise);

    laser_edge_ids_.push_back(std::pair<int, int>(id1, id2));

    graph_.addEdge(edge);
    edge_set_.insert(edge);
    std::cout << "added laser edge: " << id1 << " - " << id2 << std::endl;
}

void G2oBasedMapping::addOdomEdge(int id1, int id2)
{
    std::vector<double> data1,data2;

    graph_.vertex(id1)->getEstimateData(data1);
    graph_.vertex(id2)->getEstimateData(data2);

    g2o::SE2 vertex1(data1[0], data1[1], data1[2]);
    g2o::SE2 vertex2(data2[0], data2[1], data2[2]);

    g2o::SE2 transform = vertex1.inverse() * vertex2;
    g2o::EdgeSE2* edge = new g2o::EdgeSE2;
    edge->vertices()[0] = graph_.vertex(id1);
    edge->vertices()[1] = graph_.vertex(id2);
    edge->setMeasurement(transform);
    edge->setInformation(odom_noise_);

    graph_.addEdge(edge);
    edge_set_.insert(edge);
    std::cout << "added odometry edge: " << id1 << " - " << id2 << std::endl;
}

void G2oBasedMapping::addLandmarkVertex(double x, double y, int id)
{
    if(graph_.vertex(id))
        return;

    Eigen::Vector2d pos(x, y);
    seen_landmarks_.push_back(id);
    g2o::VertexPointXY *vertex = new g2o::VertexPointXY;
    vertex->setId(id);
    vertex->setEstimate(pos);
    graph_.addVertex(vertex);
    vertex_set_.insert(vertex);
}

void G2oBasedMapping::addLandmarkEdge(int id1, int id2, double x, double y)
{
    std::vector<double> data;
    graph_.vertex(id1)->getEstimateData(data);

    g2o::SE2 vertex1(data[0], data[1], data[2]);
    Eigen::Vector2d vertex2(x, y);
    Eigen::Vector2d measurement;
    measurement = vertex1.inverse() * vertex2;

    g2o::EdgeSE2PointXY* landmark_edge =  new g2o::EdgeSE2PointXY;
    landmark_edge->vertices()[0] = graph_.vertex(id1);
    landmark_edge->vertices()[1] = graph_.vertex(id2);
    landmark_edge->setMeasurement(measurement);
    landmark_edge->setInformation(landmark_noise_);
    graph_.addEdge(landmark_edge);
    edge_set_.insert(landmark_edge);
    robot_landmark_edge_ids_.push_back(std::pair<int, int>(id1, id2));
    std::cout << "added landmark edge: " << id1 << " - " << id2 << std::endl;
}

void G2oBasedMapping::optimizeGraph()
{
    graph_.save("state_before.g2o");
    graph_.setVerbose(true);
    visualizeOldLandmarks();
    std::cout << "Optimizing" << std::endl;

    if(first_opt_)
    {
        if(!graph_.initializeOptimization())
            std::cerr << "FAILED initializeOptimization";
    }
    else if(!graph_.updateInitialization(vertex_set_, edge_set_))
        std::cerr << "FAILED updateInitialization";

    int iterations = 10;
    graph_.optimize(iterations, !first_opt_);
    graph_.save("state_after.g2o");

    first_opt_ = false;
    vertex_set_.clear();
    edge_set_.clear();
    setRobotToVertex(robot_pose_ids_.back());
}

void G2oBasedMapping::setRobotToVertex(int id){
    std::vector<double> data;
    graph_.vertex(id)->getEstimateData(data);

    x(0,0) = data[0];
    x(1,0) = data[1];
    x(2,0) = atan2(sin(data[2]), cos(data[2]));

    updateLocalization(); 

    /*g2o::VertexSE2* v = dynamic_cast<g2o::VertexSE2*>(graph_.vertex(id));
    if (v) {
        Eigen::Vector3d estimate = v->estimate().toVector();
        x(0) = estimate(0);
        x(1) = estimate(1);
        x(2) = estimate(2);

        // Ensure theta is normalized
        x(2) = atan2(sin(x(2)), cos(x(2)));

        updateLocalization();
    }*/
}

void G2oBasedMapping::publishMapThread()
{
    ros::Rate rate(0.2); // ROS Rate at 0.2Hz
    
    while (ros::ok()) {
        ROS_INFO("Publishing map update");
        data_mutex_->lock();
        publishMap();
        data_mutex_->unlock();
        rate.sleep();
    }
}

void G2oBasedMapping::publishMap()
{

    graph_map.header.stamp = ros::Time::now();
    graph_map.info.map_load_time = ros::Time::now();

    int map_size = graph_map.info.width * graph_map.info.height;
    graph_map.data = std::vector<int8_t>(map_size, 0);
    double laser_x_trans = 0.38;
    
    for(int j = 0; j < robot_pose_ids_.size(); j++)
    {
        std::vector<double> data;
        graph_.vertex(robot_pose_ids_[j])->getEstimateData(data);

        g2o::OptimizableGraph::Data* d = graph_.vertex(robot_pose_ids_[j])->userData();

        g2o::RawLaser* rawLaser = dynamic_cast<g2o::RawLaser*>(d);
        if (rawLaser)
        {
            float angle = rawLaser->laserParams().firstBeamAngle;
            for(std::vector<double>::const_iterator i = rawLaser->ranges().begin(); i != rawLaser->ranges().end(); i++)
            {
                geometry_msgs::Point32 p;
                float x = *i*cos(angle);
                float y = *i*sin(angle);
                p.x = data[0] + laser_x_trans * std::cos(data[2]) +x*cos(data[2])-y*sin(data[2]) - graph_map.info.origin.position.x;
                p.y = data[1] + laser_x_trans * std::sin(data[2]) +x*sin(data[2])+y*cos(data[2]) - graph_map.info.origin.position.y;
                angle += rawLaser->laserParams().angularStep;

                int map_x = p.x / graph_map.info.resolution;
                int map_y = p.y / graph_map.info.resolution;

                if (map_x >= 0 && map_y >= 0 && map_x < graph_map.info.width && map_y < graph_map.info.height)
                {
                  graph_map.data[map_y * graph_map.info.width + map_x] = (int8_t) 100;
                }
            }
        }
    }


    static ros::Publisher pub = nh_.advertise<nav_msgs::OccupancyGrid>("graph_map", 0);
    pub.publish(graph_map);
}

int main(int argc, char** argv)
{
    // Initialize ROS
    ros::init (argc, argv, "g2o_based_mapping");
    ros::NodeHandle n;

    G2oBasedMapping* slamar_ptr = new G2oBasedMapping(n);

    ros::Subscriber laserscan = n.subscribe("/front_laser/scan", 1, &G2oBasedMapping::laserscanCallback, slamar_ptr);
    ros::Subscriber odometry = n.subscribe("/gazebo/odom", 1, &G2oBasedMapping::odometryCallback, slamar_ptr);
    ros::Subscriber initialpose = n.subscribe("/initialpose", 1, &G2oBasedMapping::initialposeCallback, slamar_ptr);

    std::thread worker(&G2oBasedMapping::publishMapThread, slamar_ptr);

    std::cout << "g2o based mapping started ..." << std::endl;
    ros::spin();
    worker.join();

    return 0;
}
