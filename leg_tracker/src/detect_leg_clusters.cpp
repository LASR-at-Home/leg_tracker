/*********************************************************************
* Software License Agreement (BSD License)
* 
*  Copyright (c) 2008, Willow Garage, Inc.
*  All rights reserved.
* 
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
* 
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
* 
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/
// ROS 
#include <rclcpp/rclcpp.hpp>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/buffer_interface.h>
#include <tf2/exceptions.h>
#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/ml.hpp>

// Local headers
#include <leg_tracker/laser_processor.h>
#include <leg_tracker/cluster_features.h>

// Custom messages
#include <leg_tracker_interfaces/msg/leg.hpp>
#include <leg_tracker_interfaces/msg/leg_array.hpp>



/**
* @brief Detects clusters in laser scan with leg-like shapes
*/
class DetectLegClusters
{
public:
  /**
  * @brief Constructor
  */
  DetectLegClusters(rclcpp::Node &node):
    node_(node),
    scan_num_(0),
    num_prev_markers_published_(0)
  {  
    // Get ROS parameters  
    std::string forest_file;
    std::string scan_topic;
    // if (!node_.getParam("forest_file", forest_file))
    //   RCLCPP_ERROR(node_.get_logger(), "ERROR! Could not get random forest filename");
    node_.declare_parameter("forest_file", std::string("/home/jared/robocup/leg_tracker/leg_tracker/config/trained_leg_detector_res=0.33.yaml"));
    node_.declare_parameter("scan_topic", "scan");
    node_.declare_parameter("fixed_frame", std::string("odom"));
    node_.declare_parameter("detection_threshold", -1.0);
    node_.declare_parameter("cluster_dist_euclid", 0.13);
    node_.declare_parameter("min_points_per_cluster", 3);                
    node_.declare_parameter("max_detect_distance", 10.0);   
    node_.declare_parameter("marker_display_lifetime", 0.2);   
    node_.declare_parameter("use_scan_header_stamp_for_tfs", false);
    node_.declare_parameter("max_detected_clusters", -1);

    if (!node_.get_parameter("forest_file", forest_file))
      RCLCPP_ERROR(node_.get_logger(), "ERROR! Could not get random forest filename");

    // Print back
    RCLCPP_INFO(node_.get_logger(), "forest_file: %s", node_.get_parameter("forest_file").as_string().c_str());
    RCLCPP_INFO(node_.get_logger(), "scan_topic: %s", node_.get_parameter("scan_topic").as_string().c_str());
    RCLCPP_INFO(node_.get_logger(), "fixed_frame: %s", node_.get_parameter("fixed_frame").as_string().c_str());
    RCLCPP_INFO(node_.get_logger(), "detection_threshold: %.2f", node_.get_parameter("detection_threshold").as_double());
    RCLCPP_INFO(node_.get_logger(), "cluster_dist_euclid: %.2f", node_.get_parameter("cluster_dist_euclid").as_double());
    RCLCPP_INFO(node_.get_logger(), "min_points_per_cluster: %ld", node_.get_parameter("min_points_per_cluster").as_int());
    RCLCPP_INFO(node_.get_logger(), "max_detect_distance: %.2f", node_.get_parameter("max_detect_distance").as_double()); 
    RCLCPP_INFO(node_.get_logger(), "marker_display_lifetime: %.2f", node_.get_parameter("marker_display_lifetime").as_double());
    RCLCPP_INFO(node_.get_logger(), "use_scan_header_stamp_for_tfs: %d", node_.get_parameter("use_scan_header_stamp_for_tfs").as_bool());
    RCLCPP_INFO(node_.get_logger(), "max_detected_clusters: %ld", node_.get_parameter("max_detected_clusters").as_int());    

    // Load random forest
    forest = cv::ml::StatModel::load<cv::ml::RTrees>(forest_file);
    feat_count_ = forest->getVarCount();

    latest_scan_header_stamp_with_tf_available_ = node_.get_clock()->now();

    // ROS subscribers + publishers
    scan_sub_ = node_.create_subscription<sensor_msgs::msg::LaserScan>("scan", 10, std::bind(&DetectLegClusters::laserCallback, this, std::placeholders::_1));
    markers_pub_ = node_.create_publisher<visualization_msgs::msg::Marker>("visualization_marker", 20);
    detected_leg_clusters_pub_ = node_.create_publisher<leg_tracker_interfaces::msg::LegArray>("detected_leg_clusters", 20);

    // Tf
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(node_.get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  }

private:
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

  cv::Ptr< cv::ml::RTrees > forest = cv::ml::RTrees::create();

  int feat_count_;

  ClusterFeatures cf_;

  int scan_num_;
  bool use_scan_header_stamp_for_tfs_;
  rclcpp::Time latest_scan_header_stamp_with_tf_available_;

  rclcpp::Node &node_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr markers_pub_;
  rclcpp::Publisher<leg_tracker_interfaces::msg::LegArray>::SharedPtr detected_leg_clusters_pub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;

  std::string fixed_frame_;
  
  double detection_threshold_;
  double cluster_dist_euclid_;
  int min_points_per_cluster_;  
  double max_detect_distance_;
  double marker_display_lifetime_;
  int max_detected_clusters_;

  int num_prev_markers_published_;


  /**
  * @brief Clusters the scan according to euclidian distance, 
  *        predicts the confidence that each cluster is a human leg and publishes the results
  * 
  * Called every time a laser scan is published.
  */
  void laserCallback(const sensor_msgs::msg::LaserScan::SharedPtr scan)
  {         
    laser_processor::ScanProcessor processor(*scan); 
    processor.splitConnected(cluster_dist_euclid_);        
    processor.removeLessThan(min_points_per_cluster_);    

    // OpenCV matrix needed to use the OpenCV random forest classifier
    cv::Mat tmp_mat(1, feat_count_, CV_32FC1); 
    
    leg_tracker_interfaces::msg::LegArray detected_leg_clusters;
    detected_leg_clusters.header.frame_id = scan->header.frame_id;
    detected_leg_clusters.header.stamp = scan->header.stamp;

    // Find out the time that should be used for tfs
    auto transform_available = true;
    tf2::TimePoint tf_time;
    // Use time from scan header
    if (use_scan_header_stamp_for_tfs_)
    {
      tf_time = tf2_ros::fromMsg(scan->header.stamp);

      try
      {
        tf_buffer_->lookupTransform(fixed_frame_, scan->header.frame_id, tf_time, tf2::durationFromSec(0.1));
      }
      catch(tf2::TransformException ex)
      {
        RCLCPP_INFO(node_.get_logger(), "Detect_leg_clusters: No tf available");
        transform_available = false;
      }
    }
    else
    {
      // Otherwise just use the latest tf available
      tf_time = tf2::TimePointZero;
      try
      {
        tf_buffer_->lookupTransform(fixed_frame_, scan->header.frame_id, tf_time, tf2::durationFromSec(0.1));
      }
      catch(tf2::TransformException ex)
      {
        RCLCPP_INFO(node_.get_logger(), "Detect_leg_clusters: No tf available");
        transform_available = false;
      }
    }
    
    // Store all processes legs in a set ordered according to their relative distance to the laser scanner
    std::set <leg_tracker_interfaces::msg::Leg, CompareLegs> leg_set;
    if (!transform_available)
    {
      RCLCPP_INFO(node_.get_logger(), "Not publishing detected leg clusters because no tf was available");
    }
    else // transform_available
    {
      // Iterate through all clusters
      for (std::list<laser_processor::SampleSet*>::iterator cluster = processor.getClusters().begin();
       cluster != processor.getClusters().end();
       cluster++)
      {   
        // Get position of cluster in laser frame
        tf2::Stamped<geometry_msgs::msg::Point> position((*cluster)->getPosition(), tf_time, scan->header.frame_id);
        float rel_dist = pow(position.x*position.x + position.y*position.y, 1./2.);
        
        // Only consider clusters within max_distance. 
        if (rel_dist < max_detect_distance_)
        {
          // Classify cluster using random forest classifier
          std::vector<float> f = cf_.calcClusterFeatures(*cluster, *scan);
          for (int k = 0; k < feat_count_; k++)
            tmp_mat.at<float>(k) = (float)(f[k]);
          
          cv::Mat result;
          forest->getVotes(tmp_mat, result, 0);
          int positive_votes = result.at<int>(1, 1);
          int negative_votes = result.at<int>(1, 0);
          float probability_of_leg = positive_votes / static_cast<double>(positive_votes + negative_votes);

          // Consider only clusters that have a confidence greater than detection_threshold_                 
          if (probability_of_leg > detection_threshold_)
          { 
            // Transform cluster position to fixed frame
            // This should always be succesful because we've checked earlier if a tf was available
            bool transform_successful_2;
            try
            {
                auto t = tf_buffer_->lookupTransform(fixed_frame_, scan->header.frame_id, tf_time, tf2::durationFromSec(0.1));
                geometry_msgs::msg::PointStamped p;
                geometry_msgs::msg::PointStamped p_out;
                p.point = position;
                p.header.stamp = tf2_ros::toMsg(tf_time);
                p.header.frame_id = scan->header.frame_id;
                tf2::doTransform(p, p_out, t);
                position.x = p_out.point.x;
                position.y = p_out.point.y;
                transform_successful_2 = true;
            }
            catch (tf2::TransformException ex)
            {
              RCLCPP_ERROR(node_.get_logger(), "%s",ex.what());
              transform_successful_2 = false;
            }

            if (transform_successful_2)
            {  
              // Add detected cluster to set of detected leg clusters, along with its relative position to the laser scanner
              leg_tracker_interfaces::msg::Leg new_leg;
              new_leg.position.x = position.x;
              new_leg.position.y = position.y;
              new_leg.confidence = probability_of_leg;
              leg_set.insert(new_leg);
            }
          }
        }
      }     
    }    
 

    // Publish detected legs to /detected_leg_clusters and to rviz
    // They are ordered from closest to the laser scanner to furthest  
    int clusters_published_counter = 0;
    int id_num = 1;      
    for (std::set<leg_tracker_interfaces::msg::Leg>::iterator it = leg_set.begin(); it != leg_set.end(); ++it)
    {
      // Publish to /detected_leg_clusters topic
      leg_tracker_interfaces::msg::Leg leg = *it;
      detected_leg_clusters.legs.push_back(leg);
      clusters_published_counter++;

      // Publish marker to rviz
      visualization_msgs::msg::Marker m;
      m.header.stamp = scan->header.stamp;
      m.header.frame_id = fixed_frame_;
      m.ns = "LEGS";
      m.id = id_num++;
      m.type = m.SPHERE;
      m.pose.position.x = leg.position.x ;
      m.pose.position.y = leg.position.y;
      m.pose.position.z = 0.2;
      m.scale.x = 0.13;
      m.scale.y = 0.13;
      m.scale.z = 0.13;
      m.color.a = 1;
      m.color.r = 0;
      m.color.g = leg.confidence;
      m.color.b = leg.confidence;
      markers_pub_->publish(m);

      // Comparison using '==' and not '>=' is important, as it allows <max_detected_clusters_>=-1 
      // to publish infinite markers
      if (clusters_published_counter == max_detected_clusters_) 
        break;
    }

    // Clear remaining markers in Rviz
    for (int id_num_diff = num_prev_markers_published_-id_num; id_num_diff > 0; id_num_diff--)
    {
      visualization_msgs::msg::Marker m;
      m.header.stamp = scan->header.stamp;
      m.header.frame_id = fixed_frame_;
      m.ns = "LEGS";
      m.id = id_num_diff + id_num;
      m.action = m.DELETE;
      markers_pub_->publish(m);
    }
    num_prev_markers_published_ = id_num; // For the next callback

    detected_leg_clusters_pub_->publish(detected_leg_clusters);
  }


  /**
  * @brief Comparison class to order Legs according to their relative distance to the laser scanner
  */
  class CompareLegs
  {
  public:
      bool operator ()(const leg_tracker_interfaces::msg::Leg &a, const leg_tracker_interfaces::msg::Leg &b) const
      {
          float rel_dist_a = pow(a.position.x*a.position.x + a.position.y*a.position.y, 1./2.);
          float rel_dist_b = pow(b.position.x*b.position.x + b.position.y*b.position.y, 1./2.);          
          return rel_dist_a < rel_dist_b;
      }
  };

};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("detect_leg_clusters");
  DetectLegClusters dlc(*node);
  rclcpp::spin(node);
  return 0;
}

