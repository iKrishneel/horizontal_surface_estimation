
#pragma once
#ifndef _HORIZONTAL_SURFACE_ESTIMATION_HPP_
#define _HORIZONTAL_SURFACE_ESTIMATION_HPP_

#include <ros/ros.h>
#include <ros/console.h>
#include <tf/transform_listener.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PolygonStamped.h>
#include <geometry_msgs/PointStamped.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/distances.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/segmentation/extract_clusters.h>

#include <opencv2/opencv.hpp>

class HorizontalSurfaceEstimation {

 private:
    typedef pcl::PointXYZRGB PointT;
    typedef pcl::Normal NormalT;
    typedef pcl::PointXYZRGBNormal PointNormalT;
    typedef pcl::PointCloud<PointT> PointCloud;
    typedef pcl::PointCloud<NormalT> PointNormal;
    typedef pcl::PointCloud<PointNormalT> PointCloudNormal;

    int num_threads_;
    int neigbor_size_;
    float object_diameter_;

    int seed_index_;
   
   
    std::string base_link_;
    std::string camera_link_;
   
    pcl::search::KdTree<PointT>::Ptr tree_;
    pcl::KdTreeFLANN<PointT>::Ptr kdtree_;
   
 protected:
    virtual void onInit();
    virtual void subscribe();
    virtual void unsubscribe();

    boost::mutex lock_;
    ros::NodeHandle pnh_;
    ros::Subscriber sub_point_;
    ros::Publisher pub_cloud_;
    ros::Publisher pub_indices_;
    ros::Publisher pub_centroid_;
   
 public:
    HorizontalSurfaceEstimation();
    void callback(const sensor_msgs::PointCloud2::ConstPtr &);
    void getNormals(PointNormal::Ptr, const PointCloud::Ptr);
    void euclideanClustering(PointCloud::Ptr);
    void searchPlannarSurface(PointCloud::Ptr, PointNormal::Ptr);
    int findOffsetClosestPoint(const PointCloud::Ptr, const float = 0.0f,
                               const float = 0.0f);
    template<class T>
    void getPointNeigbour(std::vector<int> &, std::vector<float> &,
                          const PointT, const T = 16, bool = true);
    float coplanarityCriteria(const Eigen::Vector4f, const Eigen::Vector4f,
                              const Eigen::Vector4f, const Eigen::Vector4f,
                              const float = 10.0f, const float = 0.02f);
    void seedCorrespondingRegion(std::vector<int> &, const PointCloud::Ptr,
                                 const PointNormal::Ptr, const int);
    float EuclideanDistance(const Eigen::Vector3f, const Eigen::Vector3f);
   
};

#endif /* _HORIZONTAL_SURFACE_ESTIMATION_HPP_*/
