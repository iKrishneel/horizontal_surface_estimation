
#include <horizontal_surface_estimation/horizontal_surface_estimation.hpp>

HorizontalSurfaceEstimation::HorizontalSurfaceEstimation() :
    num_threads_(), neigbor_size_(16) {

    this->object_diameter_ = 0.10f;
    this->base_link_ = "/base_footprint";
    this->camera_link_ = "/head_rgbd_sensor_link";

    this->tree_ = pcl::search::KdTree<PointT>::Ptr(
       new pcl::search::KdTree<PointT>);
    this->kdtree_ = pcl::KdTreeFLANN<PointT>::Ptr(
       new pcl::KdTreeFLANN<PointT>);
    
        
    this->onInit();
}

void HorizontalSurfaceEstimation::onInit() {
    this->subscribe();
    
    this->pub_cloud_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/jsk_hsr/storage_surface", 1);
    this->pub_centroid_ = this->pnh_.advertise<geometry_msgs::PointStamped>(
       "/jsk_hsr/storage_centroid", 1);
   
}

void HorizontalSurfaceEstimation::subscribe() {
    this->sub_point_ = this->pnh_.subscribe(
       "points", 1, &HorizontalSurfaceEstimation::callback, this);
   
}

void HorizontalSurfaceEstimation::unsubscribe() {
   
}

void HorizontalSurfaceEstimation::callback(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg) {
    PointCloud::Ptr cloud(new PointCloud);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if (cloud->empty()) {
      ROS_ERROR("[::cloudCB]: EMPTY INPUTS");
      return;
    }

    //! orientation of the base frame
    /*
    tf::TransformListener listener;
    tf::StampedTransform transform;
    while (true) {
       try {
          listener.waitForTransform(this->base_link_, this->camera_link_,
                                    ros::Time(0), ros::Duration(10.0));
          listener.lookupTransform(this->base_link_, this->camera_link_,
                                   ros::Time(0), transform);
          break;
       } catch (tf::TransformException ex) {
          ROS_WARN("%s", ex.what());
       }
    }
    */
    
    PointNormal::Ptr normals(new PointNormal);
    this->getNormals(normals, cloud);

    PointCloud::Ptr filtered_cloud(new PointCloud);
    PointNormal::Ptr filtered_normals(new PointNormal);
    Eigen::Vector3f direct = Eigen::Vector3f(0, -1, 0);
    for (int i = 0; i < normals->size(); i++) {
       PointT point = cloud->points[i];
       NormalT normal = normals->points[i];
       Eigen::Vector3f n = normals->points[i].getNormalVector3fMap();
       float v = direct.dot(n.normalized());
       if (v > 0.85 && v <= 1.0 && !std::isnan(v) && !std::isnan(point.x) &&
           !std::isnan(point.x) && !std::isnan(point.x)) {  //! auto estimate
          filtered_cloud->push_back(point);
          filtered_normals->push_back(normal);
       }
    }
    
    //! remove noise
    this->euclideanClustering(filtered_cloud);
    
    this->searchPlannarSurface(filtered_cloud, filtered_normals);
    
    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*filtered_cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    this->pub_cloud_.publish(ros_cloud);


    //! publish the centroid
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*filtered_cloud, centroid);
    geometry_msgs::PointStamped ros_point;
    ros_point.header = cloud_msg->header;
    ros_point.point.x = centroid(0);
    ros_point.point.y = centroid(1);
    ros_point.point.z = centroid(2);
    this->pub_centroid_.publish(ros_point);
}


void HorizontalSurfaceEstimation::searchPlannarSurface(
    PointCloud::Ptr cloud, PointNormal::Ptr normals) {
    if (cloud->empty()) {
       return;
    }

    this->kdtree_->setInputCloud(cloud);
    std::vector<int> labels(static_cast<int>(cloud->size()), -1);
    
    bool is_found = false;
    float prev_nearest = 0.0f;
    float offset = 0.0f;
    int index = -1;

    std::cout << "--------------------------------"  << "\n";
    
    while (!is_found) {
       index = this->findOffsetClosestPoint(cloud, prev_nearest, offset);
       if (index == -1) {
          break;
       }

       prev_nearest = cloud->points[index].z;
       ROS_WARN("Prev Nearest: %3.2f", prev_nearest);
       
       this->seed_index_ = index;
       this->seedCorrespondingRegion(labels, cloud, normals, index);

       float max_dist = 0.0f;
       int max_dist_idx = -1;
       int icounter = 0;

       float max_z = 0.0f;
       float max_x = 0.0f;
       float min_z = FLT_MAX;
       float min_x = FLT_MAX;
       
       PointCloud::Ptr out_cloud(new PointCloud);
       for (int i = 0; i < labels.size(); i++) {
          if (labels[i] == 1) {
             PointT pt = cloud->points[i];
             out_cloud->push_back(pt);
             float dist = this->EuclideanDistance(
                cloud->points[seed_index_].getVector3fMap(),
                pt.getVector3fMap());
             if (dist > max_dist) {
                max_dist = dist;
                max_dist_idx = icounter++;
             }

             max_z = pt.z > max_z ? pt.z : max_z;
             max_x = pt.x > max_x ? pt.x : max_x;
             min_z = pt.z < min_z ? pt.z : min_z;
             min_x = pt.x < min_x ? pt.x : min_x;
             
             labels[i] = -2;  //! outlier
          }
       }

       
       std::cout << "max dist: " << max_dist << " " << max_dist_idx  << "\n";
       std::cout << "dims: " << max_x - min_x<< " " << max_z - min_z  << "\n";
       
       if (max_dist >= this->object_diameter_ &&
           max_z - min_z >= this->object_diameter_ &&
           max_x - min_x >= this->object_diameter_) {
          cloud->clear();
          *cloud = *out_cloud;
          is_found = true;
       } else {
          /*
          PointCloud::Ptr temp_cloud(new PointCloud);
          PointNormal::Ptr temp_normals(new PointNormal);
          for (int i = 0; i < labels.size(); i++) {
             if (labels[i] == -1) {
                PointT pt = cloud->points[i];
                NormalT nt = normals->points[i];
                temp_cloud->push_back(pt);
                temp_normals->push_back(nt);
             }
          }
          cloud->clear();
          *cloud = *temp_cloud;

          normals->clear();
          *normals = *temp_normals;
          
          temp_cloud->clear();
          temp_normals->clear();

          this->kdtree_->setInputCloud(cloud);
          */

          offset = max_dist + 0.01f;
       }

       // is_found = true;
       ROS_INFO("Next iteration: %3.2f", offset);

    }
}

int HorizontalSurfaceEstimation::findOffsetClosestPoint(
    const PointCloud::Ptr cloud, const float prev_nearest, const float offset) {
    if (cloud->empty()) {
       return -1;
    }

    float min_distance = FLT_MAX;
    int closest_idx = -1;
    for (int i = 0; i < cloud->size(); i++) {
       PointT point = cloud->points[i];
       if (point.z < min_distance && !std::isnan(point.z) &&
           point.z > prev_nearest + offset) {
          min_distance = cloud->points[i].z;
          closest_idx = i;
       }
    }
    
    PointT closest_point;
    if (closest_idx != -1) {
       closest_point = cloud->points[closest_idx];
    } else {
       return -1;
    }
    return closest_idx;
    

    std::cout << "Offset: " << offset  << "\n";

    //! find the point offset distance away
    if (offset > 0.0f) {
       std::vector<int> neigbor_indices;
       std::vector<float> neigbor_distances;
       this->getPointNeigbour<float>(neigbor_indices, neigbor_distances,
                                     closest_point, offset, false);
       for (auto it = neigbor_distances.begin();
            it != neigbor_distances.end(); it++) {
          
          std::cout << "dist: " << *it  << " " << closest_point
                    << "offset: " << offset * offset << "\n";

          
          if (*it >= offset * offset) {
             int idx = it - neigbor_distances.begin();
             return neigbor_indices[idx];
          }
       }
    } else {
       return closest_idx;
    }
}


void HorizontalSurfaceEstimation::seedCorrespondingRegion(
    std::vector<int> &labels, const PointCloud::Ptr cloud,
    const PointNormal::Ptr normals, const int parent_index) {
    std::vector<int> neigbor_indices;
    std::vector<float> neigbor_distances;
    this->getPointNeigbour<int>(neigbor_indices,
                                neigbor_distances,
                                cloud->points[parent_index],
                                this->neigbor_size_);

    int neigb_lenght = static_cast<int>(neigbor_indices.size());
    std::vector<int> merge_list(neigb_lenght, -1);
    merge_list[0] = -1;

#ifdef _OPENMP
#pragma omp parallel for num_threads(this->num_threads_)        \
    shared(merge_list, labels)
#endif
    for (int i = 1; i < neigbor_indices.size(); i++) {
       int index = neigbor_indices[i];
       float dist = this->EuclideanDistance(
          cloud->points[neigbor_indices[i]].getVector3fMap(),
          cloud->points[seed_index_].getVector3fMap());
      if (index != parent_index && labels[index] == -1 &&
          dist < this->object_diameter_ + 0.02f) {
         Eigen::Vector4f parent_pt = cloud->points[
            parent_index].getVector4fMap();
         Eigen::Vector4f parent_norm = normals->points[
            parent_index].getNormalVector4fMap();
         Eigen::Vector4f child_pt = cloud->points[index].getVector4fMap();
         Eigen::Vector4f child_norm = normals->points[
            index].getNormalVector4fMap();
         if (this->coplanarityCriteria(parent_pt, child_pt,
                                       parent_norm, child_norm) == 1.0f) {
            merge_list[i] = index;
            labels[index] = 1;
         } else {
            merge_list[i] = -1;
         }
      } else {
         merge_list[i] = -1;
      }
    }

#ifdef _OPENMP
#pragma omp parallel for num_threads(this->num_threads_) schedule(guided, 1)
#endif
    for (int i = 0; i < merge_list.size(); i++) {
       int index = merge_list[i];
       if (index != -1) {
          seedCorrespondingRegion(labels, cloud, normals, index);
       }
     }
}

float HorizontalSurfaceEstimation::EuclideanDistance(
    const Eigen::Vector3f point1, const Eigen::Vector3f point2) {
    float distance = FLT_MAX;
    distance = std::pow(point1(0) - point2(0), 2) +
       std::pow(point1(1) - point2(1), 2) +
       std::pow(point1(2) - point2(2), 2);
    return std::sqrt(distance);
}


float HorizontalSurfaceEstimation::coplanarityCriteria(
    const Eigen::Vector4f centroid, const Eigen::Vector4f n_centroid,
    const Eigen::Vector4f normal, const Eigen::Vector4f n_normal,
    const float angle_thresh, const float dist_thresh) {
    float tetha = std::acos(normal.dot(n_normal) / (
                               n_normal.norm() * normal.norm()));
    float ang_thresh = angle_thresh * (M_PI/180.0f);
    float coplannarity = 0.0f;
    if (tetha < ang_thresh) {
       float direct1 = normal.dot(centroid - n_centroid);
       float direct2 = n_normal.dot(centroid - n_centroid);
       float dist = std::fabs(std::max(direct1, direct2));
       if (dist < dist_thresh) {
          coplannarity = 1.0f;
       }
    }
    return coplannarity;
}

template<class T>
void HorizontalSurfaceEstimation::getPointNeigbour(
    std::vector<int> &neigbor_indices,
    std::vector<float> &point_squared_distance, const PointT seed_point_,
    const T K, bool is_knn) {
    if (isnan(seed_point_.x) || isnan(seed_point_.y) || isnan(seed_point_.z)) {
       ROS_ERROR("THE CLOUD IS EMPTY. RETURING VOID IN GET NEIGBOUR");
       return;
    }
    neigbor_indices.clear();
    point_squared_distance.clear();
    
    if (is_knn) {
       int search_out = kdtree_->nearestKSearch(
          seed_point_, K, neigbor_indices, point_squared_distance);
    } else {
       int search_out = kdtree_->radiusSearch(
          seed_point_, K, neigbor_indices, point_squared_distance);
    }
}

void HorizontalSurfaceEstimation::getNormals(
    PointNormal::Ptr normals, const PointCloud::Ptr cloud) {
    if (cloud->empty()) {
       ROS_ERROR("-Input cloud is empty in normal estimation");
       return;
    }
    pcl::IntegralImageNormalEstimation<PointT, NormalT> ne;
    ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);
    ne.setMaxDepthChangeFactor(0.02f);
    ne.setNormalSmoothingSize(10.0f);
    ne.setInputCloud(cloud);
    ne.compute(*normals);
}

void HorizontalSurfaceEstimation::euclideanClustering(
    PointCloud::Ptr cloud) {
    std::vector<pcl::PointIndices> cluster_indices;
    this->tree_->setInputCloud(cloud);
    
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(0.02);
    ec.setMinClusterSize(1000);
    ec.setMaxClusterSize(static_cast<int>(cloud->size()));
    ec.setSearchMethod(this->tree_);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);


    std::cout << "Size: " << cluster_indices.size()  << "\n";
    
    PointCloud::Ptr temp_cloud(new PointCloud);
    for (auto it = cluster_indices.begin(); it != cluster_indices.end(); it++) {
       for (auto it2 = it->indices.begin(); it2 != it->indices.end(); it2++) {
          temp_cloud->push_back(cloud->points[*it2]);
       }
       // temp_cloud->clear();
    }

    cloud->clear();
    *cloud = *temp_cloud;
}


int main(int argc, char *argv[]) {
    ros::init(argc, argv, "horizontal_surface_estimation_node");
    HorizontalSurfaceEstimation hse;
    ros::spin();
    return 0;
}
