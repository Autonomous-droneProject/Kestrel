#ifndef DSTAR_PLANNER_H
#define DSTAR_PLANNER_H

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

#include "utils.hpp"
#include "Voxel.hpp"

#include <vector>
#include <queue>
#include <limits>
#include <memory>
#include <utility>
#include <cmath>
#include <map>
#include <unordered_set>


const float INF_FLOAT = std::numeric_limits<float>::max();

class DSTARLITE {
public:
    DSTARLITE(uint32_t X_lim, uint32_t Y_lim, uint32_t Z_lim)
    : km(0.0f),
      X_RANGE(X_lim), Y_RANGE(Y_lim), Z_RANGE(Z_lim),
      costmap(X_lim, Y_lim, Z_lim) {}

    DSTARLITE()
    : km(0.0f),
      X_RANGE(100), Y_RANGE(100), Z_RANGE(100),
      costmap(100, 100, 100) {}

    void setGoal(float x, float y, float z) { goal = vec3(x, y, z); }
    void setStart(float x, float y, float z) { start = vec3(x, y, z); }

    void initialize();
    int computeShortestPath();

    void replan(float x, float y, float z);
    int extractPath(std::vector<geometry_msgs::msg::PoseStamped> &waypoints);
    void setOccupied(int x, int y, int z);

private:
    float km;                
    vec3 goal;
    vec3 start;
    std::shared_ptr<state> goal_state;
    std::shared_ptr<state> start_state;

    uint32_t X_RANGE, Y_RANGE, Z_RANGE;
    octree::voxel<state> costmap;
    
    std::pair<float,float> calculateKey(std::shared_ptr<state> u);
    
    std::priority_queue<std::pair<std::pair<float, float>, std::shared_ptr<state>>, 
                       std::vector<std::pair<std::pair<float, float>, std::shared_ptr<state>>>,
                       StateComparator> open_list;
    
    std::unordered_set<std::shared_ptr<state>> open_set;

    bool isOccupied(int x, int y, int z);
    std::vector<std::shared_ptr<state>> getPredecessors(std::shared_ptr<state> node);
    std::vector<std::shared_ptr<state>> getSuccessors(std::shared_ptr<state> node);
    void updateVertex(std::shared_ptr<state> node);
    void expand(std::shared_ptr<state> node);

    void insertOpenList(std::shared_ptr<state> node);
    bool isInOpenList(std::shared_ptr<state> node);
    void removeFromOpenList(std::shared_ptr<state> node);
    std::shared_ptr<state> topOpenList();
    bool openListEmpty();


    float heuristic(std::shared_ptr<state> s, std::shared_ptr<state> u);


    float edgeCost(const std::shared_ptr<state>& a, const std::shared_ptr<state>& b);

};



#endif
