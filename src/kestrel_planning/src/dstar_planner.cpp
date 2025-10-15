#include "dstar_planner.hpp"

void DSTARLITE::initializeCostmap() {
    // Only call this ONCE at the very beginning
    std::cout << "[COSTMAP] Initializing costmap structure" << std::endl;
    costmap.clear();
    km = 0.0f;
    visited.clear();  // Clear visited tracking
    
    for (uint32_t x = 0; x < X_RANGE; ++x) {
        for (uint32_t y = 0; y < Y_RANGE; ++y) {
            for (uint32_t z = 0; z < Z_RANGE; ++z) {
                auto cell_state = std::make_shared<state>();
                cell_state->setPoint(vec3(x, y, z));
                cell_state->setG(INF_FLOAT);
                cell_state->setRHS(INF_FLOAT);
                cell_state->setNextStep(nullptr);
                costmap.insert(cell_state, cell_state->getPoint());
            }
        }
    }
}

void DSTARLITE::initialize() {
    std::cout << "[PATHING START] dstarlite initializing" << std::endl;
    
    km = 0.0f;
    while (!open_list.empty()) {
        open_list.pop();
    }
    open_set.clear();

    for (auto& node : visited) {
        if (node) {
            node->setG(INF_FLOAT);
            node->setRHS(INF_FLOAT);
            node->setNextStep(nullptr);
        }
    }
    visited.clear();
    
    goal_state = costmap(goal.x, goal.y, goal.z);
    start_state = costmap(start.x, start.y, start.z);

    if (!goal_state || !start_state) {
        std::cerr << "[ERROR]: Start or goal not found in costmap\n";
        return;
    }

    goal_state->setG(INF_FLOAT);
    goal_state->setRHS(0.0f);
    goal_state->setNextStep(nullptr);
    goal_state->setKey(calculateKey(goal_state));
    visited.push_back(goal_state); 
    
    start_state->setG(INF_FLOAT);
    start_state->setRHS(INF_FLOAT);
    start_state->setNextStep(nullptr);
    visited.push_back(start_state); 
    
    insertOpenList(goal_state);
}

float DSTARLITE::edgeCost(const std::shared_ptr<state>& a, const std::shared_ptr<state>& b) {
    const auto& A = a->getPoint();
    const auto& B = b->getPoint();
    if (isOccupied(B.x, B.y, B.z)) return INF_FLOAT; 
    
    float dx = float(B.x - A.x), dy = float(B.y - A.y), dz = float(B.z - A.z);
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

void DSTARLITE::insertOpenList(std::shared_ptr<state> node) {
    auto key = calculateKey(node);

    if (isInOpenList(node)) open_set.erase(node);

    open_list.push(std::make_pair(key, node));
    open_set.insert(node);
}

bool DSTARLITE::isInOpenList(std::shared_ptr<state> node) {
    return open_set.find(node) != open_set.end();
}

std::pair<float,float> DSTARLITE::calculateKey(std::shared_ptr<state> u) {
    float val = std::min(u->getG(), u->getRHS());
    return { val + heuristic(start_state, u) + km, val };
}

std::vector<std::shared_ptr<state>> DSTARLITE::getSuccessors(std::shared_ptr<state> node) {
    std::vector<std::shared_ptr<state>> succs;
    vec3 pos = node->getPoint();

    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                if (dx == 0 && dy == 0 && dz == 0) continue;

                vec3 neighbor_pos(pos.x + dx, pos.y + dy, pos.z + dz);
                if (neighbor_pos.x < 0 || neighbor_pos.x >= static_cast<int>(X_RANGE) ||
                    neighbor_pos.y < 0 || neighbor_pos.y >= static_cast<int>(Y_RANGE) ||
                    neighbor_pos.z < 0 || neighbor_pos.z >= static_cast<int>(Z_RANGE)) {
                    continue;
                }

                auto neighbor = costmap(neighbor_pos.x, neighbor_pos.y, neighbor_pos.z);
                if (neighbor && !isOccupied(neighbor_pos.x, neighbor_pos.y, neighbor_pos.z)) {
                    succs.push_back(neighbor);
                }
            }
        }
    }
    return succs;
}

std::vector<std::shared_ptr<state>> DSTARLITE::getPredecessors(std::shared_ptr<state> node) {
    std::vector<std::shared_ptr<state>> preds;
    vec3 pos = node->getPoint();

    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                if (dx == 0 && dy == 0 && dz == 0) continue;

                int nx = pos.x + dx;
                int ny = pos.y + dy;
                int nz = pos.z + dz;

                if (nx < 0 || nx >= static_cast<int>(X_RANGE) ||
                    ny < 0 || ny >= static_cast<int>(Y_RANGE) ||
                    nz < 0 || nz >= static_cast<int>(Z_RANGE))
                    continue;

                auto neighbor = costmap(nx, ny, nz);
                if (neighbor && heuristic(neighbor, node) < INF_FLOAT) {
                    preds.push_back(neighbor);
                }
            }
        }
    }
    return preds;
}

bool DSTARLITE::openListEmpty() {
    while (!open_list.empty() &&
           open_set.find(open_list.top().second) == open_set.end()) {
        open_list.pop();
    }
    return open_list.empty();
}

void DSTARLITE::removeFromOpenList(std::shared_ptr<state> node) {
    open_set.erase(node);
}

int DSTARLITE::computeShortestPath() {
    uint32_t count = 0;

    while (!openListEmpty()) {
        auto topPair = open_list.top();
        auto u = topPair.second;

        auto topKey = calculateKey(u);
        auto startKey = calculateKey(start_state);

        if (!(topKey < startKey || start_state->getRHS() != start_state->getG()))
            break;

        u = topOpenList();
        if (!u) break; 

        removeFromOpenList(u);

        if (u->getG() > u->getRHS()) {
            u->setG(u->getRHS());
            visited.push_back(u); 
            for (auto s : getSuccessors(u)) updateVertex(s);
        } else {
            u->setG(INF_FLOAT);
            visited.push_back(u);  
            updateVertex(u);
            for (auto s : getSuccessors(u)) updateVertex(s);
        }

        if (++count > X_RANGE * Y_RANGE * Z_RANGE) {
            std::cerr << "[ERROR] Max iterations reached in computeShortestPath\n";
            break;
        }
    }
    return count;
}

void DSTARLITE::updateVertex(std::shared_ptr<state> u) {
    if (u->getPoint() != goal) {
        float min_rhs = INF_FLOAT;
        std::shared_ptr<state> best = nullptr;
        for (auto s : getSuccessors(u)) {
            float c = edgeCost(u, s);
            if (c >= INF_FLOAT) continue;
            float val = s->getG() + c;
            if (val < min_rhs) { min_rhs = val; best = s; }
        }
        u->setRHS(min_rhs);
        u->setNextStep(best);
        visited.push_back(u);  
    }

    if (isInOpenList(u)) removeFromOpenList(u);
    if (!u->isConsistent()) insertOpenList(u);
}

float DSTARLITE::heuristic(std::shared_ptr<state> s, std::shared_ptr<state> u) {
    vec3 svec = s->getPoint();
    vec3 uvec = u->getPoint();

    double dx = std::abs(svec.x - uvec.x);
    double dy = std::abs(svec.y - uvec.y);
    double dz = std::abs(svec.z - uvec.z);

    if (dx < dy) std::swap(dx, dy);
    if (dx < dz) std::swap(dx, dz);
    if (dy < dz) std::swap(dy, dz);

    return (float)(
        (dx - dy)                 
        + (dy - dz) * std::sqrt(2.0)  
        + dz * std::sqrt(3.0)        
    );
}

std::shared_ptr<state> DSTARLITE::topOpenList() {
    while (!open_list.empty()) {
        auto [oldKey, node] = open_list.top();

        if (open_set.find(node) == open_set.end()) { open_list.pop(); continue; }

        auto newKey = calculateKey(node);
        if (std::tie(oldKey.first, oldKey.second) <
            std::tie(newKey.first, newKey.second)) {
            open_set.erase(node);            
            open_list.pop();                
            insertOpenList(node);            
            continue;                        
        }

        return node;
    }
    return nullptr;
}

int DSTARLITE::extractPath(std::vector<geometry_msgs::msg::PoseStamped> &waypoints) {
    std::shared_ptr<state> node = costmap(start.x, start.y, start.z);
    
    if (!node) {
        return 0;
    }
    
    if (!node->nextStep()) {
        std::cout << "Start node has no parent!\n";
        return 0;
    }

    uint32_t count = 0;
    std::set<vec3> visited_path;
    
    while (node && visited_path.find(node->getPoint()) == visited_path.end()) {
        vec3 point = node->getPoint();
        std::cout << "[DEBUG] " << point.x << " " << point.y << " " << point.z << std::endl;
        visited_path.insert(point);

        if (!(point == start)) { 
            geometry_msgs::msg::PoseStamped pose;
            pose.header.frame_id = "map";
            
            pose.pose.position.x = point.x;
            pose.pose.position.y = point.y;
            pose.pose.position.z = point.z;

            pose.pose.orientation.x = 0.0;
            pose.pose.orientation.y = 0.0;
            pose.pose.orientation.z = 0.0;
            pose.pose.orientation.w = 1.0;

            waypoints.push_back(pose);
        }

        count++;
        if (point == goal) break;

        node = node->nextStep();
    }

    if (visited_path.find(goal) == visited_path.end()) {
        std::cerr << "Path broken before reaching goal.\n";
    }

    return count;
}

bool DSTARLITE::isOccupied(int x, int y, int z) {
    std::shared_ptr<state>& s = costmap(x, y, z);
    if (!s) {
        return false;
    }
    return s->getOccupy();
}

void DSTARLITE::setOccupiedStatus(int x, int y, int z, bool value) {
    auto state_occupied = costmap(x, y, z);
    if (!state_occupied) {
        std::cerr << "Error: No state found at (" << x << ", " << y << ", " << z << ")\n";
        return;
    }
    state_occupied->setOccupation(value);
    state_occupied->setG(INF_FLOAT);
    state_occupied->setRHS(INF_FLOAT);
    state_occupied->setNextStep(nullptr);
    visited.push_back(state_occupied); 
}

void DSTARLITE::replan(float x, float y, float z) {
    int px = static_cast<int>(x + X_RANGE / 2);
    int py = static_cast<int>(y + Y_RANGE / 2);
    int pz = static_cast<int>(z);

    if (px < 0 || px >= (int)X_RANGE || 
        py < 0 || py >= (int)Y_RANGE || 
        pz < 0 || pz >= (int)Z_RANGE) {
        return;
    }

    auto old_cost = costmap(px, py, pz);
    if (!old_cost) return;

    vec3 obstacle_pos(px, py, pz);
    km += heuristic(start_state, costmap(px, py, pz));
    
    setOccupiedStatus(px, py, pz, true);
    auto node = costmap(px, py, pz);
    
    updateVertex(node);
    for (auto neighbor : getSuccessors(node)) {
        updateVertex(neighbor);
    }
    for (auto neighbor : getPredecessors(node)) {
        updateVertex(neighbor);
    }

    computeShortestPath();
}