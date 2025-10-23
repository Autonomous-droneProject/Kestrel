#include "dstar_planner.hpp"
#include <cmath>
#include <algorithm>
#include <utility>
#include <tuple>

// Add this block here ↓↓↓
namespace {
constexpr int OFF26[26][3] = {
    {-1,-1,-1},{ 0,-1,-1},{ 1,-1,-1},
    {-1, 0,-1},{ 0, 0,-1},{ 1, 0,-1},
    {-1, 1,-1},{ 0, 1,-1},{ 1, 1,-1},
    {-1,-1, 0},{ 0,-1, 0},{ 1,-1, 0},
    {-1, 0, 0},              { 1, 0, 0},
    {-1, 1, 0},{ 0, 1, 0},{ 1, 1, 0},
    {-1,-1, 1},{ 0,-1, 1},{ 1,-1, 1},
    {-1, 0, 1},{ 0, 0, 1},{ 1, 0, 1},
    {-1, 1, 1},{ 0, 1, 1},{ 1, 1, 1}
};
}
// ← stop here. Do not put anything else in this namespace.

static inline float heuristic_point(const vec3& s, const vec3& u) {
    float dx = std::fabs(float(s.x) - float(u.x));
    float dy = std::fabs(float(s.y) - float(u.y));
    float dz = std::fabs(float(s.z) - float(u.z));
    if (dy > dx) std::swap(dx, dy);
    if (dz > dx) std::swap(dx, dz);
    if (dz > dy) std::swap(dz, dy);
    return (dx - dy) + (dy - dz) * std::sqrt(2.0f) + dz * std::sqrt(3.0f);
}


void DSTARLITE::initialize() {
    std::cout << "[PATHING START] dstarlite initializing" << std::endl;

    // reset planner bookkeeping
    km = 0.0f;
    while (!open_list.empty()) open_list.pop();
    open_set.clear();

    // build cells only once; keep existing grid/state if already built
    if (!costmap(0,0,0)) {
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

    goal_state  = costmap(goal.x,  goal.y,  goal.z);
    start_state = costmap(start.x, start.y, start.z);
    if (!goal_state || !start_state) {
        std::cerr << "[ERROR]: Start or goal not found in costmap\n";
        return;
    }

    // standard D* Lite init: only goal’s RHS = 0; don’t nuke entire grid
    goal_state->setG(INF_FLOAT);
    goal_state->setRHS(0.0f);
    goal_state->setNextStep(nullptr);
    goal_state->setkey(calculateKey(goal_state));

    // start begins inconsistent; path search will set values
    start_state->setG(INF_FLOAT);
    start_state->setRHS(INF_FLOAT);
    start_state->setNextStep(nullptr);

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
    node->setkey(key);

    if (isInOpenList(node))
        open_set.erase(node);

    open_list.push(std::make_pair(key, node));
    open_set.insert(node);

    std::cout << "[DEBUG] Inserted ("<<node->getPoint().x<<","<<node->getPoint().y<<","<<node->getPoint().z
              <<") key=("<<key.first<<","<<key.second<<")\n";
}



bool DSTARLITE::isInOpenList(std::shared_ptr<state> node) {
    return open_set.find(node) != open_set.end();
}

std::shared_ptr<state> DSTARLITE::topOpenList() {
    while (!open_list.empty()) {
        auto [qkey, node] = open_list.top();

        // 1) drop if node was logically removed
        if (open_set.find(node) == open_set.end()) {
            open_list.pop();
            continue;
        }

        // 2) use the node's cached key; only recompute if it changed
        const auto cached = node->getKey();  // requires state::{getKey,setkey}()

        // If queue key != cached, refresh cached once
        if (std::tie(qkey.first, qkey.second) != std::tie(cached.first, cached.second)) {
            const auto new_key = calculateKey(node);  // recompute once
            node->setkey(new_key);

            // If the new key is *worse* than what’s in the queue, requeue with the new key
            if (std::tie(qkey.first, qkey.second) < std::tie(new_key.first, new_key.second)) {
                open_set.erase(node);     // mark old entry invalid
                open_list.pop();          // remove stale heap entry
                insertOpenList(node);     // push with updated key (also caches it)
                continue;                 // re-check new top
            }
        }

        // Keys agree (or improved); accept this node
        return node;
    }
    return nullptr;
}


std::pair<float,float> DSTARLITE::calculateKey(std::shared_ptr<state> u) {
    const float val = std::min(u->getG(), u->getRHS());
    // pointer-free heuristic; identical formula, less overhead
    const float h = heuristic_point(start_state->getPoint(), u->getPoint());
    return { val + h + km, val };
}



std::vector<std::shared_ptr<state>> DSTARLITE::getSuccessors(std::shared_ptr<state> node) {
    std::vector<std::shared_ptr<state>> succs;
    succs.reserve(26);

    const vec3 p = node->getPoint();
    const int W = static_cast<int>(X_RANGE);
    const int H = static_cast<int>(Y_RANGE);
    const int D = static_cast<int>(Z_RANGE);

    // 26-neighborhood offsets (no {0,0,0})
    static constexpr int OFF[26][3] = {
        {-1,-1,-1},{ 0,-1,-1},{ 1,-1,-1},
        {-1, 0,-1},{ 0, 0,-1},{ 1, 0,-1},
        {-1, 1,-1},{ 0, 1,-1},{ 1, 1,-1},
        {-1,-1, 0},{ 0,-1, 0},{ 1,-1, 0},
        {-1, 0, 0},              { 1, 0, 0},
        {-1, 1, 0},{ 0, 1, 0},{ 1, 1, 0},
        {-1,-1, 1},{ 0,-1, 1},{ 1,-1, 1},
        {-1, 0, 1},{ 0, 0, 1},{ 1, 0, 1},
        {-1, 1, 1},{ 0, 1, 1},{ 1, 1, 1}
    };

    for (int i = 0; i < 26; ++i) {
        const int nx = p.x + OFF26[i][0];
        const int ny = p.y + OFF26[i][1];
        const int nz = p.z + OFF26[i][2];


        if ((unsigned)nx >= (unsigned)W ||
            (unsigned)ny >= (unsigned)H ||
            (unsigned)nz >= (unsigned)D) continue;

        auto n = costmap(nx, ny, nz);     // single lookup
        if (!n) continue;
        if (!n->getOccupy()) succs.push_back(n);  // single occupancy check
    }
    return succs;
}




std::vector<std::shared_ptr<state>> DSTARLITE::getPredecessors(std::shared_ptr<state> node) {
    std::vector<std::shared_ptr<state>> preds;
    preds.reserve(26);

    const vec3 p = node->getPoint();
    const int W = static_cast<int>(X_RANGE);
    const int H = static_cast<int>(Y_RANGE);
    const int D = static_cast<int>(Z_RANGE);

    static constexpr int OFF[26][3] = {
        {-1,-1,-1},{ 0,-1,-1},{ 1,-1,-1},
        {-1, 0,-1},{ 0, 0,-1},{ 1, 0,-1},
        {-1, 1,-1},{ 0, 1,-1},{ 1, 1,-1},
        {-1,-1, 0},{ 0,-1, 0},{ 1,-1, 0},
        {-1, 0, 0},              { 1, 0, 0},
        {-1, 1, 0},{ 0, 1, 0},{ 1, 1, 0},
        {-1,-1, 1},{ 0,-1, 1},{ 1,-1, 1},
        {-1, 0, 1},{ 0, 0, 1},{ 1, 0, 1},
        {-1, 1, 1},{ 0, 1, 1},{ 1, 1, 1}
    };

    for (int i = 0; i < 26; ++i) {
        const int nx = p.x + OFF[i][0];
        const int ny = p.y + OFF[i][1];
        const int nz = p.z + OFF[i][2];

        if ((unsigned)nx >= (unsigned)W ||
            (unsigned)ny >= (unsigned)H ||
            (unsigned)nz >= (unsigned)D) continue;

        auto n = costmap(nx, ny, nz);
        if (!n) continue;
        if (n->getOccupy()) continue;
        preds.push_back(n);
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


int DSTARLITE::computeShortestPath() {
    uint32_t count = 0;

    while (!openListEmpty()) {
        std::shared_ptr<state> u = topOpenList();
        if (!u) break;

        const auto startKey = calculateKey(start_state);
        const auto uKey     = u->getKey();   // cached; kept fresh by topOpenList()

        if (!(uKey < startKey || start_state->getRHS() != start_state->getG()))
            break;

        removeFromOpenList(u);

        if (u->getG() > u->getRHS()) {
            u->setG(u->getRHS());
            for (auto s : getSuccessors(u)) updateVertex(s);
        } else {
            u->setG(INF_FLOAT);
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
    std::set<vec3> visited;
    
    while (node && visited.find(node->getPoint()) == visited.end()) {
        vec3 point = node->getPoint();
        visited.insert(point);

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

    if (visited.find(goal) == visited.end()) {
        std::cerr << "Path broken before reaching goal.\n";
    }

    return count;
}

bool DSTARLITE::isOccupied(int x, int y, int z) {
    auto s = costmap(x, y, z);
    return s ? s->getOccupy() : false;
}


void DSTARLITE::setOccupied(int x, int y, int z) {
    auto state_occupied = costmap(x, y, z);
    if (!state_occupied) {
        std::cerr << "Error: No state found at (" << x << ", " << y << ", " << z << ")\n";
        return;
    }
    state_occupied->setOccupation(true);
    state_occupied->setG(INF_FLOAT);
    state_occupied->setRHS(INF_FLOAT);
    state_occupied->setNextStep(nullptr);
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
    
    setOccupied(px, py, pz);
    auto node = costmap(px, py, pz);
    
       updateVertex(node);
    for (auto neighbor : getSuccessors(node)) {
        updateVertex(neighbor);
    }
    for (auto neighbor : getPredecessors(node)) {
        updateVertex(neighbor);
    }

    computeShortestPath();
}  // <-- end of replan()
void DSTARLITE::removeFromOpenList(std::shared_ptr<state> node) {
    open_set.erase(node);
}
