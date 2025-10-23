#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#define TOL 1e6

struct vec3 {
    int x, y, z;
    float cost;
    vec3() : x(0), y(0), z(0), cost(0.0f) {}
    vec3(int a, int b, int c) : x(a), y(b), z(c) {cost=0;}
    vec3(int a, int b, int c, float ct) : x(a), y(b), z(c), cost(ct) {} 

    vec3 operator+() const { return *this; }
    vec3 operator-() const { return vec3(-x, -y, -z); }

    vec3& operator+=(const vec3& other) {
        x += other.x; y += other.y; z += other.z;
        return *this;
    }
    vec3& operator-=(const vec3& other) {
        x -= other.x; y -= other.y; z -= other.z;
        return *this;
    }
    vec3& operator*=(int scalar) {
        x *= scalar; y *= scalar; z *= scalar;
        return *this;
    }
    vec3& operator/=(int scalar) {
        x /= scalar; y /= scalar; z /= scalar;
        return *this;
    }

    friend vec3 operator+(vec3 lhs, const vec3& rhs) {
        return lhs += rhs;
    }
    friend vec3 operator-(vec3 lhs, const vec3& rhs) {
        return lhs -= rhs;
    }
    friend vec3 operator*(vec3 lhs, int scalar) {
        return lhs *= scalar;
    }
    friend vec3 operator*(int scalar, vec3 rhs) {
        return rhs *= scalar;
    }
    friend vec3 operator/(vec3 lhs, int scalar) {
        return lhs /= scalar;
    }

    friend bool operator==(const vec3& lhs, const vec3& rhs) {
        return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
    }
    friend bool operator!=(const vec3& lhs, const vec3& rhs) {
        return !(lhs == rhs);
    }
    friend bool operator<(const vec3& lhs, const vec3& rhs) {
        if (lhs.x != rhs.x) return lhs.x < rhs.x;
        if (lhs.y != rhs.y) return lhs.y < rhs.y;
        return lhs.z < rhs.z;
    }

    friend std::ostream& operator<<(std::ostream& os, const vec3& v) {
        return os << '(' << v.x << ", " << v.y << ", " << v.z << ')';
    }

    int& operator[](int index) {
        if (index == 0) return x;
        else if (index == 1) return y;
        else return z;
    }
    const int& operator[](int index) const {
        if (index == 0) return x;
        else if (index == 1) return y;
        else return z;
    }
};


class state {
public:
    state() : point(), g(0.0f), rhs(0.0f), rhs_from(nullptr), key({0.0f, 0.0f}), occupied(false) {}
    state(int a, int b, int c) : point(a,b,c), g(0.0f), rhs(0.0f), rhs_from(nullptr), key({0.0f, 0.0f}), occupied(false) {}

    void  setG(float v)               { g = v; }
    void  setRHS(float v)             { rhs = v; }
    float getG() const                { return g; }
    float getRHS() const              { return rhs; }

    void  setPoint(vec3 pt)           { point = pt; }
    vec3  getPoint() const            { return point; }

    bool  isConsistent() const        { const float EPS = 1e-5f; return std::abs(g - rhs) < EPS; }
    bool  isOverConsistent() const    { return g > rhs; }

    std::shared_ptr<state> nextStep() const      { return rhs_from; }
    void  setNextStep(std::shared_ptr<state> u)  { rhs_from = std::move(u); }

    // key API (lowercase is canonical; CamelCase alias provided)
    void setkey(const std::pair<float,float>& k) { key = k; }
    void setKey (const std::pair<float,float>& k){ setkey(k); }
    std::pair<float,float> getKey() const        { return key; }

    void setOccupation(bool v)        { occupied = v; }
    bool getOccupy() const            { return occupied; }

private:
    vec3 point;
    float g, rhs;
    std::shared_ptr<state> rhs_from;
    std::pair<float,float> key;
    bool occupied;
};

inline bool isEqual(float a, float b) {
    if (abs(a - b) <= TOL) {
        return true;
    }
    return false;
}



inline bool compareNodes(std::shared_ptr<state> n1, std::shared_ptr<state> n2) {
    return n1->getRHS() > n2->getRHS();
}


struct StateComparator {
    bool operator()(const std::pair<std::pair<float, float>, std::shared_ptr<state>>& a,
                   const std::pair<std::pair<float, float>, std::shared_ptr<state>>& b) const {
        if (a.first.first != b.first.first) {
            return a.first.first > b.first.first; 
        }
        return a.first.second > b.first.second;  
    }
};


#endif