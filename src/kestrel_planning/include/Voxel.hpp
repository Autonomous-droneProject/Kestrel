#ifndef VOXEL_HPP
#define VOXEL_HPP

#pragma once

#include <vector>
#include <cstdint>
#include <memory>
#include <array>
#include <stdexcept>
#include <cmath>
#include <algorithm>

#include "utils.hpp"

namespace octree {

template <typename T>
struct Node {
    std::array<std::shared_ptr<Node<T>>, 8> children{};
    std::shared_ptr<T> value;
};

template <typename T>
class voxel {
public:
    voxel(uint32_t X, uint32_t Y, uint32_t Z)
        : X_RANGE(X), Y_RANGE(Y), Z_RANGE(Z) {
        uint32_t max_dim = std::max({X_RANGE, Y_RANGE, Z_RANGE});
        max_depth = static_cast<uint32_t>(std::ceil(std::log2(max_dim)));
        root = std::make_shared<Node<T>>();
    }
    voxel() : X_RANGE(0), Y_RANGE(0), Z_RANGE(0), max_depth(0), root(nullptr) {}

    

    std::shared_ptr<T>& operator()(uint32_t x, uint32_t y, uint32_t z) {
        return access(root, x, y, z, 0);
    }

    const std::shared_ptr<T>& operator()(uint32_t x, uint32_t y, uint32_t z) const {
        return access_const(root, x, y, z, 0);
    }

    void insert(std::shared_ptr<T> val, const vec3& pt) {
        auto& target = access(root, pt.x, pt.y, pt.z, 0);
        target = val;
    }

    void clear() {
        clearNode(root);
    }

private:
    uint32_t X_RANGE, Y_RANGE, Z_RANGE;
    uint32_t max_depth;
    std::shared_ptr<Node<T>> root;

    int childIndex(uint32_t x, uint32_t y, uint32_t z, uint32_t level) const {
        int shift = max_depth - level - 1;
        return ((x >> shift) & 1) << 2 |
               ((y >> shift) & 1) << 1 |
               ((z >> shift) & 1);
    }

    std::shared_ptr<T>& access(std::shared_ptr<Node<T>> node, uint32_t x, uint32_t y, uint32_t z, uint32_t level) {
        if (level == max_depth) {
            if (!node->value)
                node->value = std::make_shared<T>();
            return node->value;
        }

        int idx = childIndex(x, y, z, level);
        if (!node->children[idx])
            node->children[idx] = std::make_shared<Node<T>>();

        return access(node->children[idx], x, y, z, level + 1);
    }

    const std::shared_ptr<T> access_const(std::shared_ptr<Node<T>> node, uint32_t x, uint32_t y, uint32_t z, uint32_t level) const {
        if (x >= X_RANGE || y >= Y_RANGE || z >= Z_RANGE)
            return nullptr;

        if (level == max_depth) {
            return node->value;
        }

        int idx = childIndex(x, y, z, level);
        if (!node->children[idx])
            return nullptr;

        return access_const(node->children[idx], x, y, z, level + 1);
    }

    void clearNode(std::shared_ptr<Node<T>>& node) {
        if (!node) return;
        node->value.reset();
        for (auto& child : node->children)
            clearNode(child);
    }
};

} 

#endif