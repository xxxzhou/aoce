#pragma once
#include "PipeGraph.hpp"

namespace aoce {
PipeGraph::PipeGraph(/* args */) {}

PipeGraph::~PipeGraph() {}

PipeNodePtr PipeGraph::addInputNode(InputLayer* layer) {
    if (this->gpu != layer->gpu) {
        return nullptr;
    }
    layer->pipeGraph = this;
    PipeNodePtr ptr(new PipeNode(layer));
    inputNodes.push_back(ptr);
    return ptr;
}

bool PipeGraph::resetGraph(const PipeNodePtr& node) {
    if (node->layer == nullptr) {
        return false;
    }
    for (int32_t i = 0; i < node->nodes.size(); i++) {
        const ChildNode& childNode = node->nodes[i];
        const auto* childLayer = childNode.node->layer;
        if (childLayer == nullptr) {
            return false;
        }
        if (childLayer->inputFormats[childNode.index].imageType !=
            node->layer->outputFormats[i].imageType) {
            logMessage(LogLevel::error, "not match imagetype");
            return false;
        }
    }
}

bool PipeGraph::resetGraph() {
    for (auto& node : inputNodes) {
        for (int32_t i = 0; i < node->nodes.size(); i++) {
            const ChildNode& childNode = node->nodes[i];
            aoce::BaseLayer* layer = childNode.node->layer;

            if (childNode.node->layer->inputFormats[childNode.index]
                    .imageType != node->layer->outputFormats[i].imageType) {
                logMessage(LogLevel::error, "not match imagetype");
                return false;
            }
        }
    }
    return true;
}

bool PipeGraph::run() {
    if (bReset) {
        bReset = false;
        if (!resetGraph()) {
            return false;
        }
    }
    return true;
}

}  // namespace aoce
