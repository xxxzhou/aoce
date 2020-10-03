#pragma once
#include "PipeGraph.hpp"

#include <list>
#include <queue>

namespace aoce {
PipeGraph::PipeGraph(/* args */) {}

PipeGraph::~PipeGraph() {}

PipeNodePtr PipeGraph::addNode(BaseLayer* layer) {
    if (this->gpu != layer->gpu) {
        return nullptr;
    }
    layer->pipeGraph = this;
    PipeNodePtr ptr(new PipeNode(layer));
    ptr->graphIndex = (int32_t)nodes.size();
    nodes.push_back(ptr);
    return ptr;
}

void PipeGraph::addLine(PipeNodePtr from, PipeNodePtr to, int32_t formOut,
                        int32_t toIn) {
    PipeLine line = {};
    line.fromNode = from->graphIndex;
    line.fromOutIndex = formOut;
    line.toNode = to->graphIndex;
    line.toInIndex = toIn;
    if (std::find(lines.begin(), lines.end(), line) == lines.end()) {
        lines.push_back(std::move(line));
    }
}

bool PipeGraph::resetGraph() {
    nodeExcs.clear();
    // 每个节点需要的前置节点
    std::vector<std::vector<int>> reqNodes(nodes.size());
    // 保存还没查找到前置节点
    std::queue<int32_t> tempQueue;
    for (auto& line : lines) {
        reqNodes[line.toNode].push_back(line.fromNode);
        tempQueue.push(line.toNode);
    }   
    // 重新构建有序无环图的执行顺序
    while (!tempQueue.empty()) {
        int32_t tnode = tempQueue.front();
        tempQueue.pop();
        auto& reqNode = reqNodes[tnode];
        bool bfind = true;
        // 如果当前的前置节点已经全部添加到nodeExcs中,则可以放入执行列表中
        for (auto& rnode : reqNode) {
            if (std::find(nodeExcs.begin(), nodeExcs.end(), rnode) ==
                nodeExcs.end()) {
                bfind = false;
                break;
            }
        }
        if (bfind) {
            nodeExcs.push_back(tnode);
        } else {
            tempQueue.push(tnode);
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
