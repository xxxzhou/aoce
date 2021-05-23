
#include "PipeGraph.hpp"

#include <cmath>
#include <list>
#include <queue>
#include <stack>

#include "OutputLayer.hpp"

namespace aoce {

PipeGraph::PipeGraph(/* args */) {}

PipeGraph::~PipeGraph() {}

GpuType PipeGraph::getGpuType() { return gpu; }

void PipeGraph::reset() { bReset = true; }

IBaseLayer* PipeGraph::getNode(int32_t index) {
    int32_t count = (int32_t)nodes.size();
    if (index >= 0 && index < count) {
        return nodes[index]->layer;
    }
    return nullptr;
}

IBaseLayer* PipeGraph::addNode(IBaseLayer* ilayer) {
    BaseLayer* layer = static_cast<BaseLayer*>(ilayer);
    logAssert(layer != nullptr, "node layer can not be empty");
    if (!layer->bNoCompute) {
        logAssert(this->gpu == layer->gpu,
                  "node layer gpu type no equal graph");
    }
    if (layer->pipeGraph != nullptr) {
        logAssert(layer->pipeGraph != this, "layer have attach other graph.");
        std::string message;
        string_format(
            message,
            "layer already in graph,may generate problems,node:", nodes.size());
        logMessage(LogLevel::error, message);
    }
    layer->pipeGraph = this;
    layer->onInit();
    PipeNodePtr ptr(new PipeNode(layer));
    ptr->graphIndex = (int32_t)nodes.size();
    layer->pipeNode = ptr;
    nodes.push_back(ptr);
    layer->onInitNode();
    return layer;
}

IBaseLayer* PipeGraph::addNode(ILayer* layer) {
    assert(layer != nullptr);
    return addNode(layer->getLayer());
}

bool PipeGraph::addLine(int32_t from, int32_t to, int32_t formOut,
                        int32_t toIn) {
    PipeLinePtr line(new PipeLine());
    line->fromNode = from;
    line->fromOutIndex = formOut;
    line->toNode = to;
    line->toInIndex = toIn;
    // 数据连接节点无效
    if (formOut >= nodes[from]->layer->outCount ||
        toIn >= nodes[to]->layer->inCount) {
        return false;
    }
    // 节点有效性与重复性
    if (line->valid() &&
        std::find(lines.begin(), lines.end(), line) == lines.end()) {
        lines.push_back(line);
        return true;
    }
    return false;
}

bool PipeGraph::addLine(IBaseLayer* from, IBaseLayer* to, int32_t formOut,
                        int32_t toIn) {
    return addLine(from->getGraphIndex(), to->getGraphIndex(), formOut, toIn);
}

bool PipeGraph::getLayerOutFormat(int32_t nodeIndex, int32_t outputIndex,
                                  ImageFormat& format, bool bOutput) {
    if (nodeIndex < nodes.size() &&
        outputIndex < nodes[nodeIndex]->layer->outFormats.size()) {
        auto& tempFormat = nodes[nodeIndex]->layer->outFormats[outputIndex];
        format.width = tempFormat.width;
        format.height = tempFormat.height;
        if (bOutput) {
            format.imageType = tempFormat.imageType;
        }
        return true;
    }
    return false;
}

bool PipeGraph::getLayerInFormat(int32_t nodeIndex, int32_t inputIndex,
                                 ImageFormat& format) {
    if (nodeIndex < nodes.size() &&
        inputIndex < nodes[nodeIndex]->layer->inFormats.size()) {
        auto& tempFormat = nodes[nodeIndex]->layer->inFormats[inputIndex];
        format.width = tempFormat.width;
        format.height = tempFormat.height;
        format.imageType = tempFormat.imageType;
        return true;
    }
    return false;
}

void PipeGraph::clearLines() {
    lines.clear();
    validLines.clear();
    bReset = true;
}

void PipeGraph::clear() {
    clearLines();
    // 置空,免指向野指针
    for (auto& node : nodes) {
        node->getLayer()->pipeGraph = nullptr;
    }
    // layer weak_ptr node,故node清空后,layer的node指向自动不可用
    nodes.clear();
    nodeExcs.clear();
    bReset = true;
}

bool PipeGraph::checkHaveValid(PipeLinePtr nline) {
    // 检查是否已经有值,注意,这里nline可能在validLines没有,检查值相等
    auto find =
        std::find_if(validLines.begin(), validLines.end(),
                     [&nline](PipeLinePtr ptr) { return *ptr == *nline; });
    return find != validLines.end();
}

void PipeGraph::validNode() {
    // 本节点的输出线
    std::vector<std::vector<PipeLinePtr>> formLines(nodes.size());
    validLines.clear();
    for (auto& line : lines) {
        formLines[line->fromNode].push_back(line);
    }
    // 检查有效链路,保存到validLines
    for (auto& node : nodes) {
        // 从可用的输入节点向下深度优先搜索
        if (!node->bDisable && !node->bInvisible && node->layer->bInput) {
            // stack深度优先搜索
            std::stack<int32_t> inputNodes;
            inputNodes.push(node->graphIndex);
            int32_t nodeIndex = node->graphIndex;
            // 查找所有有效链路
            while (!inputNodes.empty()) {
                int32_t currentIndex = inputNodes.top();
                inputNodes.pop();
                // 当前节点向下连接
                auto& nextLines = formLines[currentIndex];
                // 当前节点不可用,继续用上一个可用节点
                if (!nodes[currentIndex]->bInvisible) {
                    nodeIndex = currentIndex;
                }
                // 验证这些连接是否有效
                for (auto line : nextLines) {
                    // 如果当前节点不可用,这条线就不可用了
                    int toNode = line->toNode;
                    if (nodes[toNode]->bDisable) {
                        continue;
                    }
                    // 当前连接节点不可见的话,尝试自动去掉当前节点链接下一节点
                    if (nodes[toNode]->bInvisible) {
                        inputNodes.push(line->toNode);
                        continue;
                    }
                    PipeLinePtr nline(new PipeLine());
                    nline->fromNode = nodeIndex;
                    nline->fromOutIndex = line->fromOutIndex;
                    nline->toNode = toNode;
                    nline->toInIndex = line->toInIndex;
                    // 正常情况,检查是否已经搜索过.
                    if (!checkHaveValid(nline)) {
                        validLines.push_back(nline);
                        inputNodes.push(toNode);
                    }
                }
            }
        }
    }
}

bool PipeGraph::resetGraph() {
    // 1. 得到节点的enable/visable验证过的lines
    validNode();
    // 2. 得到有效节点的前置节点  如:2[1] 3[2] 4[1] 5[3,4]
    // (2需要1),(5需要3,4)
    std::vector<std::vector<int>> reqNodes(nodes.size());
    // 需要确定顺序的节点
    std::queue<int32_t> tempQueue;
    std::vector<int> checkRepeat;
    for (auto& line : validLines) {
        reqNodes[line->toNode].push_back(line->fromNode);
        if (std::find(checkRepeat.begin(), checkRepeat.end(), line->toNode) ==
            checkRepeat.end()) {
            checkRepeat.push_back(line->toNode);
            tempQueue.push(line->toNode);
        }
        // 填充layer的,此时经过enable/visable过滤后,每个输入节点应该是一一对应的
        nodes[line->toNode]->layer->addInLayer(line->toInIndex, line->fromNode,
                                               line->fromOutIndex);
        nodes[line->fromNode]->layer->addOutLayer(
            line->fromOutIndex, line->toNode, line->toInIndex);
    }
    nodeExcs.clear();
    for (int32_t i = 0; i < nodes.size(); i++) {
        const auto& reqNode = reqNodes[i];
        if (nodes[i]->layer->bInput) {
            nodeExcs.push_back(i);
        }
    }
    // 3. 重新构建有序无环图的执行顺序
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
    // 4. onInitLayer,节点得到所需大小
    for (auto index : nodeExcs) {
        if (!nodes[index]->layer->vaildInLayers()) {
            std::string message;
            string_format(message, nodes[index]->layer->getMark(),
                          " vaild error ");
            logMessage(LogLevel::error, message);
            return false;
        }
        nodes[index]->layer->initLayer();
    }
    // 5. 检查节点连接的ImageType是否符合
    for (auto& index : nodeExcs) {
        if (nodes[index]->layer->bInput) {
            continue;
        }
        int size = nodes[index]->layer->inLayers.size();
        for (int i = 0; i < size; i++) {
            const auto& fromNode = nodes[index]->layer->inLayers[i];
            if (nodes[fromNode.nodeIndex]
                    ->layer->outFormats[fromNode.siteIndex]
                    .imageType != nodes[index]->layer->inFormats[i].imageType) {
                std::string message;
                string_format(message, "graph error,", "from : ",
                              nodes[fromNode.nodeIndex]->layer->getMark(), "-",
                              fromNode.siteIndex, " not match image type in :",
                              nodes[index]->layer->getMark(), "-", i);
                logMessage(LogLevel::error, message);
                return false;
            }
        }
    }
    onInitLayers();
    for (auto index : nodeExcs) {
        nodes[index]->layer->onInitBuffer();
    }    
    onInitBuffers();
    return true;
}

bool PipeGraph::run() {
    // 保证PipeGraph对象一次只处理一桢(MF
    // 异步模式每次读取的数据可能并不在同一线程上)
    std::lock_guard<std::mutex> mtx_locker(mtx);
    if (bReset) {
        logMessage(LogLevel::info, "start build graph.");
#if AOCE_DEBUG_TYPE
        logMessage(LogLevel::info, "---the graph all node name.");
        for (auto node : nodes) {
            logMessage(LogLevel::info, node->layer->getMark());
        }
        logMessage(LogLevel::info, "--- end graph.");
#endif
        bReset = false;
        onReset();
        if (!resetGraph()) {
            logMessage(LogLevel::warn, "build graph failed.");
            return false;
        } else {
            logMessage(LogLevel::info, "build graph success.");
#if AOCE_DEBUG_TYPE
            logMessage(LogLevel::info,
                       "--- the order of execution of the graph.");
            for (auto index : nodeExcs) {
                logMessage(LogLevel::info, nodes[index]->layer->getMark());
            }
            logMessage(LogLevel::info, "--- end graph.");
#endif
        }
    }
    return onRun();
}

}  // namespace aoce
