
#include "PipeGraph.hpp"

#include <cmath>
#include <list>
#include <queue>
#include <stack>

namespace aoce {
PipeGraph::PipeGraph(/* args */) {}

PipeGraph::~PipeGraph() {}

PipeNodePtr PipeGraph::addNode(BaseLayer* layer) {
    if (layer == nullptr) {
        logMessage(LogLevel::error, "node layer can not be empty");
    }
    assert(layer != nullptr);    
    if (this->gpu != layer->gpu) {
        logMessage(LogLevel::error, "node layer gpu type no equal graph");
    }
    assert(this->gpu == layer->gpu);
    layer->pipeGraph = this;
    layer->onInit();
    PipeNodePtr ptr(new PipeNode(layer));
    ptr->graphIndex = (int32_t)nodes.size();
    nodes.push_back(ptr);
    return ptr;
}

PipeNodePtr PipeGraph::addNode(ILayer* layer) {
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

bool PipeGraph::addLine(PipeNodePtr from, PipeNodePtr to, int32_t formOut,
                        int32_t toIn) {
    return addLine(from->graphIndex, to->graphIndex, formOut, toIn);
}

void PipeGraph::getImageFormat(int32_t nodeIndex, int32_t outputIndex,
                               ImageFormat& format) {
    if (nodeIndex < nodes.size() &&
        outputIndex < nodes[nodeIndex]->layer->outFormats.size()) {
        format = nodes[nodeIndex]->layer->outFormats[outputIndex];
    }
}

void PipeGraph::validNode() {
    // 本节点的输入线
    std::vector<std::vector<PipeLinePtr>> toLines(nodes.size());
    // 本节点的输出线
    std::vector<std::vector<PipeLinePtr>> formLines(nodes.size());
    validLines.clear();
    for (auto& line : lines) {
        toLines[line->toNode].push_back(line);
        formLines[line->fromNode].push_back(line);
    }
    // 检查节点自身的输入输出线是否正确(注意,可能一个接点有多个输入,配合后面disable检查去掉相应)
    for (auto& node : nodes) {
        // 检查输入节点
        if (!node->bDisable && !node->layer->bInput) {
            auto& tns = toLines[node->graphIndex];
            uint32_t tmask = 0;
            for (auto& tn : tns) {
                uint32_t maskIndex = std::pow(2, tn->toInIndex);
                tmask |= maskIndex;
            }
            int32_t inCount = node->layer->inCount;
            // 输入节点差
            if (tmask != (std::pow(2, inCount) - 1)) {
                node->bDisable = true;
            }
        }
        // 检查输出结点,要检查输出节点吗?
    }
    // 检查有效链路
    for (auto& node : nodes) {
        if (!node->bDisable && node->layer->bInput) {
            std::stack<int32_t> inputNodes;
            inputNodes.push(node->graphIndex);
            // 查找所有有效链路
            while (!inputNodes.empty()) {
                int32_t inputNode = inputNodes.top();
                inputNodes.pop();  // stack深度优先搜索
                auto& nextLines = formLines[inputNode];
                for (auto line : nextLines) {
                    if (nodes[line->toNode]->bDisable) {
                        continue;
                    }
                    // 不可见的话,尝试自动去掉当前节点链接下一节点
                    if (nodes[line->toNode]->bInvisible) {
                        if (toLines[line->toNode].size() == 1 &&
                            formLines[line->toNode].size() == 1) {
                            PipeLinePtr nline(new PipeLine());
                            // 新线取当前节点的输入线
                            nline->fromNode = toLines[nline->toNode][0]->toNode;
                            nline->fromOutIndex =
                                toLines[nline->toNode][0]->toInIndex;
                            nline->toNode =
                                formLines[nline->toNode][0]->fromNode;
                            nline->toInIndex =
                                formLines[nline->toNode][0]->fromOutIndex;
                            // 检查是否已经有值,注意,这里nline在validLines没有,检查值相等
                            auto find = std::find_if(validLines.begin(),
                                                     validLines.end(),
                                                     [&nline](PipeLinePtr ptr) {
                                                         return *ptr == *nline;
                                                     });
                            if (find == validLines.end()) {
                                validLines.push_back(nline);
                                inputNodes.push(
                                    toLines[nline->toNode][0]->toNode);
                            }
                        }
                        continue;
                    }
                    // 这个链路已经检查过,不需要在检查
                    if (std::find(validLines.begin(), validLines.end(), line) ==
                        validLines.end()) {
                        validLines.push_back(line);
                        inputNodes.push(line->toNode);
                    }
                }
            }
        }
    }
}

bool PipeGraph::resetGraph() {
    nodeExcs.clear();
    // 1. 得到节点的enable/visable验证过的lines
    validNode();
    // 2. 得到有效节点的前置节点  如:2[1] 3[2] 4[1] 5[3,4] (2需要1),(5需要3,4)
    std::vector<std::vector<int>> reqNodes(nodes.size());
    std::queue<int32_t> tempQueue;
    for (auto& line : validLines) {
        reqNodes[line->toNode].push_back(line->fromNode);
        tempQueue.push(line->toNode);
        // 填充layer的,此时经过enable/visable过滤后,每个输入节点应该是一一对应的
        nodes[line->toNode]->layer->addInLayer(line->toInIndex, line->fromNode,
                                               line->fromOutIndex);
    }
    for (int32_t i = 0; i < nodes.size(); i++) {
        const auto& reqNode = reqNodes[i];
        if (reqNode.size() == 0 || nodes[i]->layer->bInput) {
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
            string_format(message, "layer vaild error,node: ", index);
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
                    ->layer->outFormats[fromNode.outputIndex]
                    .imageType != nodes[index]->layer->inFormats[i].imageType) {
                std::string message;
                string_format(message, "graph error,",
                              "from node: ", fromNode.nodeIndex, "-",
                              fromNode.outputIndex,
                              " not match image type in node:", index, "-", i);
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
    if (bReset) {
        bReset = false;
        if (!resetGraph()) {
            return false;
        }
    }    
    return onRun();
}

}  // namespace aoce
