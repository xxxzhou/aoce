#pragma once

#include "../Aoce.hpp"
namespace aoce {

// 可用来统计桢率或是码率
class ACOE_EXPORT FrameCount {
   private:
    /* data */
    // 每次间隔时间 毫秒
    int64_t totalTime = 1000;
    float fps = 1.f;
    // 间隔时间内平均值
    float avageValue = 0.f;
    // 间隔时间内运行次数
    int32_t runCount = 0;
    float totalValue = 0.f;
    int64_t tempTime = 0;
    int64_t lastTime = 0;

   public:
    FrameCount(/* args */);
    ~FrameCount();

   public:
    void run();
    void run(float sum);
    // 统计的更新时间(默认一秒更新一次)
    inline void setDelteTime(int64_t ms) { totalTime = ms; }
    // 得到统计的平均桢率
    inline float getFps() { return fps; }
    // 得到每次运行传入sum(vaule)数据的平均值
    inline float getNum() { return avageValue; }
};

}  // namespace aoce