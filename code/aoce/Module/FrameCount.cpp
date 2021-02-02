#include "FrameCount.hpp"

namespace aoce {

FrameCount::FrameCount(/* args */) {}

FrameCount::~FrameCount() {}

void FrameCount::run() {
	run(0.f);
}

void FrameCount::run(float sum) {
	auto delta = getNowTimeStamp() - lastTime;
	//第一次或是间隔太久
	if (lastTime == 0 || delta > totalTime * 5) {
		lastTime = getNowTimeStamp();
		runCount = 0;
		tempTime = 0;
		avageValue = 0.f;
		totalValue = 0.f;
		return;
	}
	runCount++;
	totalValue += sum;
	tempTime += delta;
	if (tempTime > totalTime) {
		tempTime -= totalTime;
		avageValue = totalValue / ((float)totalTime / 1000.f);
		fps = ((float)runCount) / ((float)totalTime / 1000.f);
		totalValue = 0;
		runCount = 0;
	}
	lastTime = getNowTimeStamp();
}

}  // namespace aoce