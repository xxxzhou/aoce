#pragma once

#include "AoceNcnnExport.h"
#include "net.h"

namespace aoce {

class AoceInterop {
   private:
    /* data */
   public:
    AoceInterop(/* args */);
    ~AoceInterop();

   public:
    void getTest(ncnn::Net* net);
};

}  // namespace aoce