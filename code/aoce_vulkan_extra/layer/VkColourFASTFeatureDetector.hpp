#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "VkSeparableLinearLayer.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

// 按GPUImageColourFASTFeatureDetector里来说,应该先box blur一次.
// 然后把box blur纹理给VkColourFASTFeatureDetector第二个输入.
// 但是实际上在fastFeatureDetector中第二纹理并没参与任何事情.

class VkColourFASTFeatureDetector : public VkLayer,
                                    public ITLayer<FASTFeatureParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkColourFASTFeatureDetector)
   private:
    /* data */
    std::unique_ptr<VkBoxBlurSLayer> boxBlur = nullptr;

   public:
    VkColourFASTFeatureDetector(/* args */);
    ~VkColourFASTFeatureDetector();

   protected:
    virtual void onUpdateParamet() override;
    virtual bool getSampled(int inIndex) override;
    virtual void onInitGraph() override;
    virtual void onInitNode() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce