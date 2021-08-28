#include "CNNHelper.hpp"

#include "FaceDetector.hpp"
namespace aoce {

int32_t getPixelType(ImageType imageType) {
    switch (imageType) {
        case ImageType::bgra8:
            return ncnn::Mat::PIXEL_BGRA;
        case ImageType::rgba8:
            return ncnn::Mat::PIXEL_RGBA;
        case ImageType::r8:
            return ncnn::Mat::PIXEL_GRAY;
        case ImageType::bgr8:
            return ncnn::Mat::PIXEL_BGR;
        case ImageType::rgb8:
            return ncnn::Mat::PIXEL_RGB;
        default:
            return 0;
    }
}

ncnn::Mat getMat(uint8_t* data, const ImageFormat& inFormat,
                 const ImageFormat& outFormat) {
    ncnn::Mat in = {};
    int32_t inPixel = getPixelType(inFormat.imageType);
    int32_t outPixel = getPixelType(outFormat.imageType);
    if (inPixel > 0) {
        ncnn::Mat::PixelType pixelType = (ncnn::Mat::PixelType)inPixel;
        if (inPixel != outPixel) {
            pixelType = (ncnn::Mat::PixelType)(inPixel | (outPixel << 16));
        }
        if (inFormat.width == outFormat.width &&
            inFormat.height == outFormat.height) {
            in = ncnn::Mat::from_pixels(data, pixelType, inFormat.width,
                                        inFormat.height);
        } else {
            in = ncnn::Mat::from_pixels_resize(data, pixelType, inFormat.width,
                                               inFormat.height, outFormat.width,
                                               outFormat.height);
        }
    }
    return in;
}

IFaceDetector* createFaceDetector() {
    FaceDetector* detector = new FaceDetector();
    return detector;
}

}  // namespace aoce