#ifndef OPENCV_FEATURES_DEEP_FEATURES_HPP
#define OPENCV_FEATURES_DEEP_FEATURES_HPP

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp" 

namespace cv {

/** @brief Class implementing the DISK (Deep Image Structure and Keypoints) feature detector
 *
 * Wrapping the inference of the ONNX model provided by LightGlue-ONNX.
 */
class CV_EXPORTS DISK : public Feature2D // Changed from CV_EXPORTS_W to CV_EXPORTS
{
public:
    /**
     * @param modelPath Path to the ONNX model file.
     * @param backendId The DNN backend to use (default: DNN_BACKEND_DEFAULT).
     * @param targetId The DNN target to use (default: DNN_TARGET_CPU).
     */
    static Ptr<DISK> create(const String& modelPath,
                            int backendId = 0,
                            int targetId = 0); // Removed CV_WRAP

    virtual String getDefaultName() const CV_OVERRIDE;
};

} // namespace cv

#endif // OPENCV_FEATURES_DEEP_FEATURES_HPP