#ifndef OPENCV_FEATURES_DEEP_FEATURES_HPP
#define OPENCV_FEATURES_DEEP_FEATURES_HPP

#include "opencv2/features.hpp"
#include "opencv2/dnn.hpp" // Required for Backend/Target enums

namespace cv {

/** @brief Class implementing the DISK (Deep Image Structure and Keypoints) feature detector
 *
 * Wrapping the inference of the ONNX model provided by LightGlue-ONNX.
 */
class CV_EXPORTS DISK : public Feature2D
{
public:
    /**
     * @param modelPath Path to the ONNX model file.
     * @param backendId The DNN backend to use (default: DNN_BACKEND_DEFAULT).
     * @param targetId The DNN target to use (default: DNN_TARGET_CPU).
     */
    static Ptr<DISK> create(const String& modelPath,
                            int backendId = dnn::DNN_BACKEND_DEFAULT,
                            int targetId = dnn::DNN_TARGET_CPU);

    virtual String getDefaultName() const CV_OVERRIDE;
};

} // namespace cv

#endif // OPENCV_FEATURES_DEEP_FEATURES_HPP