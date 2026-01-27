#ifndef OPENCV_FEATURES_DEEP_FEATURES_HPP
#define OPENCV_FEATURES_DEEP_FEATURES_HPP

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/dnn.hpp"

namespace cv {

/** @brief Class implementing the DISK (Deep Image Structure and Keypoints) feature detector
 *
 * Wrapping the inference of the ONNX model provided by LightGlue-ONNX.
 */
class CV_EXPORTS_W DISK : public Feature2D  // Changed to CV_EXPORTS_W for Python support
{
public:
    /**
     * @param modelPath Path to the ONNX model file.
     * @param backendId The DNN backend to use (default: DNN_BACKEND_DEFAULT).
     * @param targetId The DNN target to use (default: DNN_TARGET_CPU).
     */
    CV_WRAP static Ptr<DISK> create(const String& modelPath,
                                    int backendId = dnn::DNN_BACKEND_DEFAULT,
                                    int targetId = dnn::DNN_TARGET_CPU); // Added CV_WRAP

    virtual String getDefaultName() const CV_OVERRIDE;
};

} // namespace cv

#endif // OPENCV_FEATURES_DEEP_FEATURES_HPP