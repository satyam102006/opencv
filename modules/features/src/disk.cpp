#include "precomp.hpp"
#include "opencv2/deep_features.hpp"
#include <opencv2/dnn.hpp>

namespace cv {

using namespace dnn;

class DISK_Impl : public DISK {
public:

    DISK_Impl(const String& _modelPath, int _backendId, int _targetId)
        : modelPath(_modelPath), backendId(_backendId), targetId(_targetId)
    {
        net = readNetFromONNX(modelPath);
        net.setPreferableBackend(backendId);
        net.setPreferableTarget(targetId);
    }

    void detectAndCompute(InputArray _image, InputArray _mask,
                          std::vector<KeyPoint>& keypoints,
                          OutputArray _descriptors,
                          bool /*useProvidedKeypoints*/) CV_OVERRIDE {

        Mat image = _image.getMat();
        if (image.empty()) return;

        // 1. Preprocessing (DISK expects 1024x1024)
        const int inputW = 1024;
        const int inputH = 1024;

        float scaleX = (float)image.cols / inputW;
        float scaleY = (float)image.rows / inputH;
        Mat inputBlob;
        blobFromImage(image, inputBlob, 1.0/255.0, Size(inputW, inputH), Scalar(), true, false);

        net.setInput(inputBlob, "image");

        // 2. Inference
        std::vector<String> outNames = {"keypoints", "scores", "descriptors"};
        std::vector<Mat> outs;
        net.forward(outs, outNames);

        // 3. Parse Outputs
        Mat kptsBlob = outs[0]; // [1, N, 2]
        Mat scoresBlob = outs[1];    // [1, N]
        Mat descBlob = outs[2];      // [1, N, 128]

        int numFeatures = kptsBlob.size[1];
        float* kptsData = (float*)kptsBlob.data;
        float* scoresData = (float*)scoresBlob.data;

        keypoints.clear();
        std::vector<int> validIndices;
        validIndices.reserve(numFeatures); // Optimization

        for (int i = 0; i < numFeatures; ++i) {
            float score = scoresData[i];
            if (score > 0.0f) {
                float x = kptsData[i * 2] * scaleX;
                float y = kptsData[i * 2 + 1] * scaleY;

                KeyPoint kp(x, y, 1.0f, -1, score);
                keypoints.push_back(kp);
                validIndices.push_back(i);
            }
        }

        // 4. Filter Descriptors
        if (_descriptors.needed()) {
            // Read dimension from the blob instead of hardcoding '128'
            int dim = descBlob.size[2];
            _descriptors.create((int)validIndices.size(), dim, CV_32F);
            Mat descriptors = _descriptors.getMat();

            // Flatten batch: [1, N, D] -> [N, D]
            Mat flatDesc(numFeatures, dim, CV_32F, descBlob.ptr<float>());

            for (size_t i = 0; i < validIndices.size(); ++i) {
                flatDesc.row(validIndices[i]).copyTo(descriptors.row((int)i));
            }
        }
    }

    String getDefaultName() const CV_OVERRIDE {
        return "Feature2D.DISK";
    }

private:
    String modelPath;
    int backendId;
    int targetId;
    Net net;
};

String DISK::getDefaultName() const {
    return "Feature2D.DISK";
}

// Updated factory method
Ptr<DISK> DISK::create(const String& modelPath, int backendId, int targetId) {
    return makePtr<DISK_Impl>(modelPath, backendId, targetId);
}

} // namespace cv