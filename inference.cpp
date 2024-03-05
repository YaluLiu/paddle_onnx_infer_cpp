#include "inference.h"
#include <regex>

// #define benchmark
#define min(a,b)            (((a) < (b)) ? (a) : (b))
YOLO_V8::YOLO_V8() {

}


YOLO_V8::~YOLO_V8() {
    delete m_session;
}

#ifdef USE_CUDA
namespace Ort
{
    template<>
    struct TypeToTensorType<half> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; };
}
#endif


template<typename T>
char* BlobFromImage(cv::Mat& iImg, T& iBlob) {
    int channels = iImg.channels();
    int imgHeight = iImg.rows;
    int imgWidth = iImg.cols;

    for (int c = 0; c < channels; c++)
    {
        for (int h = 0; h < imgHeight; h++)
        {
            for (int w = 0; w < imgWidth; w++)
            {
                iBlob[c * imgWidth * imgHeight + h * imgWidth + w] = typename std::remove_pointer<T>::type(
                    (iImg.at<cv::Vec3b>(h, w)[c]) / 255.0f);
            }
        }
    }
    return RET_OK;
}


char* YOLO_V8::PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg)
{
    if (iImg.channels() == 3)
    {
        oImg = iImg.clone();
        cv::cvtColor(oImg, oImg, cv::COLOR_BGR2RGB);
    }
    else
    {
        cv::cvtColor(iImg, oImg, cv::COLOR_GRAY2RGB);
    }
    if (modelType == YOLO_PADDLE){
        cv::resize(oImg, oImg, cv::Size(iImgSize.at(0), iImgSize.at(1)), 0, 0, cv::INTER_NEAREST);
    } else if (modelType == YOLO_DETECT_V8){
        if (iImg.cols >= iImg.rows) {
            resizeScales = iImg.cols / (float)iImgSize.at(0);
            cv::resize(oImg, oImg, cv::Size(iImgSize.at(0), int(iImg.rows / resizeScales)));
        }
        else {
            resizeScales = iImg.rows / (float)iImgSize.at(0);
            cv::resize(oImg, oImg, cv::Size(int(iImg.cols / resizeScales), iImgSize.at(1)));
        }
        cv::Mat tempImg = cv::Mat::zeros(iImgSize.at(0), iImgSize.at(1), CV_8UC3);
        oImg.copyTo(tempImg(cv::Rect(0, 0, oImg.cols, oImg.rows)));
        oImg = tempImg;
    }
    m_img_h = iImg.rows;
    m_img_w = iImg.cols;
    return RET_OK;
}


char* YOLO_V8::CreateSession(DL_INIT_PARAM& iParams) {
    char* Ret = RET_OK;
    std::regex pattern("[\u4e00-\u9fa5]");
    bool result = std::regex_search(iParams.modelPath, pattern);
    if (result)
    {
        //防止警告，const str to char*
        char* err_info = new char[150];
        std::strcpy(err_info,"[YOLO_V8]:Your model path is error.Change your model path without chinese characters.");
        std::cout << err_info << std::endl;
        return err_info;
    }
    try
    { 
        rectConfidenceThreshold = iParams.rectConfidenceThreshold;
        iouThreshold = iParams.iouThreshold;
        imgSize = iParams.imgSize;
        modelType = iParams.modelType;
        m_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolo");
        Ort::SessionOptions sessionOption;
        if (iParams.cudaEnable)
        {
            m_cudaEnable = iParams.cudaEnable;
            OrtCUDAProviderOptions cudaOption;
            cudaOption.device_id = 0;
            sessionOption.AppendExecutionProvider_CUDA(cudaOption);
        }
        sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        sessionOption.SetIntraOpNumThreads(iParams.intraOpNumThreads);
        sessionOption.SetLogSeverityLevel(iParams.logSeverityLevel);

#ifdef _WIN32
        int ModelPathSize = MultiByteToWideChar(CP_UTF8, 0, iParams.modelPath.c_str(), static_cast<int>(iParams.modelPath.length()), nullptr, 0);
        wchar_t* wide_cstr = new wchar_t[ModelPathSize + 1];
        MultiByteToWideChar(CP_UTF8, 0, iParams.modelPath.c_str(), static_cast<int>(iParams.modelPath.length()), wide_cstr, ModelPathSize);
        wide_cstr[ModelPathSize] = L'\0';
        const wchar_t* modelPath = wide_cstr;
#else
        const char* modelPath = iParams.modelPath.c_str();
#endif // _WIN32
        
        
        m_session = new Ort::Session(m_env, modelPath, sessionOption);
        Ort::AllocatorWithDefaultOptions allocator;
        size_t inputNodesNum = m_session->GetInputCount();
        for (size_t i = 0; i < inputNodesNum; i++)
        {
            Ort::AllocatedStringPtr input_node_name = m_session->GetInputNameAllocated(i, allocator);
            char* temp_buf = new char[50];
            strcpy(temp_buf, input_node_name.get());
            inputNodeNames.push_back(temp_buf);
        }

        size_t OutputNodesNum = m_session->GetOutputCount();
        for (size_t i = 0; i < OutputNodesNum; i++)
        {
            Ort::AllocatedStringPtr output_node_name = m_session->GetOutputNameAllocated(i, allocator);
            char* temp_buf = new char[50];
            strcpy(temp_buf, output_node_name.get());
            outputNodeNames.push_back(temp_buf);
        }

        // std::cout << "--------------------------------" << std::endl;
        // for (int i = 0; i<inputNodeNames.size(); ++i)
        // {
        //   std::cout << inputNodeNames[i] << std::endl;
        // }
        // for (int i = 0; i<outputNodeNames.size(); ++i)
        // {
        //   std::cout << outputNodeNames[i] << std::endl;
        // }
        // std::cout << "--------------------------------" << std::endl;
        return RET_OK;
    }
    catch (const std::exception& e)
    {
        const char* str1 = "[YOLO_V8]:";
        const char* str2 = e.what();
        std::string result = std::string(str1) + std::string(str2);
        char* merged = new char[result.length() + 1];
        std::strcpy(merged, result.c_str());
        std::cout << merged << std::endl;
        delete[] merged;
        //防止警告，const str to char*
        char* err_info = new char[100];
        std::strcpy(err_info,"[YOLO_V8]:Create session failed.");
        return err_info;
    }

}


char* YOLO_V8::RunSession(cv::Mat& iImg, std::vector<DL_RESULT>& oResult) {
    char* Ret = RET_OK;
    cv::Mat processedImg;
    PreProcess(iImg, imgSize, processedImg);
    float* blob = new float[processedImg.total() * 3];
    BlobFromImage(processedImg, blob);
    if (modelType == YOLO_PADDLE) {
        PaddleProcess(iImg, blob, oResult);
    }
    else if(modelType == YOLO_DETECT_V8) {
        YOLO_origin_Process(iImg, blob, oResult);
    }
    // else {
    //     //处理half,FP16
    //     half* blob = new half[processedImg.total() * 3];
    //     BlobFromImage(processedImg, blob);
    //     PaddleProcess(iImg, blob, oResult);
    // }

    return Ret;
}


template<typename N>
char* YOLO_V8::PaddleProcess(cv::Mat& iImg, N& blob, std::vector<DL_RESULT>& oResult) {
    std::vector<int64_t> inputNodeDims = { 1, 3, imgSize.at(0), imgSize.at(1) };
    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(Ort::Value::CreateTensor<typename std::remove_pointer<N>::type>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1),
        inputNodeDims.data(), inputNodeDims.size()));

    //Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::MemoryInfo mem_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    Ort::RunOptions options = Ort::RunOptions{nullptr};

    // scale_factor, float, [x_factor, y_factor]
    auto input_w = imgSize.at(1);
    auto input_h = imgSize.at(0);
    float x_factor = input_w / m_img_w;
    float y_factor = input_h / m_img_h;
    // std::cout << "factor:" << x_factor << ", " << y_factor << std::endl;
    std::vector<float> input_1_data = {y_factor,x_factor};
    std::vector<int64_t> input_1_dims = {1, 2};
    ort_inputs.push_back(
      Ort::Value::CreateTensor<float>(mem_info, const_cast<float*>(input_1_data.data()),
                                      input_1_data.size(), input_1_dims.data(), input_1_dims.size()));

    auto outputTensor = m_session->Run(options, inputNodeNames.data(), ort_inputs.data(), inputNodeNames.size(),
                              outputNodeNames.data(), outputNodeNames.size());

    // 开始处理输出数据
    delete[] blob;
    Ort::TypeInfo typeInfo = outputTensor.front().GetTypeInfo();
    auto tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputNodeDims = tensor_info.GetShape();
    auto output = outputTensor.front().GetTensorMutableData<typename std::remove_pointer<N>::type>();
    int signalResultNum = outputNodeDims[0];//280
    int strideNum = outputNodeDims[1];//6
    cv::Mat rawData;
    rawData = cv::Mat(strideNum, signalResultNum, CV_32F, output);
    float* data = (float*)rawData.data;

    for (int i = 0; i < signalResultNum; ++i)
    {
        float confidence = float(data[1]);
        if(confidence > 0.5){
          int left = int(data[2]);
          int top = int(data[3]);

          int right = int(data[4]);
          int bottom = int(data[5]);

          int width = right - left;
          int height = bottom - top;

          DL_RESULT result;
          result.classId = int(data[0]);
          result.confidence = float(data[1]);
          result.box = cv::Rect(left, top, width, height);
          oResult.push_back(result);
      }
      data += strideNum;
    }
    
    return RET_OK;
}


template<typename N>
char* YOLO_V8::YOLO_origin_Process(cv::Mat& iImg, N& blob, std::vector<DL_RESULT>& oResult) {
    std::vector<int64_t> inputNodeDims = { 1,3,imgSize.at(0),imgSize.at(1) };
    Ort::Value inputTensor = Ort::Value::CreateTensor<typename std::remove_pointer<N>::type>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1),
        inputNodeDims.data(), inputNodeDims.size());

    auto outputTensor = m_session->Run(Ort::RunOptions{ nullptr }, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(),
        outputNodeNames.size());

    Ort::TypeInfo typeInfo = outputTensor.front().GetTypeInfo();
    auto tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputNodeDims = tensor_info.GetShape();
    auto output = outputTensor.front().GetTensorMutableData<typename std::remove_pointer<N>::type>();
    delete[] blob;
    switch (modelType)
    {
    case YOLO_DETECT_V8:
    case YOLO_DETECT_V8_HALF:
    {
        int strideNum = outputNodeDims[1];//84
        int signalResultNum = outputNodeDims[2];//8400
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        cv::Mat rawData;
        if (modelType == YOLO_DETECT_V8)
        {   
            // FP32
            rawData = cv::Mat(strideNum, signalResultNum, CV_32F, output);
        }
        else
        {   
            // FP16
            rawData = cv::Mat(strideNum, signalResultNum, CV_16F, output);
            rawData.convertTo(rawData, CV_32F);
        }
        //Note:
        //ultralytics add transpose operator to the output of yolov8 model.which make yolov8/v5/v7 has same shape
        //https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt
        //rowData = rowData.t();
        rawData = rawData.t();
        float* data = (float*)rawData.data;

        for (int i = 0; i < signalResultNum; ++i)
        {
            float* classesScores = data + 4;
            cv::Mat scores(1, this->classes.size(), CV_32FC1, classesScores);
            cv::Point class_id;
            double maxClassScore;
            cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
            if (maxClassScore > rectConfidenceThreshold)
            {
                confidences.push_back(maxClassScore);
                class_ids.push_back(class_id.x);
                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * resizeScales);
                int top = int((y - 0.5 * h) * resizeScales);

                int width = int(w * resizeScales);
                int height = int(h * resizeScales);

                boxes.push_back(cv::Rect(left, top, width, height));
            }
            data += strideNum;
        }
        std::vector<int> nmsResult;
        cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold, iouThreshold, nmsResult);
        for (int i = 0; i < nmsResult.size(); ++i)
        {
            int idx = nmsResult[i];
            DL_RESULT result;
            result.classId = class_ids[idx];
            result.confidence = confidences[idx];
            result.box = boxes[idx];
            oResult.push_back(result);
        }
        break;
    }
    default:
        std::cout << "[YOLO_V8]: " << "Not support model type." << std::endl;
    }
    return RET_OK;

}