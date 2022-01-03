package com.reactnativetensorflowlite;

import android.util.Log;

import androidx.annotation.NonNull;

import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.bridge.WritableNativeArray;
import com.facebook.react.bridge.WritableNativeMap;
import com.facebook.react.module.annotations.ReactModule;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.image.ImageProcessor;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;


//import org.tensorflow:tensorflow-lite;

@ReactModule(name = TensorflowLiteModule.NAME)
public class TensorflowLiteModule extends ReactContextBaseJavaModule {
    public static final String NAME = "TensorflowLite";

    public TensorflowLiteModule(ReactApplicationContext reactContext) {
        super(reactContext);
    }

    @Override
    @NonNull
    public String getName() {
        return NAME;
    }


    // Example method
    // See https://reactnative.dev/docs/native-modules-android
    @ReactMethod
    public void multiply(int a, int b, Promise promise) {
        promise.resolve(a * b);
    }

    @ReactMethod
    public void getModelInfo(ReadableMap args, Promise promise) {
        String modelPath = args.getString("model");
        if(modelPath == null) {
            promise.reject("NO_MODEL_PATH", "No model path specified");
            return;
        }
        if(modelPath.startsWith("file:///")) {
            modelPath = modelPath.substring(7);
        }
        File modelFile = new File(modelPath);
        try (Interpreter interpreter = new Interpreter(modelFile)) {
          interpreter.allocateTensors();
          int tensorCount = interpreter.getInputTensorCount();
          int outTensorCount = interpreter.getOutputTensorCount();
          if(tensorCount == 0 || outTensorCount == 0) {
            promise.reject("TF_TENSOR_ALLOC_ERROR", "TF Lite interpreter creation failed");
            return;
          }

          WritableArray inputShapes = new WritableNativeArray();
          WritableArray outputShapes = new WritableNativeArray();
          for(int i = 0; i < tensorCount; i++) {
            Tensor tensor = interpreter.getInputTensor(i);
            WritableMap shapeMap = tensorShapeMapForTensor(tensor);
            inputShapes.pushMap(shapeMap);
          }
          for(int i = 0; i < outTensorCount; i++) {
            Tensor tensor = interpreter.getOutputTensor(i);
            WritableMap shapeMap = tensorShapeMapForTensor(tensor);
            outputShapes.pushMap(shapeMap);
          }
          WritableMap ret = new WritableNativeMap();
          ret.putArray("input", inputShapes);
          ret.putArray("output", outputShapes);
          promise.resolve(ret);
        }
        promise.reject("TF_SOMETHING_ERROR", "IDK IT JUST CRASHED HELP");
    }

  @ReactMethod
  public void runModelWithFiles(ReadableMap args, Promise promise) {
    String modelPath = args.getString("model");
    ReadableArray filePathsArray = args.getArray("files");
    String fileMode = args.getString("fileMode");
    ReadableArray shapesArray = args.getArray("shapes");
    ReadableMap groupMode = args.getMap("groupMode");
    boolean imageModeIsFit = args.getString("imageScaleMode") == "fit";
    ReadableArray imageCropsArray = args.getArray("imageCrops");
    boolean cropsAreRelative = args.getString("imageCropsMode") == "relative";
    boolean grayscale = args.getBoolean("grayScale");

    if(modelPath == null) {
      promise.reject("NO_MODEL_PATH", "No model path specified");
      return;
    }
    if(modelPath.startsWith("file:///")) {
      modelPath = modelPath.substring(7);
    }
    if(filePathsArray.size() <= 0) {
      promise.reject("NO_FILE_PATHS", "No file paths specified");
      return;
    }
    try (Interpreter interpreter = getBestInterpreter(new File(modelPath))) {
      interpreter.allocateTensors();
      int tensorCount = interpreter.getInputTensorCount();
      int outTensorCount = interpreter.getOutputTensorCount();
      if(tensorCount == 0 || outTensorCount == 0) {
        promise.reject("TF_TENSOR_ALLOC_ERROR", "TF Lite interpreter creation failed");
        return;
      }

      WritableArray inputShapes = new WritableNativeArray();
      WritableArray outputShapes = new WritableNativeArray();
      for(int i = 0; i < tensorCount; i++) {
        Tensor tensor = interpreter.getInputTensor(i);
        WritableMap shapeMap = tensorShapeMapForTensor(tensor);
        inputShapes.pushMap(shapeMap);
      }
      for(int i = 0; i < outTensorCount; i++) {
        Tensor tensor = interpreter.getOutputTensor(i);
        WritableMap shapeMap = tensorShapeMapForTensor(tensor);
        outputShapes.pushMap(shapeMap);
      }
      WritableMap ret = new WritableNativeMap();
      ret.putArray("input", inputShapes);
      ret.putArray("output", outputShapes);
      promise.resolve(ret);
    }
    promise.reject("TF_SOMETHING_ERROR", "IDK IT JUST CRASHED HELP");
  }


  public static native int nativeMultiply(int a, int b);

  public WritableMap tensorShapeMapForTensor(Tensor tensor) {
    int[] shape = tensor.shape();
    WritableArray shapeNumArr = new WritableNativeArray();
    for(int shapeIdx = 0; shapeIdx < shape.length; shapeIdx++) {
      shapeNumArr.pushInt(shape[shapeIdx]);
    }
    WritableMap shapeMap = new WritableNativeMap();
    shapeMap.putString("name", tensor.name());
    shapeMap.putInt("dataType", tensor.dataType().ordinal());
    shapeMap.putArray("shape", shapeNumArr);
    return shapeMap;
  }

  public Interpreter getBestInterpreter(File model) {
    Interpreter.Options options = new Interpreter.Options();
    CompatibilityList compatList = new CompatibilityList();
    if(compatList.isDelegateSupportedOnThisDevice()){
      // if the device has a supported GPU, add the GPU delegate
      GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
      GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
      options.addDelegate(gpuDelegate);
    } else {
      // if the GPU is not supported, run on 4 threads
      options.setNumThreads(4);
    }
    Interpreter interpreter = new Interpreter(model, options);
    return interpreter;
  }

}
