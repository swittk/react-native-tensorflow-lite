package com.reactnativetensorflowlite;

import androidx.annotation.NonNull;

import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.module.annotations.ReactModule;

import org.tensorflow:tensorflow-lite;

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
    public void runModelWithFiles(ReadableMap args, Promise promise) {
        String model = args.getString("model");
        String files = args.getString("files");
        if(model == null) {
            promise.reject("NO_MODEL_PATH", "No model path specified");
            return;
        }
        if(files == null) {
            promise.reject("NO_FILE_PATHS", "No file paths specified");
            return;
        }

        // TODO: Maybe do this using TFLite support library on Android..
        promise.resolve(0);
    }

    public static native int nativeMultiply(int a, int b);
}
