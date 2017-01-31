package com.ss.tf.faceevaluator;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class TensorFlowFaceEvaluator {

    private TensorFlowInferenceInterface mInferenceInterface;

    public int initializeTensorFlow(AssetManager assetManager, String modelFilename) {
        //TODO: 推論の準備を可能な限り済ませておく。
        mInferenceInterface = new TensorFlowInferenceInterface();
        return mInferenceInterface.initializeTensorFlow(assetManager, modelFilename);
    }

    public Float[] runInference(Bitmap face) {
        // 引数を変換し、TFで結果をえる。
        return null;
    }

    public void close() {
        mInferenceInterface.close();
    }
}
