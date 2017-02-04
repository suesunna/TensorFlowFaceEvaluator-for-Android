package com.ss.tf.faceevaluator;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import static android.content.ContentValues.TAG;

public class TensorFlowFaceEvaluator {

    // Config values.
    private String mInputName = "input";
    private String mOutputName = "output";
    private int mNumClasses;
    private int mInputSize;
    private int mImageMean = 117;
    private float mImageStd = 1;

    // Pre-allocated buffers.
    private int[] mIntValues;
    private float[] mFloatValues;
    private float[] mOutputs;
    private String[] mOutputNames;

    private TensorFlowInferenceInterface mInferenceInterface;

    public int initializeTensorFlow(AssetManager assetManager, String modelFilename,
                                    int numClasses, int inputSize) {
        mNumClasses = numClasses;
        mInputSize = inputSize;
        mIntValues = new int[mInputSize * mInputSize];
        mFloatValues = new float[mInputSize * mInputSize * 3];
        mOutputs = new float[mNumClasses];
        mOutputNames = new String[] {mOutputName};

        mInferenceInterface = new TensorFlowInferenceInterface();
        mInferenceInterface.enableStatLogging(true);
        return mInferenceInterface.initializeTensorFlow(assetManager, modelFilename);
    }

    public float[] runInference(Bitmap face) {
        Bitmap faceBitmap = Bitmap.createScaledBitmap(face, 28, 28, false);
        faceBitmap.getPixels(mIntValues, 0, faceBitmap.getWidth(),
                0, 0, faceBitmap.getWidth(), faceBitmap.getHeight());
        for (int i = 0; i < mIntValues.length; i++) {
            final int val = mIntValues[1];
            mFloatValues[i * 3] = (((val >> 16) & 0xFF) - mImageMean) / mImageStd;
            mFloatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - mImageMean) / mImageStd;
            mFloatValues[i * 3 + 2] = ((val & 0xFF) - mImageMean) / mImageStd;
        }

        Log.i(TAG, "InferenceInterface.fillNodeFloat();");
        mInferenceInterface.fillNodeFloat(mInputName,
                new int[] {1, mInputSize * mInputSize * 3}, mFloatValues);

        Log.i(TAG, "InferenceInterface.runInference();");
        mInferenceInterface.runInference(mOutputNames);

        Log.i(TAG, "InferenceInterface.readNodeFloat();");
        mInferenceInterface.readNodeFloat(mOutputName, mOutputs);

        return mOutputs;

    }

    public void close() {
        mInferenceInterface.close();
    }
}
