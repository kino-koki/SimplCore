package com.qingxiaotuan.simpcore.training;

import java.util.HashMap;
import java.util.Map;

public class GradientResult {
    private final float[] dInput;
    private final Map<String, float[]> parameterGradients;

    public GradientResult(float[] dInput) {
        this.dInput = dInput;
        this.parameterGradients = new HashMap<>();
    }

    public float[] getdInput() {
        return dInput;
    }

    public Map<String, float[]> getParameterGradients() {
        return parameterGradients;
    }

    public void addGradient(String paramName, float[] gradient) {
        parameterGradients.put(paramName, gradient);
    }

    public float[] getGradient(String paramName) {
        return parameterGradients.get(paramName);
    }
}