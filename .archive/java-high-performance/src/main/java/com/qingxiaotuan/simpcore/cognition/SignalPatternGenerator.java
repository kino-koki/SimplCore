package com.qingxiaotuan.simpcore.cognition;

import java.util.*;
import java.util.concurrent.*;

public class SignalPatternGenerator {

    public enum FrequencyBand {
        DELTA(0.5f, 4.0f, "低频"),
        THETA(4.0f, 8.0f, "次低频"),
        ALPHA(8.0f, 12.0f, "中频"),
        BETA(12.0f, 30.0f, "高频"),
        GAMMA(30.0f, 100.0f, "超高频");

        private final float minHz;
        private final float maxHz;
        private final String description;

        FrequencyBand(float minHz, float maxHz, String description) {
            this.minHz = minHz;
            this.maxHz = maxHz;
            this.description = description;
        }

        public float getMinHz() { return minHz; }
        public float getMaxHz() { return maxHz; }
        public float getCenterHz() { return (minHz + maxHz) / 2; }
        public String getDescription() { return description; }

        public static FrequencyBand fromHz(float hz) {
            if (hz < 4.0f) return DELTA;
            if (hz < 8.0f) return THETA;
            if (hz < 12.0f) return ALPHA;
            if (hz < 30.0f) return BETA;
            return GAMMA;
        }
    }

    public enum SignalMode {
        LOW_FREQUENCY(new FrequencyBand[]{FrequencyBand.DELTA}, "低频模式"),
        SUB_LOW(new FrequencyBand[]{FrequencyBand.THETA, FrequencyBand.ALPHA}, "次低中频模式"),
        MEDIUM(new FrequencyBand[]{FrequencyBand.ALPHA}, "中频模式"),
        HIGH(new FrequencyBand[]{FrequencyBand.BETA}, "高频模式"),
        WIDE_BAND(new FrequencyBand[]{FrequencyBand.THETA, FrequencyBand.ALPHA, FrequencyBand.GAMMA}, "宽频模式"),
        ULTRA_HIGH(new FrequencyBand[]{FrequencyBand.BETA, FrequencyBand.GAMMA}, "超高频模式");

        private final FrequencyBand[] frequencies;
        private final String description;

        SignalMode(FrequencyBand[] frequencies, String description) {
            this.frequencies = frequencies;
            this.description = description;
        }

        public FrequencyBand[] getFrequencies() { return frequencies; }
        public String getDescription() { return description; }
    }

    private final Random random;
    private final ConcurrentHashMap<String, double[]> patterns;

    public SignalPatternGenerator() {
        this.random = new Random(42);
        this.patterns = new ConcurrentHashMap<>();
    }

    public SignalPatternGenerator(long seed) {
        this.random = new Random(seed);
        this.patterns = new ConcurrentHashMap<>();
    }

    public double[] generateWave(FrequencyBand frequency, int sampleCount, float sampleRate) {
        double[] wave = new double[sampleCount];
        float centerHz = frequency.getCenterHz();
        float amplitude = random.nextFloat() * 0.5f + 0.5f;

        for (int i = 0; i < sampleCount; i++) {
            double t = i / sampleRate;
            double phase = random.nextDouble() * Math.PI * 2;
            wave[i] = amplitude * Math.sin(2 * Math.PI * centerHz * t + phase);
        }

        return wave;
    }

    public double[] generateCompositeWave(SignalMode mode, int sampleCount, float sampleRate) {
        double[] composite = new double[sampleCount];
        FrequencyBand[] frequencies = mode.getFrequencies();
        
        float[] weights = new float[frequencies.length];
        float totalWeight = 0;
        for (int i = 0; i < frequencies.length; i++) {
            weights[i] = random.nextFloat() * 0.3f + 0.1f;
            totalWeight += weights[i];
        }
        for (int i = 0; i < frequencies.length; i++) {
            weights[i] /= totalWeight;
        }

        for (int i = 0; i < frequencies.length; i++) {
            double[] wave = generateWave(frequencies[i], sampleCount, sampleRate);
            for (int j = 0; j < sampleCount; j++) {
                composite[j] += wave[j] * weights[i];
            }
        }

        double maxAmplitude = Arrays.stream(composite).map(Math::abs).max().orElse(1.0);
        for (int i = 0; i < sampleCount; i++) {
            composite[i] /= maxAmplitude;
        }

        composite = addNoise(composite, 0.05f);
        composite = applyEnvelope(composite, sampleRate);

        return composite;
    }

    private double[] addNoise(double[] signal, float noiseLevel) {
        double[] result = new double[signal.length];
        for (int i = 0; i < signal.length; i++) {
            result[i] = signal[i] + (random.nextDouble() - 0.5) * 2 * noiseLevel;
        }
        return result;
    }

    private double[] applyEnvelope(double[] signal, float sampleRate) {
        double[] result = new double[signal.length];
        int attackSamples = (int) (0.1f * sampleRate);
        int decaySamples = (int) (0.2f * sampleRate);
        int sustainSamples = signal.length - attackSamples - decaySamples;
        float sustainLevel = 0.8f;

        for (int i = 0; i < attackSamples; i++) {
            result[i] = signal[i] * (i / (float) attackSamples);
        }

        for (int i = 0; i < sustainSamples; i++) {
            int idx = attackSamples + i;
            float envelope = 1.0f - (1.0f - sustainLevel) * (i / (float) sustainSamples);
            result[idx] = signal[idx] * envelope;
        }

        for (int i = 0; i < decaySamples; i++) {
            int idx = attackSamples + sustainSamples + i;
            result[idx] = signal[idx] * sustainLevel * (1.0f - i / (float) decaySamples);
        }

        return result;
    }

    public double[][] generateMultiChannelPattern(SignalMode mode, int channels, 
                                                  int sampleCount, float sampleRate) {
        double[][] pattern = new double[channels][sampleCount];
        
        for (int c = 0; c < channels; c++) {
            pattern[c] = generateCompositeWave(mode, sampleCount, sampleRate);
        }

        pattern = addChannelCorrelation(pattern, channels);

        return pattern;
    }

    private double[][] addChannelCorrelation(double[][] pattern, int channels) {
        double[][] result = new double[channels][pattern[0].length];
        
        for (int i = 0; i < channels; i++) {
            System.arraycopy(pattern[i], 0, result[i], 0, pattern[i].length);
        }

        for (int i = 0; i < channels; i++) {
            for (int j = i + 1; j < channels; j++) {
                double correlation = random.nextDouble() * 0.5 + 0.3;
                for (int k = 0; k < result[i].length; k++) {
                    result[i][k] = result[i][k] * (1 - correlation) + result[j][k] * correlation * 0.5;
                    result[j][k] = result[j][k] * (1 - correlation) + result[i][k] * correlation * 0.5;
                }
            }
        }

        return result;
    }

    public double[] generateEventRelatedPotential(int sampleCount, float sampleRate, 
                                                  ERPEvent eventType) {
        double[] erp = new double[sampleCount];
        int eventTime = (int) (0.2f * sampleCount);

        switch (eventType) {
            case P300:
                generateP300(erp, eventTime, sampleRate);
                break;
            case N400:
                generateN400(erp, eventTime, sampleRate);
                break;
            case MMN:
                generateMMN(erp, eventTime, sampleRate);
                break;
            case N200:
                generateN200(erp, eventTime, sampleRate);
                break;
        }

        return erp;
    }

    private void generateP300(double[] signal, int eventTime, float sampleRate) {
        int p300Peak = eventTime + (int) (0.3f * sampleRate);
        float amplitude = 0.8f;
        
        for (int i = 0; i < signal.length; i++) {
            double t = (i - p300Peak) / sampleRate;
            signal[i] += amplitude * Math.exp(-t * t / (2 * 0.05 * 0.05)) * Math.cos(2 * Math.PI * 8 * t);
        }
    }

    private void generateN400(double[] signal, int eventTime, float sampleRate) {
        int n400Peak = eventTime + (int) (0.4f * sampleRate);
        float amplitude = -0.6f;
        
        for (int i = 0; i < signal.length; i++) {
            double t = (i - n400Peak) / sampleRate;
            signal[i] += amplitude * Math.exp(-t * t / (2 * 0.08 * 0.08));
        }
    }

    private void generateMMN(double[] signal, int eventTime, float sampleRate) {
        int mmnPeak = eventTime + (int) (0.15f * sampleRate);
        float amplitude = -0.4f;
        
        for (int i = 0; i < signal.length; i++) {
            double t = (i - mmnPeak) / sampleRate;
            signal[i] += amplitude * Math.exp(-t * t / (2 * 0.03 * 0.03));
        }
    }

    private void generateN200(double[] signal, int eventTime, float sampleRate) {
        int n200Peak = eventTime + (int) (0.2f * sampleRate);
        float amplitude = -0.5f;
        
        for (int i = 0; i < signal.length; i++) {
            double t = (i - n200Peak) / sampleRate;
            signal[i] += amplitude * Math.exp(-t * t / (2 * 0.04 * 0.04));
        }
    }

    public void savePattern(String name, double[] pattern) {
        patterns.put(name, pattern);
    }

    public double[] loadPattern(String name) {
        return patterns.get(name);
    }

    public void saveMultiChannelPattern(String name, double[][] pattern) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < pattern.length; i++) {
            if (i > 0) sb.append("|");
            for (int j = 0; j < pattern[i].length; j++) {
                if (j > 0) sb.append(",");
                sb.append(pattern[i][j]);
            }
        }
        patterns.put(name + "_channels", pattern[0]);
    }

    public double[][] loadMultiChannelPattern(String name, int channels) {
        double[] firstChannel = patterns.get(name + "_channels");
        if (firstChannel == null) return null;
        
        double[][] result = new double[channels][firstChannel.length];
        for (int i = 0; i < channels; i++) {
            System.arraycopy(firstChannel, 0, result[i], 0, firstChannel.length);
        }
        
        return result;
    }

    public SignalMode classifyPattern(double[] pattern, float sampleRate) {
        float dominantHz = computeDominantFrequency(pattern, sampleRate);
        FrequencyBand freq = FrequencyBand.fromHz(dominantHz);

        if (freq == FrequencyBand.DELTA) return SignalMode.LOW_FREQUENCY;
        if (freq == FrequencyBand.THETA) return SignalMode.SUB_LOW;
        if (freq == FrequencyBand.ALPHA) return SignalMode.MEDIUM;
        if (freq == FrequencyBand.BETA) return SignalMode.HIGH;
        return SignalMode.ULTRA_HIGH;
    }

    private float computeDominantFrequency(double[] signal, float sampleRate) {
        int n = signal.length;
        double[] fft = new double[n];
        
        for (int i = 0; i < n; i++) {
            double sumReal = 0;
            double sumImag = 0;
            for (int j = 0; j < n; j++) {
                double angle = 2 * Math.PI * i * j / n;
                sumReal += signal[j] * Math.cos(angle);
                sumImag += signal[j] * Math.sin(angle);
            }
            fft[i] = Math.sqrt(sumReal * sumReal + sumImag * sumImag);
        }

        int maxIdx = 0;
        double maxVal = 0;
        for (int i = 1; i < n / 2; i++) {
            if (fft[i] > maxVal) {
                maxVal = fft[i];
                maxIdx = i;
            }
        }

        return (float) (maxIdx * sampleRate / n);
    }

    public String analyzePattern(double[] pattern, float sampleRate) {
        StringBuilder analysis = new StringBuilder();
        analysis.append("=== Signal Pattern Analysis ===\n");
        
        float dominantHz = computeDominantFrequency(pattern, sampleRate);
        FrequencyBand freq = FrequencyBand.fromHz(dominantHz);
        SignalMode mode = classifyPattern(pattern, sampleRate);
        
        analysis.append("Dominant Frequency: " + String.format("%.2f Hz", dominantHz) + "\n");
        analysis.append("Frequency Band: " + freq.name() + " (" + freq.getDescription() + ")\n");
        analysis.append("Signal Mode: " + mode.name() + " (" + mode.getDescription() + ")\n");
        
        double mean = Arrays.stream(pattern).average().orElse(0);
        double std = Math.sqrt(Arrays.stream(pattern).map(x -> Math.pow(x - mean, 2)).average().orElse(0));
        double max = Arrays.stream(pattern).max().orElse(0);
        double min = Arrays.stream(pattern).min().orElse(0);
        
        analysis.append("\nSignal Statistics:\n");
        analysis.append("  Mean: " + String.format("%.4f", mean) + "\n");
        analysis.append("  Std: " + String.format("%.4f", std) + "\n");
        analysis.append("  Max: " + String.format("%.4f", max) + "\n");
        analysis.append("  Min: " + String.format("%.4f", min) + "\n");
        analysis.append("  Peak-to-Peak: " + String.format("%.4f", max - min) + "\n");
        
        analysis.append("\n===================================");
        
        return analysis.toString();
    }

    public enum ERPEvent {
        P300,
        N400,
        MMN,
        N200
    }
}