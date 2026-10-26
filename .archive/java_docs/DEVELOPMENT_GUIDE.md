# SimplCore 深入开发指南

**版本**: v1.2  
**日期**: 2025-01-16  
**作者**: Qingxiaotuan AI Team  
**状态**: 持续更新

---

## 目录

1. [视觉系统开发](#1-视觉系统开发)
2. [模型训练模块](#2-模型训练模块)
3. [性能优化](#3-性能优化)
4. [安全模块](#4-安全模块)
5. [API使用示例](#5-api使用示例)

---

## 1. 视觉系统开发

### 1.1 摄像头驱动

#### 真实摄像头驱动

`RealCameraSource` 实现了基于 JavaCV 的真实摄像头捕获功能：

```java
// 创建真实摄像头源
RealCameraSource camera = new RealCameraSource(0);  // 0 表示第一个摄像头

// 配置摄像头属性
CameraSource.CameraProperties props = new CameraSource.CameraProperties();
props.setWidth(1920);
props.setHeight(1080);
props.setFps(30);
camera.setProperties(props);

// 初始化并启动
camera.initialize();
camera.start();

// 设置帧回调
camera.setCallback(new CameraSource.FrameCallback() {
    @Override
    public void onFrame(CameraSource.Frame frame) {
        byte[] data = frame.getData();
        int width = frame.getWidth();
        int height = frame.getHeight();
    }
    
    @Override
    public void onError(CameraSource.CameraException e) {
        // 处理错误
    }
});

// 手动捕获帧
CameraSource.Frame frame = camera.capture();

// 停止和释放
camera.stop();
camera.release();
```

#### 摄像头发现

```java
// 发现可用摄像头
List<CameraSource.CameraProperties> available = camera.getAvailableProperties();
for (CameraSource.CameraProperties prop : available) {
    System.out.println("Camera: " + prop.getWidth() + "x" + prop.getHeight() + "@" + prop.getFps());
}
```

### 1.2 图像预处理

新增的图像处理器：

| 处理器 | 功能 | 参数 |
|--------|------|------|
| `HistogramEqualizationProcessor` | 直方图均衡化 | 无 |
| `BlurProcessor` | 高斯模糊 | kernelSize (默认5) |
| `SharpenProcessor` | 图像锐化 | amount (默认1.0) |
| `GrayscaleProcessor` | 灰度转换 | 无 |

#### 使用示例

```java
PreprocessingPipeline pipeline = new PreprocessingPipeline();

// 添加预处理步骤
pipeline.addProcessor(new ImageProcessors.DenoisingProcessor());
pipeline.addProcessor(new ImageProcessors.HistogramEqualizationProcessor());
pipeline.addProcessor(new ImageProcessors.SharpenProcessor(1.5f));
pipeline.addProcessor(new ImageProcessors.ColorNormalizationProcessor());

// 处理帧
CameraSource.Frame rawFrame = camera.capture();
PreprocessingPipeline.ProcessedFrame processedFrame = pipeline.process(rawFrame);
```

### 1.3 特征提取

#### 特征提取器列表

| 提取器 | 功能 | 特征维度 |
|--------|------|---------|
| `SimpleKeypointExtractor` | 关键点检测 | 128/关键点 |
| `ColorHistogramExtractor` | 颜色直方图 | 512 (8x8x8) |
| `HogExtractor` | HOG特征 | 3780 |
| `CnnFeatureExtractor` | CNN特征 | 512 |

---

## 2. 模型训练模块

### 2.1 优化器

#### AdamW 优化器

```java
// 创建AdamW优化器
AdamWOptimizer optimizer = new AdamWOptimizer(
    0.001f,    // 学习率
    0.01f,     // 权重衰减
    0.9f,      // beta1
    0.999f,    // beta2
    1e-8f,     // epsilon
    1.0f       // 梯度裁剪
);

// 设置学习率调度器
optimizer.setScheduler(new LearningRateScheduler.CosineAnnealingScheduler(0.001f, 10000));

// 设置梯度累积
optimizer.setGradientAccumulationSteps(4);

// 更新权重
optimizer.update(weights, gradients);
```

#### 优化器对比

| 优化器 | 特点 | 适用场景 |
|--------|------|---------|
| `Optimizer` | 带动量的SGD | 传统训练 |
| `AdamWOptimizer` | Adam + 权重衰减解耦 | 大规模模型训练 |

### 2.2 学习率调度器

#### 新增调度器

| 调度器 | 功能 | 参数 |
|--------|------|------|
| `PolynomialDecayScheduler` | 多项式衰减 | power, endLearningRate |
| `ReduceLROnPlateauScheduler` | 根据损失自动调整 | factor, patience, threshold |
| `CyclicLR` | 周期性学习率 | maxLearningRate, stepSize |
| `NoamScheduler` | Transformer专用 | modelSize, warmupSteps |

#### 使用示例

```java
// Cosine Annealing with Warm Restarts
LearningRateScheduler scheduler = new LearningRateScheduler.CosineAnnealingWarmRestarts(
    0.001f,  // base LR
    1000,    // T_0
    2,       // T_mult
    0.0f     // min LR
);

// ReduceLROnPlateau (基于损失)
LearningRateScheduler.ReduceLROnPlateauScheduler plateauScheduler = 
    new LearningRateScheduler.ReduceLROnPlateauScheduler(0.001f, 0.1f, 10, 1e-7f, 0.0f);
plateauScheduler.step(currentLoss);

// Noam Scheduler (Transformer)
LearningRateScheduler.NoamScheduler noamScheduler = 
    new LearningRateScheduler.NoamScheduler(dModel, warmupSteps);
```

### 2.3 训练策略

#### 梯度累积

```java
optimizer.setGradientAccumulationSteps(4);

for (int i = 0; i < iterations; i++) {
    float[] gradients = model.backward(input, target);
    optimizer.accumulateGradients(gradients);
    
    if (optimizer.isAccumulationReady()) {
        optimizer.updateWithAccumulated(model.getWeights());
    }
}
```

#### 混合精度训练

```java
TrainingLoop trainer = new TrainingLoop(model, tokenizer, optimizer);
trainer.enableMixedPrecision(true);
```

---

## 3. 性能优化

### 3.1 SIMD向量优化

#### 优化的操作

| 操作 | SIMD支持 | 性能提升 |
|------|---------|---------|
| `matmul` | ✅ | 2-4x |
| `add/sub` | ✅ | 4-8x |
| `mul` (标量) | ✅ | 4-8x |
| `dotProduct` | ✅ | 4-8x |
| `relu` | ✅ | 4-8x |
| `gelu` | ✅ | 3-6x |
| `silu` | ✅ | 3-6x |
| `layerNorm` | ✅ | 2-4x |
| `softmax` | ✅ | 2-4x |
| `hadamardProduct` | ✅ | 4-8x |
| `sum` | ✅ | 4-8x |
| `clamp` | ✅ | 4-8x |

#### SIMD启用控制

```java
// 在MatrixOperations中，VECTOR_ENABLED默认为true
// 可通过设置为false禁用SIMD
// private static final boolean VECTOR_ENABLED = true;
```

### 3.2 并行计算

#### ForkJoinPool 并行矩阵乘法

```java
float[] C = MatrixOperations.matmul(A, B, M, K, N);
```

### 3.3 内存优化

#### 对象池

```java
MemoryManager memoryManager = new MemoryManager();
float[] buffer = memoryManager.allocateFloatArray(size);
memoryManager.release(buffer);
```

---

## 4. 安全模块

### 4.1 模型加密

#### AES-GCM 加密

```java
// 创建加密器
String key = ModelEncryption.generateKeyString();
ModelEncryption encryption = new ModelEncryption(key);

// 加密模型权重
float[] weights = model.getWeights();
byte[] encrypted = encryption.encrypt(weights);

// 解密
float[] decrypted = encryption.decrypt(encrypted);

// 带HMAC的加密（更安全）
byte[] encryptedWithHmac = encryption.encryptWithHMAC(weights);
float[] decryptedWithHmac = encryption.decryptWithHMAC(encryptedWithHmac);

// 加密字符串
byte[] encryptedStr = encryption.encryptString("sensitive data");
String decryptedStr = encryption.decryptString(encryptedStr);
```

### 4.2 RBAC访问控制

#### 创建RBAC系统

```java
// 创建默认系统
RBACSystem rbac = RBACSystem.createDefaultSystem();

// 创建用户
rbac.createUser("user1", "researcher");
rbac.createUser("admin", "admin");

// 创建会话
String sessionId = rbac.createSession("user1");

// 验证权限
boolean canExecute = rbac.hasPermissionWithSession(sessionId, "inference", "execute");
boolean canModify = rbac.hasPermissionWithSession(sessionId, "model:model1", "modify");

// 验证会话
boolean valid = rbac.validateSession(sessionId);

// 失效会话
rbac.invalidateSession(sessionId);
```

#### 默认角色权限

| 角色 | 权限 |
|------|------|
| `admin` | 所有权限 |
| `model_engineer` | 模型访问/修改/创建，推理执行，训练启动/停止/监控 |
| `researcher` | 模型访问，推理执行，推理监控，训练监控 |
| `user` | 推理执行 |
| `monitor` | 推理监控，训练监控，配置访问 |

#### 审计日志

```java
// 获取审计日志
List<RBACSystem.AuditLog> logs = rbac.getAuditLogs(startTime, endTime);

// 获取用户日志
List<RBACSystem.AuditLog> userLogs = rbac.getAuditLogsForUser("user1");

// 获取活跃会话数
int activeSessions = rbac.getActiveSessionCount();

// 获取活跃用户
Set<String> activeUsers = rbac.getActiveUsers();
```

---

## 5. API使用示例

### 5.1 完整训练流程

```java
// 配置
int vocabSize = 32000;
int dModel = 768;
int nLayers = 12;
int nHeads = 12;
int maxSeqLen = 512;

// 初始化
Tokenizer tokenizer = new Tokenizer(vocabSize);
SimplCoreModel model = new SimplCoreModel(dModel, nLayers, nHeads, vocabSize, maxSeqLen);

AdamWOptimizer optimizer = new AdamWOptimizer(0.0001f, 0.01f);
optimizer.setScheduler(new LearningRateScheduler.CosineAnnealingScheduler(0.0001f, 100000));

TrainingLoop trainer = new TrainingLoop(model, tokenizer, optimizer);
trainer.enableMixedPrecision(true);

CheckpointManager checkpointManager = new CheckpointManager("checkpoints/", 5);

// 训练循环
for (int epoch = 0; epoch < 10; epoch++) {
    for (String text : trainingData) {
        float loss = trainer.trainStep(text, batchSize);
        
        if (step % 100 == 0) {
            System.out.printf("Step %d: Loss = %.4f, LR = %.6f%n", 
                step, loss, optimizer.getCurrentLearningRate());
            checkpointManager.save(model, optimizer, step, loss);
        }
    }
    
    // 评估
    ModelEvaluator evaluator = new ModelEvaluator(model, tokenizer);
    ModelEvaluator.EvaluationMetrics metrics = evaluator.evaluate(validationData);
    System.out.printf("Epoch %d: Perplexity = %.2f, Accuracy = %.2f%n", 
        epoch, metrics.perplexity(), metrics.accuracy());
}
```

### 5.2 视觉系统集成

```java
// 创建视觉系统
RealTimeVisionSystem visionSystem = new RealTimeVisionSystem();
visionSystem.initialize();

// 使用真实摄像头
RealCameraSource camera = new RealCameraSource(0);
camera.initialize();
camera.start();

// 配置处理管线
PreprocessingPipeline preprocessing = new PreprocessingPipeline();
preprocessing.addProcessor(new ImageProcessors.DenoisingProcessor());
preprocessing.addProcessor(new ImageProcessors.HistogramEqualizationProcessor());

FeatureExtractionLayer features = new FeatureExtractionLayer();
features.addExtractor(new FeatureExtractors.ColorHistogramExtractor());
features.addExtractor(new FeatureExtractors.HogExtractor());

InferenceDecisionLayer inference = new InferenceDecisionLayer();
inference.addEngine(new InferenceEngines.ObjectDetectionEngine());

VisionPipeline pipeline = new VisionPipeline(preprocessing, features, inference);

// 处理帧
CameraSource.Frame frame = camera.capture();
VisualAPILayer api = new VisualAPILayer();
api.setCameraSource(camera);
api.setPipeline(pipeline);
VisualAPILayer.VisionResult result = api.processFrame(frame);

// 获取检测结果
List<VisualAPILayer.DetectionResult> detections = api.getDetections(result);
for (VisualAPILayer.DetectionResult det : detections) {
    System.out.println(det.getClassName() + " at (" + det.getX() + ", " + det.getY() + ")");
}
```

---

## 6. 性能基准测试

### 6.1 运行基准测试

```bash
mvn test -Dtest=PerformanceBenchmark
```

### 6.2 矩阵运算性能

| 操作 | 尺寸 | 时间(ms) |
|------|------|---------|
| matmul | 1024x1024 | ~50 |
| relu | 1M elements | ~1 |
| layerNorm | 1024 elements | ~0.5 |
| softmax | 1024 elements | ~0.3 |

---

## 7. 架构定位（v1.2 更新）

### 7.1 架构演进

**v1.0**: 创新组件架构（无标准注意力）

**v1.2**: 标准Transformer + 创新组件混合架构

### 7.2 核心架构

```
SimplCore Block (v1.2):
  ├── Multi-Head Attention (标准QKV)
  │     ├── Q = X @ W_q
  │     ├── K = X @ W_k  
  │     ├── V = X @ W_v
  │     └── Attention(Q,K,V) = softmax(QK^T/√d) @ V
  ├── Add & LayerNorm
  │
  ├── ResonanceFieldProcessor (频率编码增强)
  ├── DynamicMemoryChannel (外部记忆)
  ├── TopologyAwareRouter (动态路由)
  ├── CausalInterventionNetwork (因果推理)
  ├── InformationBottleneck (信息压缩)
  ├── Feed-Forward Network
  ├── MetaRepresentationDistiller (特征蒸馏)
  ├── SpontaneousPlasticityEngine (在线适应)
  └── Add & LayerNorm
```

### 7.3 架构特点

| 特性 | 状态 | 说明 |
|------|------|------|
| 标准QKV注意力 | ✅ | 完全兼容Transformer |
| 多头并行处理 | ✅ | nHeads独立计算 |
| 残差连接 + LayerNorm | ✅ | 标准Transformer结构 |
| 因果掩码 | ✅ | 自回归生成支持 |
| K-V Cache | ✅ | 推理优化 |
| 创新增强组件 | ⭐ | 7个独特组件 |

### 7.4 兼容性

- ✅ **与现有Transformer生态兼容**
- ✅ **支持HuggingFace集成**
- ✅ **可独立使用标准注意力**
- ✅ **可组合创新组件**

---

**文档结束**
