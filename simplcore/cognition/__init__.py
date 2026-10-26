# -*- coding: utf-8 -*-
"""SimplCore · 认知模块.

类人认知推理模块，整合所有类人化能力:
1. 自然语言理解 (NLU)
2. 对话状态跟踪
3. 长程记忆
4. 情绪智能
5. 响应多样性
6. 认知推理 (Tree of Thought, Reflexion, Associative Reasoning)
7. 知识推理
8. 多模态认知整合

核心组件:
- HumanLikeCognitiveReasoner - 类人认知推理器
- MultimodalCognitiveIntegration - 多模态认知整合系统
"""

from SimplCore.cognition.human_like_reasoner import (
    HumanLikeMode,
    HumanLikeCognitiveReasoner,
)

from SimplCore.cognition.multimodal_cognition import (
    IntegrationMode,
    PerceptionCognitionLoop,
    CrossModalReasoner,
    MultimodalCognitiveIntegration,
)

from SimplCore.cognition.tree_of_thought import TreeOfThought

from SimplCore.cognition.reflexion import ReflexionModule

from SimplCore.cognition.associative_reasoner import AssociativeReasoner

from SimplCore.cognition.cognitive_reasoner import CognitiveReasoner

from SimplCore.cognition.emotional_engine import EmotionalDecisionEngine

from SimplCore.cognition.metacognition import MetacognitiveMonitor

from SimplCore.cognition.advanced_reasoning import AdvancedReasoningEngine

from SimplCore.cognition.causal_reasoner import CausalReasoner

from SimplCore.cognition.analogical_reasoner import AnalogicalReasoner

from SimplCore.cognition.nonlinear_thinking import NonlinearThinking

from SimplCore.cognition.self_improvement import (
    ImprovementType,
    ImprovementPriority,
    ImprovementRecord,
    PerformanceMetric,
    BlindSpotDetector,
    SelfImprovementModule,
    SelfLearningLoop,
    IntegratedSelfImprovement,
)

from SimplCore.cognition.vision_cognition import (
    ImageFeature,
    DetectionResult,
    VQAResult,
    CaptionResult,
    ImageFeatureExtractor,
    CLIPFeatureExtractor,
    ObjectDetector,
    YOLODetector,
    VisualQuestionAnswering,
    BLIPVQA,
    ImageCaptioner,
    BLIPCaptioner,
    VisualReasoningEngine,
    create_feature_extractor,
    create_detector,
    create_vqa,
    create_captioner,
    create_visual_engine,
)

__all__ = [
    "HumanLikeMode",
    "HumanLikeCognitiveReasoner",
    "IntegrationMode",
    "PerceptionCognitionLoop",
    "CrossModalReasoner",
    "MultimodalCognitiveIntegration",
    "TreeOfThought",
    "ReflexionModule",
    "AssociativeReasoner",
    "CognitiveReasoner",
    "EmotionalDecisionEngine",
    "MetacognitiveMonitor",
    "AdvancedReasoningEngine",
    "CausalReasoner",
    "AnalogicalReasoner",
    "NonlinearThinking",
    "ImprovementType",
    "ImprovementPriority",
    "ImprovementRecord",
    "PerformanceMetric",
    "BlindSpotDetector",
    "SelfImprovementModule",
    "SelfLearningLoop",
    "IntegratedSelfImprovement",
    "ImageFeature",
    "DetectionResult",
    "VQAResult",
    "CaptionResult",
    "ImageFeatureExtractor",
    "CLIPFeatureExtractor",
    "ObjectDetector",
    "YOLODetector",
    "VisualQuestionAnswering",
    "BLIPVQA",
    "ImageCaptioner",
    "BLIPCaptioner",
    "VisualReasoningEngine",
    "create_feature_extractor",
    "create_detector",
    "create_vqa",
    "create_captioner",
    "create_visual_engine",
]
