# -*- coding: utf-8 -*-
"""SimplCore · NLU模块.

自然语言理解模块，增强语义解析精度、上下文关联能力及歧义消解机制.

核心组件:
1. SemanticParser - 语义解析器
2. ContextualDisambiguator - 上下文感知消歧器
3. ContextAwareSemanticParser - 语境感知语义解析器
4. MultiTurnComprehension - 多轮对话理解器
"""

from SimplCore.nlu.semantic_parser import (
    SemanticRole,
    DependencyRelation,
    IntentType,
    SemanticFrame,
    SemanticRoleLabeler,
    DependencyParser,
    IntentClassifier,
    ContextualSemanticEncoder,
    SemanticSimilarityCalculator,
    SemanticParser,
)

from SimplCore.nlu.disambiguator import (
    AmbiguityType,
    ResolutionConfidence,
    ReferentCandidate,
    AnaphoraResolver,
    WordSenseDisambiguator,
    EllipsisResolver,
    ContextualDisambiguator,
    DisambiguationPipeline,
)

from SimplCore.nlu.context_aware_parser import (
    CoherenceLevel,
    TopicState,
    ContextMemory,
    CoherenceDetector,
    TopicTracker,
    ContextAdaptiveWeighting,
    ContextAwareSemanticParser,
    MultiTurnComprehension,
)

__all__ = [
    "SemanticRole",
    "DependencyRelation",
    "IntentType",
    "SemanticFrame",
    "SemanticRoleLabeler",
    "DependencyParser",
    "IntentClassifier",
    "ContextualSemanticEncoder",
    "SemanticSimilarityCalculator",
    "SemanticParser",
    "AmbiguityType",
    "ResolutionConfidence",
    "ReferentCandidate",
    "AnaphoraResolver",
    "WordSenseDisambiguator",
    "EllipsisResolver",
    "ContextualDisambiguator",
    "DisambiguationPipeline",
    "CoherenceLevel",
    "TopicState",
    "ContextMemory",
    "CoherenceDetector",
    "TopicTracker",
    "ContextAdaptiveWeighting",
    "ContextAwareSemanticParser",
    "MultiTurnComprehension",
]
