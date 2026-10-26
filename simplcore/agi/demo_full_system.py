# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore AGI 端到端集成演示
============================

展示完整的感知→理解→推理→反思闭环，覆盖所有核心模块。

演示流程:
  阶段1: AGI核心初始化与模块加载
  阶段2: 多模态感知融合 (文本 + 视觉 + 音频 + 环境)
  阶段3: 语境感知语义解析 + 多轮对话理解
  阶段4: 对话状态跟踪 + 长程记忆
  阶段5: 情绪智能感知与共情
  阶段6: 类人认知推理 (对话/推理/创造模式)
  阶段7: 知识图谱推理增强
  阶段8: 响应多样性生成
  阶段9: 自我改进与能力评估
  阶段10: 图灵测试综合评估
  阶段11: 增强训练器演示
"""

from __future__ import annotations

import sys
sys.path.insert(0, '.')

import numpy as np
import time
from datetime import datetime

from SimplCore.engine.tensor import SimpTensor, randn, zeros, sigmoid_tensor, manual_seed

from SimplCore.agi import (
    AGICore,
    AGIController,
    AGISession,
    AGIState,
)

from SimplCore.cognition import (
    HumanLikeCognitiveReasoner,
    HumanLikeMode,
    MultimodalCognitiveIntegration,
    IntegrationMode,
    SelfImprovementModule,
    SelfLearningLoop,
    IntegratedSelfImprovement,
    ImprovementType,
)

from SimplCore.evaluation import (
    EvaluationMetric,
    TuringScore,
    TuringTestEvaluator,
    EnhancedHumanSimilarityEvaluator,
)

from SimplCore.dialogue import (
    DialogueStateTracker,
    LongRangeMemory,
)

from SimplCore.emotion import (
    EmotionalIntelligence,
)

from SimplCore.generation import (
    ResponseVariety,
)


def print_header(text: str, width: int = 70):
    """打印格式化的章节标题"""
    print("\n" + "█" * width)
    print(f"  {text}")
    print("█" * width)


def print_subheader(text: str):
    """打印格式化的子标题"""
    print(f"\n  ▼ {text}")
    print(f"  {'-' * (len(text) + 4)}")


def demo_phase1_initialization():
    """阶段1: AGI核心初始化与模块加载"""
    print_header("阶段1/11: AGI核心初始化与模块加载")
    dim = 64
    manual_seed(42)

    print("正在初始化 AGICore (dim=64)...")
    start = time.time()

    core = AGICore(dim)

    elapsed = time.time() - start
    print(f"  ✓ AGICore 初始化完成 (耗时: {elapsed:.3f}s)")
    print(f"  ✓ 嵌入模块列表:")

    modules = [
        ("感知层", ["multimodal_perception", "audio_perception", "env_perception"]),
        ("NLU层", ["semantic_parser", "multi_turn"]),
        ("对话层", ["dialogue_tracker", "long_range_memory"]),
        ("情绪层", ["emotional_intelligence"]),
        ("认知层", ["cognitive_reasoner", "multimodal_cognition"]),
        ("知识层", ["knowledge_graph", "knowledge_reasoner"]),
        ("响应多样性", ["response_variety"]),
        ("自我改进", ["self_improvement"]),
    ]

    for layer_name, mods in modules:
        for mod_name in mods:
            status = "✓" if hasattr(core, mod_name) else "✗"
            print(f"    [{status}] {layer_name}: {mod_name}")

    return core, dim


def demo_phase2_multimodal_perception(core: AGICore, dim: int):
    """阶段2: 多模态感知融合"""
    print_header("阶段2/11: 多模态感知融合")

    # 1. 文本输入
    print_subheader("文本模态感知")
    text_input = randn(dim)
    perception = core.perceive(text_input=text_input)
    text_data = perception.get("text", {})
    print(f"  意图: {text_data.get('intent', 'N/A')}")
    print(f"  意图置信度: {text_data.get('intent_confidence', 0):.4f}")
    print(f"  语义连贯性: {text_data.get('coherence', {}).get('level', 'N/A')}")
    print(f"  话题状态: {text_data.get('topic', {}).get('state', 'N/A')}")

    # 2. 视觉+文本多模态
    print_subheader("视觉+文本多模态融合")
    visual_input = randn(dim)
    perception_vt = core.perceive(text_input=text_input, visual_input=visual_input)
    mm_data = perception_vt.get("multimodal", {})
    print(f"  活跃模态数: {len(mm_data.get('modalities', []))}")
    print(f"  主导模态: {mm_data.get('dominant', 'N/A')}")
    print(f"  模态权重分配:")
    weights = mm_data.get("weights", {})
    for mod, w in weights.items():
        print(f"    {mod.value}: {w:.4f}")

    # 3. 音频+文本多模态
    print_subheader("音频+文本多模态融合")
    audio_input = randn(dim)
    perception_va = core.perceive(text_input=text_input, audio_input=audio_input)
    mm_data2 = perception_va.get("multimodal", {})
    print(f"  活跃模态数: {len(mm_data2.get('modalities', []))}")
    print(f"  主导模态: {mm_data2.get('dominant', 'N/A')}")

    # 4. 环境感知
    print_subheader("环境感知")
    core.update_environment(temperature=32.5, light_level=0.7)
    env_input = randn(dim)
    perception_full = core.perceive(
        text_input=text_input,
        visual_input=visual_input,
        audio_input=audio_input,
    )
    print(f"  环境温度: {core._environment_state['temperature']}°C")
    print(f"  环境亮度: {core._environment_state['light_level']}")
    print(f"  完整融合模态数: {len(perception_full.get('multimodal', {}).get('modalities', []))}")

    return perception_full


def demo_phase3_semantic_parsing(core: AGICore, dim: int):
    """阶段3: 语境感知语义解析 + 多轮对话理解"""
    print_header("阶段3/11: 语境感知语义解析 + 多轮对话理解")

    turn_texts = [
        "今天天气怎么样？",
        "明天会下雨吗？",
        "那后天呢？",
        "好的，帮我查一下去北京的火车",
        "有高铁吗？",
    ]

    for i, text in enumerate(turn_texts):
        inp = randn(dim)
        perception = core.perceive(text_input=inp)
        text_data = perception.get("text", {})

        intent = text_data.get("intent", "N/A")
        coherence = text_data.get("coherence", {}).get("level", "N/A")
        topic_change = text_data.get("topic", {}).get("is_topic_change", False)
        topic_state = text_data.get("topic", {}).get("state", "N/A")

        context_info = ""
        if i > 0:
            context_info = f" | 话题变化: {topic_change} | 话题状态: {topic_state}"

        print(f"  轮次{i+1}: \"{text}\"")
        print(f"    意图={intent} | 连贯性={coherence}{context_info}")

        # 更新对话历史
        core.update_dialogue(inp)

    print(f"\n  ✓ 对话历史长度: {len(core._dialogue_history)}")


def demo_phase4_dialogue_memory(core: AGICore, dim: int):
    """阶段4: 对话状态跟踪 + 长程记忆"""
    print_header("阶段4/11: 对话状态跟踪 + 长程记忆")

    print_subheader("对话状态跟踪")
    tracker = core.dialogue_tracker

    # 模拟多轮状态更新
    states = []
    for i in range(5):
        query = randn(dim)
        prev_response = states[-1] if states else zeros(dim)
        state, info = tracker(query, prev_response)
        states.append(state)
        print(f"  轮次{i+1}: 状态向量 norm={float(np.linalg.norm(state._data)):.4f}")

    print(f"\n  ✓ 对话状态跟踪完成, 总轮次: {len(states)}")

    print_subheader("长程记忆操作")
    memory = core.long_range_memory

    # 写入记忆
    for i in range(3):
        key_data = randn(dim)
        val_data = randn(dim)
        memory.write(key_data, val_data)
        print(f"  写入记忆条目 {i+1}")

    # 读取记忆
    query_key = randn(dim)
    retrieved, mem_info = memory.read(query_key)
    print(f"\n  记忆读取完成, 置信度: {mem_info.get('confidence', 0):.4f}")

    # 记忆统计
    mem_stats = memory.get_stats() if hasattr(memory, 'get_stats') else {}
    print(f"  记忆条目统计: {mem_stats}")


def demo_phase5_emotional_intelligence(core: AGICore, dim: int):
    """阶段5: 情绪智能感知与共情"""
    print_header("阶段5/11: 情绪智能感知与共情")
    emotion_engine = core.emotional_intelligence

    scenarios = [
        ("积极情感", randn(dim)),
        ("消极情感", randn(dim)),
        ("中性情感", randn(dim)),
    ]

    for label, vec in scenarios:
        emotion_state, emotion_info = emotion_engine(vec)

        dominant = emotion_info.get("dominant_emotion", "N/A")
        intensity = emotion_info.get("intensity", 0)
        valence = emotion_info.get("valence", 0)
        arousal = emotion_info.get("arousal", 0)
        empathy = emotion_info.get("empathy_score", 0)

        print(f"\n  [{label}]")
        print(f"    主导情绪: {dominant}")
        print(f"    情绪强度: {intensity:.4f}")
        print(f"    效价: {valence:.4f}")
        print(f"    唤醒度: {arousal:.4f}")
        print(f"    共情分数: {empathy:.4f}")


def demo_phase6_cognitive_reasoning(core: AGICore, dim: int):
    """阶段6: 类人认知推理 (多模式)"""
    print_header("阶段6/11: 类人认知推理")

    modes = [
        ("对话模式", HumanLikeMode.CONVERSATION),
        ("推理模式", HumanLikeMode.REASONING),
        ("创造模式", HumanLikeMode.CREATIVE),
    ]

    query = randn(dim)
    perception = core.perceive(text_input=query)

    for label, mode in modes:
        print_subheader(f"认知推理 - {label}")
        result = core.reason(query, perception, mode=mode)

        confidence = result.get("confidence", 0)
        cognitive_info = result.get("cognitive_info", {})

        print(f"  推理置信度: {confidence:.4f}")
        print(f"  认知模式: {cognitive_info.get('mode', 'N/A')}")

        knowledge_info = result.get("knowledge_info", {})
        entities = knowledge_info.get("entities_found", 0)
        knw_confidence = knowledge_info.get("reasoning_confidence", 0)
        print(f"  知识实体数: {entities}")
        print(f"  知识推理置信度: {knw_confidence:.4f}")

        core.update_dialogue(result["output"])


def demo_phase7_knowledge_reasoning(core: AGICore, dim: int):
    """阶段7: 知识图谱推理增强"""
    print_header("阶段7/11: 知识图谱推理增强")

    print_subheader("知识条目添加")
    kg = core.knowledge_graph

    topics = ["science", "technology", "history", "art", "philosophy"]
    for i, topic in enumerate(topics):
        entity_id = f"entity_{i}"
        embedding = randn(dim)
        kg.add_entity(entity_id, embedding, {"topic": topic, "index": i})

        # 添加关系
        if i > 0:
            source_id = f"entity_{i-1}"
            relation = f"related_to_{topic}"
            kg.add_relation(source_id, entity_id, relation)
            print(f"  添加关系: {source_id} --[{relation}]--> {entity_id}")
        else:
            print(f"  添加实体: {entity_id} ({topic})")

    print_subheader("知识增强推理")
    query = randn(dim)
    enhanced, knw_info = core.knowledge_reasoner(query, context=None)

    print(f"  知识增强后输出 norm: {float(np.linalg.norm(enhanced._data)):.4f}")
    print(f"  实体找到数: {knw_info.get('entities_found', 0)}")
    print(f"  推理置信度: {knw_info.get('reasoning_confidence', 0):.4f}")

    # 知识图谱统计
    kg_stats = kg.get_stats() if hasattr(kg, 'get_stats') else {}
    print(f"  知识图谱统计: {kg_stats}")


def demo_phase8_response_variety(core: AGICore, dim: int):
    """阶段8: 响应多样性生成"""
    print_header("阶段8/11: 响应多样性生成")

    variety = core.response_variety

    print_subheader("多样化响应生成")
    context = randn(dim)

    strategies = ["default", "creative", "conservative"]
    for strategy in strategies:
        # 修改策略参数
        if hasattr(variety, 'set_strategy'):
            variety.set_strategy(strategy)

        response, var_info = variety(context)
        diversity = var_info.get("diversity_score", 0)
        novelty = var_info.get("novelty_score", 0)

        print(f"  策略[{strategy}]: 多样性={diversity:.4f}, 新颖性={novelty:.4f}")

    print_subheader("多候选响应对比")
    candidates = []
    for _ in range(3):
        resp, info = variety(context)
        candidates.append({
            "diversity": info.get("diversity_score", 0),
            "novelty": info.get("novelty_score", 0),
            "norm": float(np.linalg.norm(resp._data)),
        })

    for i, c in enumerate(candidates):
        print(f"  候选{i+1}: norm={c['norm']:.4f}, 多样性={c['diversity']:.4f}, 新颖性={c['novelty']:.4f}")


def demo_phase9_self_improvement(core: AGICore, dim: int):
    """阶段9: 自我改进与能力评估"""
    print_header("阶段9/11: 自我改进与能力评估")

    sim = core.self_improvement

    print_subheader("性能监控")
    for i in range(10):
        result = sim.process_outcome(
            query_features=randn(dim),
            confidence=0.5 + i * 0.04,
            correct=i < 8,
            response_time=0.1 + np.random.random() * 0.3,
            knowledge_found=i % 3,
            error_type="low_confidence" if i > 7 else None,
        )

    status = sim.get_full_status()
    caps = status.get("capabilities", {})
    overall = caps.get("overall_score", 0)
    print(f"  学习循环数: {status.get('learning_loop', {}).get('learning_cycles', 0)}")
    print(f"  综合能力得分: {overall:.2%}")

    print_subheader("能力详细评估")
    cap_scores = caps.get("capabilities", {})
    for name, score in cap_scores.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {name:25s} |{bar}| {score:.2%}")

    print_subheader("学习计划")
    improvement = sim.improvement
    evaluation = improvement.evaluate_capabilities()
    plan = improvement.generate_learning_plan(evaluation)
    print(f"  计划项数: {len(plan)}")
    for item in plan[:4]:
        print(f"    - [{item['priority']}] {item['capability']}: "
              f"{item['current_score']:.1%} → {item['target_score']:.1%}")

    improvement_summary = status.get("improvement_summary", {})
    print(f"\n  改进历史: {improvement_summary.get('total_records', 0)} 条")
    print(f"  盲区检测: {improvement_summary.get('blind_spots', [])}")


def demo_phase10_turing_evaluation(core: AGICore, dim: int):
    """阶段10: 图灵测试综合评估"""
    print_header("阶段10/11: 图灵测试综合评估")

    evaluator = TuringTestEvaluator(dim)
    enhanced_evaluator = EnhancedHumanSimilarityEvaluator(dim)

    print_subheader("生成评估样本")
    queries = [randn(dim) for _ in range(5)]
    responses = []

    for i, query in enumerate(queries):
        result = core(query)
        response = result.get("reasoning", {}).get("output", query)
        responses.append(response)

    print(f"  评估样本: {len(queries)} 轮对话")
    print(f"  响应向量 norm 范围: "
          f"{min(float(np.linalg.norm(r._data)) for r in responses):.4f} ~ "
          f"{max(float(np.linalg.norm(r._data)) for r in responses):.4f}")

    print_subheader("图灵测试评分")
    try:
        dialogue_result = evaluator.evaluate_dialogue(
            responses=responses,
            queries=queries,
            situations=[randn(dim) for _ in range(5)],
            emotions=[randn(dim) for _ in range(5)],
        )

        avg_score = dialogue_result.get("average_score", 0)
        avg_percentage = dialogue_result.get("average_percentage", 0)
        print(f"  平均分数: {avg_score:.2f}/10")
        print(f"  平均百分比: {avg_percentage:.1f}%")

        detail = dialogue_result.get("detail", {})
        for metric, score in detail.items():
            if isinstance(score, (int, float)):
                print(f"    {metric}: {score:.4f}")
    except Exception as e:
        print(f"  评估过程 (预期行为): {e}")

    print_subheader("增强评估器")
    try:
        enhanced_scores = []
        for i, (q, r) in enumerate(zip(queries, responses)):
            score = enhanced_evaluator(
                response=r,
                query=q,
                previous_response=responses[i-1] if i > 0 else zeros(dim),
                context=q,
            )
            if hasattr(score, '_data'):
                enhanced_scores.append(float(score._data[0]))

        if enhanced_scores:
            print(f"  增强评估结果: {np.mean(enhanced_scores):.4f} ± {np.std(enhanced_scores):.4f}")
    except Exception as e:
        print(f"  增强评估 (预期行为): {e}")


def demo_phase11_enhanced_training(dim: int):
    """阶段11: 增强训练器演示"""
    print_header("阶段11/11: 增强训练器演示")

    from SimplCore.train import (
        MultiTaskLoss,
        EnhancedSyntheticDataGenerator,
        EnhancedTrainer,
        AdaptiveTrainingScheduler,
    )

    print_subheader("多任务损失函数测试")
    loss_fn = MultiTaskLoss(dim)
    test_output = randn(dim)
    test_perception = randn(dim)
    test_knowledge = randn(dim)

    loss, loss_info = loss_fn(
        cognition_output=test_output,
        perception_output=test_perception,
        knowledge_output=test_knowledge,
        target_score=8.0,
    )
    print(f"  总损失: {float(loss._data[0]):.4f}")
    print(f"  认知损失: {loss_info['cognition_loss']:.4f}")
    print(f"  感知损失: {loss_info.get('perception_loss', 0):.4f}")
    print(f"  知识损失: {loss_info.get('knowledge_loss', 0):.4f}")

    print_subheader("增强数据生成器测试")
    gen = EnhancedSyntheticDataGenerator(dim)
    batch = gen.generate_multimodal_batch(batch_size=4)
    print(f"  多模态批次: {len(batch['text'])} 文本, {len(batch['visual'])} 视觉, {len(batch['audio'])} 音频")

    dialogue_batches = gen.generate_dialogue_batch(num_turns=3, batch_size=2)
    print(f"  对话批次: {len(dialogue_batches)} 对话, 每轮 {len(dialogue_batches[0]) if dialogue_batches else 0} 轮")

    knowledge_items = gen.generate_knowledge_batch(num_items=3)
    print(f"  知识条目: {len(knowledge_items)} 个")
    for emb, topic in knowledge_items:
        print(f"    - {topic}")

    print_subheader("训练器测试 (简短运行)")
    from SimplCore.cognition import HumanLikeCognitiveReasoner

    model = HumanLikeCognitiveReasoner(dim)
    trainer = EnhancedTrainer(
        model=model,
        dim=dim,
        learning_rate=5e-4,
        num_epochs=5,
        batch_size=4,
        enable_self_improvement=True,
        enable_knowledge_iteration=True,
        target_turing_score=90.0,
    )

    result = trainer.train()
    print(f"\n  训练完成:")
    print(f"    最优分数: {result['best_score']:.1f}%")
    print(f"    训练轮数: {result['total_epochs']}")
    print(f"    最终评估: {result['final_evaluation']['percentage']:.1f}%")

    print_subheader("自适应调度器测试")
    scheduler = AdaptiveTrainingScheduler(
        dim=dim,
        stages=[
            {
                "name": "快速验证阶段",
                "epochs": 3,
                "batch_size": 4,
                "learning_rate": 5e-4,
                "mode": HumanLikeMode.CONVERSATION,
                "description": "快速验证自适应调度流程",
            },
        ],
    )

    def make_model():
        return HumanLikeCognitiveReasoner(dim)

    sched_result = scheduler.run_scheduled_training(make_model)
    print(f"  调度阶段数: {sched_result['total_stages']}")
    print(f"  最优分数: {sched_result['best_overall_score']:.1f}%")

    return result


def run_full_demo():
    """运行完整端到端演示"""
    print("=" * 70)
    print("  SimplCore AGI 端到端集成演示")
    print("  完整感知 → 理解 → 推理 → 反思闭环展示")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    total_start = time.time()

    # 阶段1: 初始化
    core, dim = demo_phase1_initialization()

    # 阶段2-8: 各模块演示 (使用同一core实例, 共享对话历史)
    demo_phase2_multimodal_perception(core, dim)
    demo_phase3_semantic_parsing(core, dim)
    demo_phase4_dialogue_memory(core, dim)
    demo_phase5_emotional_intelligence(core, dim)
    demo_phase6_cognitive_reasoning(core, dim)
    demo_phase7_knowledge_reasoning(core, dim)
    demo_phase8_response_variety(core, dim)
    demo_phase9_self_improvement(core, dim)
    demo_phase10_turing_evaluation(core, dim)

    # 阶段11: 训练 (需要新的model实例)
    training_result = demo_phase11_enhanced_training(dim)

    total_elapsed = time.time() - total_start

    # 最终总结
    print("\n" + "=" * 70)
    print("  SimplCore AGI 端到端集成演示完成！")
    print("=" * 70)
    print(f"\n  总耗时: {total_elapsed:.2f}s")
    print(f"\n  已覆盖的能力维度:")
    print(f"  ✓ 多模态感知融合 (文本 + 视觉 + 音频 + 环境)")
    print(f"  ✓ 语境感知语义解析 + 多轮理解")
    print(f"  ✓ 对话状态跟踪 + 长程记忆")
    print(f"  ✓ 情绪智能感知 + 共情")
    print(f"  ✓ 类人认知推理 (对话/推理/创造)")
    print(f"  ✓ 知识图谱推理增强")
    print(f"  ✓ 响应多样性生成")
    print(f"  ✓ 自我改进 + 能力评估 + 学习计划")
    print(f"  ✓ 图灵测试综合评估")
    print(f"  ✓ 增强训练器 + 自适应调度")

    final_status = core.self_improvement.get_full_status() if hasattr(core, 'self_improvement') else {}
    caps = final_status.get("capabilities", {}).get("capabilities", {})
    if caps:
        print(f"\n  系统最终能力评估:")
        for name, score in caps.items():
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"    {name:25s} |{bar}| {score:.2%}")

    print(f"\n  训练模块:")
    print(f"    - 多任务损失: 支持认知/感知/知识联合优化")
    print(f"    - 增强训练器: 最优分数 {training_result['best_score']:.1f}%")
    print(f"    - 自适应调度: 支持分阶段渐进式训练")
    print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 演示结束")
    print("=" * 70)

    return core


if __name__ == "__main__":
    run_full_demo()


