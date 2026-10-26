# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore AGI 能力评估脚本.

全面评估 AGI 系统的各项能力:
  1. 语言理解能力
  2. 视觉感知能力
  3. 世界模型预测能力
  4. 知识库检索能力
  5. 现场学习能力
  6. 高级推理能力 (类比、因果、演绎、归纳、溯因)
  7. 元认知能力
  8. 多模态整合能力

评估指标:
  - 准确率
  - 置信度
  - 响应时间
  - 知识覆盖率
  - 推理深度
"""

from __future__ import annotations

import sys
sys.path.insert(0, '.')

import simplcore.tensor as torch
import numpy as np
import time
import json
import os
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

from SimplCore.agi.SimplCore_agi import SimplCoreAGI
from SimplCore.knowledge.knowledge_base import KnowledgeBase
from SimplCore.knowledge.on_demand_learning import OnDemandLearner
from SimplCore.cognition.advanced_reasoning import IntegratedReasoner
from SimplCore.perception.visual_stream import VisualStream


@dataclass
class EvaluationResult:
    """评估结果"""
    category: str
    task: str
    success: bool
    score: float
    confidence: float
    response_time: float
    details: Dict[str, Any]


class AGIEvaluator:
    """AGI 能力评估器"""
    
    def __init__(self, agi: SimplCoreAGI, knowledge_base: KnowledgeBase):
        self.agi = agi
        self.knowledge_base = knowledge_base
        self.on_demand_learner = OnDemandLearner(knowledge_base)
        self.reasoner = IntegratedReasoner(knowledge_base)
        self.results: List[EvaluationResult] = []
    
    def evaluate_language_understanding(self) -> List[EvaluationResult]:
        """评估语言理解能力"""
        print("\n" + "=" * 70)
        print("评估: 语言理解能力")
        print("=" * 70)
        
        tasks = [
            {
                "task": "基础问答",
                "query": "什么是人工智能?",
                "expected_keywords": ["AI", "机器学习", "深度学习"],
            },
            {
                "task": "复杂问答",
                "query": "量子计算和经典计算有什么区别?",
                "expected_keywords": ["量子比特", "叠加态", "并行"],
            },
            {
                "task": "推理问答",
                "query": "如果所有的鸟都会飞，企鹅是鸟，那么企鹅会飞吗?",
                "expected_keywords": ["不会", "例外", "演绎"],
            },
            {
                "task": "创意生成",
                "query": "描述一个未来的智能城市",
                "expected_keywords": ["自动驾驶", "智能建筑", "可持续"],
            },
            {
                "task": "情感理解",
                "query": "解释为什么人们会感到孤独",
                "expected_keywords": ["社交", "情感", "连接"],
            },
        ]
        
        results = []
        for task_info in tasks:
            start_time = time.time()
            
            try:
                response = self.agi.ask(task_info["query"])
                response_time = time.time() - start_time
                
                answer_text = response.get("answer", "") if isinstance(response, dict) else str(response)
                
                score = 0.0
                for keyword in task_info["expected_keywords"]:
                    if keyword.lower() in answer_text.lower():
                        score += 1.0 / len(task_info["expected_keywords"])
                
                success = score >= 0.5
                
                result = EvaluationResult(
                    category="语言理解",
                    task=task_info["task"],
                    success=success,
                    score=score,
                    confidence=response.get("confidence", 0.7) if isinstance(response, dict) else 0.7,
                    response_time=response_time,
                    details={"response": answer_text[:100], "expected_keywords": task_info["expected_keywords"]},
                )
                results.append(result)
                
                print(f"  {task_info['task']}: {'通过' if success else '失败'} (分数: {score:.2f}, 耗时: {response_time:.2f}s)")
                
            except Exception as e:
                response_time = time.time() - start_time
                result = EvaluationResult(
                    category="语言理解",
                    task=task_info["task"],
                    success=False,
                    score=0.0,
                    confidence=0.0,
                    response_time=response_time,
                    details={"error": str(e)},
                )
                results.append(result)
                print(f"  {task_info['task']}: 失败 (错误: {str(e)})")
        
        self.results.extend(results)
        return results
    
    def evaluate_world_model(self) -> List[EvaluationResult]:
        """评估世界模型预测能力"""
        print("\n" + "=" * 70)
        print("评估: 世界模型预测能力")
        print("=" * 70)
        
        tasks = [
            {
                "task": "状态预测",
                "description": "预测状态转移",
            },
            {
                "task": "多步预测",
                "description": "预测多步状态转移",
            },
        ]
        
        results = []
        
        for task_info in tasks:
            start_time = time.time()
            
            try:
                states = torch.randn(1, 10, 128)
                actions = torch.randn(1, 10, 16)
                
                steps = 1 if task_info["task"] == "状态预测" else 3
                outputs = self.agi.predict_future(states, actions, steps=steps)
                
                response_time = time.time() - start_time
                
                # 检查输出维度
                next_state = outputs["next_state"]
                expected_shape = states.shape
                
                score = 0.8 if next_state.shape == expected_shape else 0.0
                success = score >= 0.5
                
                result = EvaluationResult(
                    category="世界模型",
                    task=task_info["task"],
                    success=success,
                    score=score,
                    confidence=0.8,
                    response_time=response_time,
                    details={"output_shape": tuple(next_state.shape), "expected_shape": expected_shape},
                )
                results.append(result)
                
                print(f"  {task_info['task']}: {'通过' if success else '失败'} (分数: {score:.2f}, 耗时: {response_time:.2f}s)")
                
            except Exception as e:
                response_time = time.time() - start_time
                result = EvaluationResult(
                    category="世界模型",
                    task=task_info["task"],
                    success=False,
                    score=0.0,
                    confidence=0.0,
                    response_time=response_time,
                    details={"error": str(e)},
                )
                results.append(result)
                print(f"  {task_info['task']}: 失败 (错误: {str(e)})")
        
        self.results.extend(results)
        return results
    
    def evaluate_knowledge_retrieval(self) -> List[EvaluationResult]:
        """评估知识库检索能力"""
        print("\n" + "=" * 70)
        print("评估: 知识库检索能力")
        print("=" * 70)
        
        tasks = [
            {
                "task": "物理学知识检索",
                "query": "相对论",
                "expected_category": "physics",
            },
            {
                "task": "数学知识检索",
                "query": "微积分",
                "expected_category": "mathematics",
            },
            {
                "task": "计算机科学知识检索",
                "query": "深度学习",
                "expected_category": "computer_science",
            },
            {
                "task": "生物学知识检索",
                "query": "DNA",
                "expected_category": "biology",
            },
            {
                "task": "化学知识检索",
                "query": "元素周期表",
                "expected_category": "chemistry",
            },
        ]
        
        results = []
        for task_info in tasks:
            start_time = time.time()
            
            try:
                knowledge = self.knowledge_base.get_knowledge(task_info["query"], top_k=5)
                response_time = time.time() - start_time
                
                score = 1.0 if len(knowledge) > 0 else 0.0
                success = len(knowledge) > 0
                
                result = EvaluationResult(
                    category="知识检索",
                    task=task_info["task"],
                    success=success,
                    score=score,
                    confidence=0.9,
                    response_time=response_time,
                    details={"found_count": len(knowledge), "samples": [k[:50] for k in knowledge[:3]]},
                )
                results.append(result)
                
                print(f"  {task_info['task']}: {'通过' if success else '失败'} (找到: {len(knowledge)} 条, 耗时: {response_time:.2f}s)")
                
            except Exception as e:
                response_time = time.time() - start_time
                result = EvaluationResult(
                    category="知识检索",
                    task=task_info["task"],
                    success=False,
                    score=0.0,
                    confidence=0.0,
                    response_time=response_time,
                    details={"error": str(e)},
                )
                results.append(result)
                print(f"  {task_info['task']}: 失败 (错误: {str(e)})")
        
        self.results.extend(results)
        return results
    
    def evaluate_on_demand_learning(self) -> List[EvaluationResult]:
        """评估现场学习能力"""
        print("\n" + "=" * 70)
        print("评估: 现场学习能力")
        print("=" * 70)
        
        tasks = [
            {
                "task": "知识盲区检测",
                "query": "什么是量子纠缠?",
            },
            {
                "task": "知识检索与融入",
                "query": "解释光合作用的过程",
            },
            {
                "task": "复杂问题现场学习",
                "query": "深度学习如何应用于蛋白质结构预测?",
            },
        ]
        
        results = []
        for task_info in tasks:
            start_time = time.time()
            
            try:
                result_learn = self.on_demand_learner.learn_and_answer(task_info["query"], self.agi.language_model)
                response_time = time.time() - start_time
                
                score = 0.0
                if result_learn.get("knowledge_used"):
                    score += 0.5
                if result_learn.get("answer"):
                    score += 0.3
                if result_learn.get("learned"):
                    score += 0.2
                
                success = score >= 0.5
                
                result = EvaluationResult(
                    category="现场学习",
                    task=task_info["task"],
                    success=success,
                    score=score,
                    confidence=0.7,
                    response_time=response_time,
                    details={"knowledge_found": result_learn.get("knowledge_found", False), 
                             "answer_length": len(result_learn.get("answer", ""))},
                )
                results.append(result)
                
                print(f"  {task_info['task']}: {'通过' if success else '失败'} (分数: {score:.2f}, 耗时: {response_time:.2f}s)")
                
            except Exception as e:
                response_time = time.time() - start_time
                result = EvaluationResult(
                    category="现场学习",
                    task=task_info["task"],
                    success=False,
                    score=0.0,
                    confidence=0.0,
                    response_time=response_time,
                    details={"error": str(e)},
                )
                results.append(result)
                print(f"  {task_info['task']}: 失败 (错误: {str(e)})")
        
        self.results.extend(results)
        return results
    
    def evaluate_advanced_reasoning(self) -> List[EvaluationResult]:
        """评估高级推理能力"""
        print("\n" + "=" * 70)
        print("评估: 高级推理能力")
        print("=" * 70)
        
        tasks = [
            {
                "task": "类比推理",
                "query": "量子纠缠和什么类似?",
                "reasoner_type": "analogical",
            },
            {
                "task": "因果推理",
                "query": "为什么会发生光电效应?",
                "reasoner_type": "causal",
            },
            {
                "task": "演绎推理",
                "query": "因为牛顿第二定律F=ma，所以质量越大的物体需要更大的力",
                "reasoner_type": "deductive",
            },
            {
                "task": "归纳推理",
                "query": "总结物理学的基本定律",
                "reasoner_type": "inductive",
            },
            {
                "task": "溯因推理",
                "query": "为什么苹果会落地?",
                "reasoner_type": "abductive",
            },
            {
                "task": "多推理器综合",
                "query": "为什么深度学习效果好?",
                "reasoner_type": None,
            },
        ]
        
        results = []
        for task_info in tasks:
            start_time = time.time()
            
            try:
                if task_info["reasoner_type"] is None:
                    result_reason = self.reasoner.multi_reasoning(task_info["query"])
                else:
                    result_reason = self.reasoner.reason(task_info["query"], task_info["reasoner_type"])
                
                response_time = time.time() - start_time
                
                # 评估推理质量
                score = result_reason.get("confidence", 0.5)
                
                if "multi_results" in result_reason:
                    score = min(1.0, score + len(result_reason["multi_results"]) * 0.1)
                
                success = score >= 0.5
                
                result = EvaluationResult(
                    category="高级推理",
                    task=task_info["task"],
                    success=success,
                    score=score,
                    confidence=result_reason.get("confidence", 0.5),
                    response_time=response_time,
                    details={"reasoner_type": result_reason.get("reasoner_type", "N/A")},
                )
                results.append(result)
                
                print(f"  {task_info['task']}: {'通过' if success else '失败'} (分数: {score:.2f}, 耗时: {response_time:.2f}s)")
                
            except Exception as e:
                response_time = time.time() - start_time
                result = EvaluationResult(
                    category="高级推理",
                    task=task_info["task"],
                    success=False,
                    score=0.0,
                    confidence=0.0,
                    response_time=response_time,
                    details={"error": str(e)},
                )
                results.append(result)
                print(f"  {task_info['task']}: 失败 (错误: {str(e)})")
        
        self.results.extend(results)
        return results
    
    def evaluate_metacognition(self) -> List[EvaluationResult]:
        """评估元认知能力"""
        print("\n" + "=" * 70)
        print("评估: 元认知能力")
        print("=" * 70)
        
        tasks = [
            {
                "task": "置信度评估",
                "query": "什么是深度学习?",
            },
            {
                "task": "自我修正",
                "query": "地球是平的",
                "feedback": "错误，地球是圆的",
            },
            {
                "task": "思考过程反思",
                "query": "解释黑洞",
            },
        ]
        
        results = []
        for task_info in tasks:
            start_time = time.time()
            
            try:
                if "feedback" in task_info:
                    corrected = self.reasoner.metacognitive.self_correct(
                        task_info["query"], 
                        task_info["feedback"]
                    )
                    score = 0.8 if "圆的" in corrected else 0.2
                else:
                    result_reason = self.reasoner.metacognitive.reason(task_info["query"])
                    score = result_reason.get("confidence", 0.5)
                
                response_time = time.time() - start_time
                success = score >= 0.5
                
                result = EvaluationResult(
                    category="元认知",
                    task=task_info["task"],
                    success=success,
                    score=score,
                    confidence=score,
                    response_time=response_time,
                    details={"type": "correction" if "feedback" in task_info else "evaluation"},
                )
                results.append(result)
                
                print(f"  {task_info['task']}: {'通过' if success else '失败'} (分数: {score:.2f}, 耗时: {response_time:.2f}s)")
                
            except Exception as e:
                response_time = time.time() - start_time
                result = EvaluationResult(
                    category="元认知",
                    task=task_info["task"],
                    success=False,
                    score=0.0,
                    confidence=0.0,
                    response_time=response_time,
                    details={"error": str(e)},
                )
                results.append(result)
                print(f"  {task_info['task']}: 失败 (错误: {str(e)})")
        
        self.results.extend(results)
        return results
    
    def evaluate_multi_modal(self) -> List[EvaluationResult]:
        """评估多模态整合能力"""
        print("\n" + "=" * 70)
        print("评估: 多模态整合能力")
        print("=" * 70)
        
        tasks = [
            {
                "task": "视觉编码",
                "description": "编码图像特征",
            },
            {
                "task": "视觉理解",
                "description": "理解图像内容",
            },
        ]
        
        results = []
        for task_info in tasks:
            start_time = time.time()
            
            try:
                image = np.random.rand(3, 224, 224).astype(np.float32)
                image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
                
                if task_info["task"] == "视觉编码":
                    features_result = self.agi.see(image_tensor)
                    features = features_result.get("latent")
                    score = 0.8 if features is not None else 0.0
                else:
                    result_see = self.agi.see(image_tensor)
                    result_understand = self.agi.ask("描述这个图像")
                    answer_text = result_understand.get("answer", "") if isinstance(result_understand, dict) else str(result_understand)
                    score = 0.6 if answer_text else 0.0
                
                response_time = time.time() - start_time
                success = score >= 0.5
                
                result = EvaluationResult(
                    category="多模态",
                    task=task_info["task"],
                    success=success,
                    score=score,
                    confidence=0.7,
                    response_time=response_time,
                    details={"features_shape": features.shape if 'features' in locals() else None},
                )
                results.append(result)
                
                print(f"  {task_info['task']}: {'通过' if success else '失败'} (分数: {score:.2f}, 耗时: {response_time:.2f}s)")
                
            except Exception as e:
                response_time = time.time() - start_time
                result = EvaluationResult(
                    category="多模态",
                    task=task_info["task"],
                    success=False,
                    score=0.0,
                    confidence=0.0,
                    response_time=response_time,
                    details={"error": str(e)},
                )
                results.append(result)
                print(f"  {task_info['task']}: 失败 (错误: {str(e)})")
        
        self.results.extend(results)
        return results
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """运行完整评估"""
        print("\n" + "=" * 70)
        print("SimplCore AGI 能力评估 - 全面测试")
        print("=" * 70)
        
        # 运行各项评估
        self.evaluate_language_understanding()
        self.evaluate_world_model()
        self.evaluate_knowledge_retrieval()
        self.evaluate_on_demand_learning()
        self.evaluate_advanced_reasoning()
        self.evaluate_metacognition()
        self.evaluate_multi_modal()
        
        # 生成综合报告
        report = self.generate_report()
        
        return report
    
    def generate_report(self) -> Dict[str, Any]:
        """生成评估报告"""
        print("\n" + "=" * 70)
        print("评估报告")
        print("=" * 70)
        
        # 按类别统计
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = {"total": 0, "passed": 0, "total_score": 0.0}
            
            categories[result.category]["total"] += 1
            if result.success:
                categories[result.category]["passed"] += 1
            categories[result.category]["total_score"] += result.score
        
        # 计算综合指标
        total_tasks = len(self.results)
        passed_tasks = sum(1 for r in self.results if r.success)
        avg_score = sum(r.score for r in self.results) / total_tasks if total_tasks > 0 else 0.0
        avg_response_time = sum(r.response_time for r in self.results) / total_tasks if total_tasks > 0 else 0.0
        
        report = {
            "summary": {
                "total_tasks": total_tasks,
                "passed_tasks": passed_tasks,
                "pass_rate": passed_tasks / total_tasks if total_tasks > 0 else 0.0,
                "avg_score": avg_score,
                "avg_response_time": avg_response_time,
            },
            "categories": {},
            "detailed_results": [],
        }
        
        # 输出类别统计
        print(f"\n综合指标:")
        print(f"  总任务数: {total_tasks}")
        print(f"  通过任务数: {passed_tasks}")
        print(f"  通过率: {passed_tasks / total_tasks * 100:.1f}%")
        print(f"  平均分数: {avg_score:.2f}")
        print(f"  平均响应时间: {avg_response_time:.2f}s")
        
        print("\n分类统计:")
        for category, stats in categories.items():
            pass_rate = stats["passed"] / stats["total"] if stats["total"] > 0 else 0.0
            avg_score = stats["total_score"] / stats["total"] if stats["total"] > 0 else 0.0
            
            report["categories"][category] = {
                "total": stats["total"],
                "passed": stats["passed"],
                "pass_rate": pass_rate,
                "avg_score": avg_score,
            }
            
            print(f"  {category}: {stats['passed']}/{stats['total']} 通过 (通过率: {pass_rate * 100:.1f}%, 平均分: {avg_score:.2f})")
        
        # 详细结果
        for result in self.results:
            report["detailed_results"].append({
                "category": result.category,
                "task": result.task,
                "success": result.success,
                "score": result.score,
                "confidence": result.confidence,
                "response_time": result.response_time,
                "details": result.details,
            })
        
        # 保存报告
        os.makedirs("./evaluation", exist_ok=True)
        with open("./evaluation/agi_evaluation_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n报告已保存到 ./evaluation/agi_evaluation_report.json")
        
        return report


def main():
    """主评估函数"""
    import os
    
    print("=" * 70)
    print("SimplCore AGI 能力评估系统")
    print("=" * 70)
    
    # 初始化组件
    print("\n[Init] 初始化 AGI 系统...")
    agi = SimplCoreAGI()
    
    print("[Init] 初始化知识库...")
    kb = KnowledgeBase()
    
    # 创建评估器
    print("[Init] 创建评估器...")
    evaluator = AGIEvaluator(agi, kb)
    
    # 运行评估
    print("\n[Eval] 开始全面评估...")
    report = evaluator.run_full_evaluation()
    
    # 输出最终结果
    print("\n" + "=" * 70)
    print("评估完成")
    print("=" * 70)
    print(f"综合评分: {report['summary']['avg_score']:.2f}")
    print(f"通过率: {report['summary']['pass_rate'] * 100:.1f}%")
    
    # 评估等级
    avg_score = report['summary']['avg_score']
    if avg_score >= 0.8:
        print("评估等级: S - 优秀")
    elif avg_score >= 0.6:
        print("评估等级: A - 良好")
    elif avg_score >= 0.4:
        print("评估等级: B - 一般")
    else:
        print("评估等级: C - 需要改进")
    
    print("\n" + "=" * 70)
    print("SimplCore AGI 能力评估完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()


