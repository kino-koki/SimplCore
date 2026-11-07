# 01 · cfp_lock.md — 当年会议参数锁定表

> 基线日期 2026-11-07（Asia/Shanghai）。查询来源均为 ICML 官方网站或 ICML Blog 原文。
> 本表遵守技能包 §2.2：每一项标注「已核验 / 沿用上一版待复核 / 官方未公布」，禁止凭记忆写死。

## 输入契约放行检查（Phase 0 入口）

| 输入 | 状态 | 位置 |
|---|---|---|
| paper.pdf + 可检索全文（含全部附录、References、Broader Impact） | ✅ | `simplcore.pdf`（16 页）+ `paper_fulltext.txt` |
| 当年官方 CFP / Author Instructions / 重要日期页原文要点 | ✅ | 本文件 §1，官网原文已取 |
| 原始数据文件清单（每个数字的来源、seed 数、精度、样本量） | ✅ | `number_freeze.md` + Appendix I |
| 上一版稿件与版本差异说明 | ✅ | `V65_REPORT.md`（v6.5）→ 本轮 v7.1 差异见 `REVISION_LOG.md` |
| 委托人「不主张清单」 | ✅ | 见 §4 |

三项触发条件全部满足（ICML 主会理论/分析类；含 ≥3 条自称紧界/IFF/EXACTLY 的声明；含撤回登记与跨版本前提演化记录）。**Phase 0 放行。**

## 1. 锁定表

```yaml
venue: ICML 2027 (Forty-Fourth International Conference on Machine Learning)
conference_dates: 2027-07-31 .. 2027-08-06      # 已核验: icml.cc/Conferences/2027 + /Conferences/current/Dates
date_structure:                                  # 已核验: 官网 2027 Conference 页
  expo: 2027-07-31
  tutorials: 2027-08-01
  main_conference: 2027-08-02 .. 2027-08-04
  workshops: 2027-08-05 .. 2027-08-06
location: Riocentro, Rio de Janeiro, Brazil      # 已核验: 官网 "ICML 2027 dates and location confirmed: Riocentro, Rio de Janeiro, Brazil"
cfp_published: false                             # 已核验: 2026-11-07 查询
cfp_evidence: >-
  icml.cc/Conferences/2027/AuthorInstructions 原文:
  "The ICML 2027 author instructions will be published with the Call for Papers."
  icml.cc 2027 Conference 页 Important Dates 原文:
  "Key deadlines and events will be posted as they are confirmed ...
   The ICML 2027 submission deadlines and registration dates will appear
   here once they are confirmed."
submission_deadline: UNCONFIRMED                 # 官方未公布；禁止写入任何估算值
abstract_deadline: UNCONFIRMED                   # 官方未公布
page_limit_main: 8                               # 沿用上一版待复核 (ICML 2026 值, 非 2027 结论)
references_count_toward_limit: false             # 沿用上一版待复核 (ICML 2026: 不计入 8 页)
appendix_policy: unlimited_pages                 # 沿用上一版待复核 (ICML 2026: 附录/参考文献不限页, 整份 PDF <= 10 MB)
style_file_version: icml2026.sty                 # 沿用上一版待复核; 2027 模板未发布
style_file_2027: UNPUBLISHED
double_blind: true                               # 沿用上一版待复核
anonymization_rules: 待 2027 CFP 逐条摘录
broader_impact_position: before_references       # 本稿已按此排布 (见 paper_main.pdf 第 8 页)
broader_impact_counts_to_limit: false            # 沿用上一版待复核
reproducibility_requirements: 待 2027 CFP 逐条摘录; 本稿已备 Appendix C + 全部原始结果表
llm_use_disclosure: 2027 政策随 CFP 发布          # 查询日期 2026-11-07; 当前稿不作绝对化披露句
prompt_injection_prohibited: true                # 已核验: ICML 2026 Peer Review Blog
hallucinated_citations_desk_reject: true         # 已核验: ICML 2026 desk-rejection grounds
thinly_sliced_contributions: strict              # 已核验: ICML 2026 强化 concurrent submission 约束
reciprocal_reviewer_required: true               # 已核验: ICML 2026 "each paper must nominate a qualified reciprocal reviewer"
reviewer_failure_cascades: true                  # 已核验: ICML 2026 "if any reviewer or AC fails in their duties, all of their papers might be desk rejected"
concurrent_submission_policy: 待 2027 CFP 逐条摘录
desk_reject_grounds: 待 2027 CFP 逐条摘录; 2026 已含超页数/格式/破坏匿名/幻觉引用/占位摘要/摘要与全文差异过大/prompt injection
camera_ready_deadline: UNCONFIRMED
```

## 2. 一个必须记录的核验判断

第三方会议截止日汇总站（OpenCurious）列出 `ICML 2027 paper submission deadline = Jan 22, 2027`，但**同一页面**写着 "The ICML 2027 location has not yet been announced" —— 而 ICML 官网已公布地点为 Riocentro, Rio de Janeiro。该第三方条目与官方已核验事实冲突，判为陈旧数据，**不采信**。技能包 §2.2 禁止把估算值写成承诺，故 `submission_deadline = UNCONFIRMED`。

另一个同名缩写 `ICML 2027`（icml2027.eurac.edu，XXI International Conference on Minority Languages）与机器学习会议无关，搜索噪声，已排除。

## 3. 政策风险对本稿的具体含义（2026 版观察，2027 须重核）

| 风险 | 对本稿 | 本轮处置 |
|---|---|---|
| thinly sliced contributions / concurrent submission 连坐 | 本稿有 v5→v6.5 多版演化，附录曾带另一 self-evolution 项目的工程日志 | v7 已删旧 ledger；本轮进一步把正文重复完整证明移入附录，全稿只投一篇；`submit_checklist.md` 列为截稿前硬项 |
| 每篇须提名合格 reciprocal reviewer；≥4 篇者须实际参与评审 | 单一作者/小团队尤其脆弱 | `submit_checklist.md` 列为截稿前 4 周硬项，不排到截稿后 |
| desk reject 含幻觉引用、格式、破坏匿名、摘要与全文差异 | References 曾有 venue 缺失与自证句；曾出现对审稿人喊话 | v7 已统一 venue、删自证句、元话术清零；本轮 `p0_scan.md` 复核为 0 |
| 作者侧生成式 AI 政策逐年变化 | 2027 版随 CFP 发布 | 不写「AI 未产生任何内容」类绝对句；`submit_checklist.md` 挂起 |
| 双盲 | 附录曾出现源码路径、结果文件名 | 本轮拆分后 `paper_main.pdf` / `paper_appendix.pdf` 自识别路径扫描 **0** |

## 4. 委托人「不主张清单」（本轮沿用，全文强制）

- 不作语言模型 parity / perplexity / 生成可读性主张
- 不作与 attention 的等价、优越、低劣或表达力优势主张
- 不作吞吐绝对值、SOTA、通用自进化主张
- 不作任何录用概率承诺

## 5. Phase 0 判定

`cfp_published = false` ⇒ **允许继续 Phase 1–6；Phase 7（`submit_checklist.md`）挂起**。

交付时必须原样附这句话：

> 页数、披露、互审条款以当年 CFP 为准；本轮按已公布信息 + 往年惯例预排，不构成合规承诺。2027 官方 `.sty` 发布后需替换模板并复核页数、LLM 使用披露与 reciprocal reviewer 提名。
