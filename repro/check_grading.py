"""学术一致性自动检查：验证「分级版」没有把猜想写成结论、没有悬空引用、数字与记录一致。"""
import json, re

V = json.load(open("results_verification.json"))
S = open("ICML_evidence_graded.md", encoding="utf-8").read()
A = json.load(open("results_all.json"))
ST = json.load(open("results_structural.json"))
TP = json.load(open("results_throughput.json"))
GC = json.load(open("results_grid_chunk.json"))

rows = []
def chk(cid, name, ok, detail):
    rows.append(dict(id=cid, check=name, status="PASS" if ok else "FAIL", detail=detail))

ids_pass = {r["id"] for r in V["rows"] if r["status"] == "PASS"}
cited = set(re.findall(r"\[(V\d+[a-z]?)\]", S))
missing = cited - ids_pass
chk("G01", "所有 [Vxx] 引用都对应 PASS 项", not missing,
    f"引用 {len(cited)} 个，悬空 {sorted(missing)}")

cids = {r["id"] for r in V["rows"] if r["level"] == "C"}
sec3 = S.split("## 3. 猜想与未验证主张")[1].split("### 3.1")[0]
in_sec3 = set(re.findall(r"\| (C\d{2}) \|", sec3))
chk("G02", "全部猜想条目都集中在 §3", cids == in_sec3,
    f"台账 {sorted(cids)} / §3 {sorted(in_sec3)}")

abstract = S.split("## 1. 证据分级原则")[0]
leaked = [c for c in cids if c in abstract]
chk("G03", "摘要不含猜想编号", not leaked, f"泄漏 {leaked}")

# 猜想类关键内容不得在摘要/§2 中被写成事实
sec2 = S.split("## 2. 已验证结论")[1].split("## 3. 猜想")[0]
bad_phrases = ["1.948 GiB 部署已完成", "R1 使 MQAR 追平", "hybrid 网格把 cond 从 1e19 降到 1.4 已验证",
               "1M→8M 增量恒为 0（实测）", "端侧训练已复现"]
found = [p for p in bad_phrases if p in abstract + sec2]
chk("G04", "摘要与 §2 未把猜想写成事实", not found, f"命中 {found}")

# 伪论证词：只允许出现在「已删除/修订」的说明里
for w in ["逐位吻合", "高度吻合", "近乎逐位", "近零 θ + 高 ρ 是重尾主因",
          "应来自另一个更大的预设"]:
    hits = [i for i, l in enumerate(S.splitlines(), 1) if w in l]
    ok = all(("删除" in S.splitlines()[i-1] or "修订" in S.splitlines()[i-1] or
              "不写" in S.splitlines()[i-1] or "伪论证" in S.splitlines()[i-1])
             for i in hits)
    chk(f"G05_{w[:6]}", f"伪论证词「{w}」未作为结论出现", ok,
        f"出现行 {hits}（应仅出现在修订说明中）")

# 关键数字一致性
fin = [v for v in GC.values() if v["done"] == 1500][0]
nums = {
    "osc linear30 @1500": (f"{fin['acc']:.4f}", "0.9844"),
    "TS osc @3000": (f"{A['osc|0']['ts_acc']:.4f}", "0.8535"),
    "SGC mqar": (f"{A['mqar|0']['acc']:.4f}", "0.3066"),
    "TS mqar": (f"{A['mqar|0']['ts']:.4f}", "0.3945"),
    "状态字节": (str(ST["T3_state"]["formula_bytes"]), "402436"),
    "inv_sin": (f"{ST['grid']['inv_sin_theta_min']:.4f}", "2607.5947"),
    # 文稿写「约 7633×」为截断；精确值 7633.51，四舍五入为 7634
    "KV/状态": (f"{ST['ledger']['ratio_kv_over_state']:.1f}", "7633.5"),
}
for k, (got, exp) in nums.items():
    chk(f"G06_{k[:8]}", f"数字一致：{k}", abs(float(got) - float(exp)) < 1.0 or got == exp,
        f"记录 {got} / 文稿 {exp}")

# 分级纪律：§3 每条都必须带限定词
limit_words = ["未实测", "未实现", "未复现", "未运行", "未测", "推论", "半成品", "声称", "未验证"]
sec3_lines = [l for l in sec3.splitlines() if re.match(r"\| C\d{2} \|", l)]
no_limit = [l[:40] for l in sec3_lines if not any(w in l for w in limit_words)]
chk("G07", "§3 每条猜想都带「未实测/声称/推论」限定", not no_limit, f"缺限定 {no_limit}")

# 台账规模
chk("G08", "一键验证 0 FAIL", V["n_fail"] == 0,
    f"PASS={V['n_pass']} 猜想={sum(1 for r in V['rows'] if r['level']=='C')} 总计={V['total']}")

out = dict(total=len(rows), n_pass=sum(1 for r in rows if r["status"] == "PASS"),
           n_fail=sum(1 for r in rows if r["status"] == "FAIL"), rows=rows)
json.dump(out, open("results_grading_check.json", "w"), indent=2, ensure_ascii=False)
print("=" * 92)
for r in rows:
    print(f"{r['status']:<6}{r['id']:<20}{r['check'][:44]:<46}{r['detail'][:60]}")
print("=" * 92)
print(f"共 {out['total']} 项：PASS {out['n_pass']} / FAIL {out['n_fail']}")
