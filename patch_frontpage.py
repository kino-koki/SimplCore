p = "/data/workspace/build_simplcore_pdf_v5.py"
s = open(p, encoding="utf-8").read()


def rep(old, new, label):
    global s
    assert s.count(old) == 1, "%s: count=%d" % (label, s.count(old))
    s = s.replace(old, new)


def cut(start, end, new, label):
    """替换 [start, end) 区间；end 必须出现在 start 之后。"""
    global s
    i = s.index(start)
    j = s.index(end, i)
    assert i < j, label
    s = s[:i] + new + s[j:]


# 1) d.table 不解析 HTML 标签，<b> 会被原样画进 PDF
rep(
    '         "<b>THEOREM A (main increment): a TIGHT log2|A|-bit lower bound on INPUT-SIDE role information; a proof that Toeplitz duality FORCES those bits to the input; three-way carrier conservation; a RIGHT-ANGLE bit-depth frontier (no depth can substitute)."],',
    '         "THEOREM A (main increment): a TIGHT log2|A|-bit lower bound on INPUT-SIDE role information; a proof that Toeplitz duality FORCES those bits to the input; three-way carrier conservation; a RIGHT-ANGLE bit-depth frontier (no depth can substitute)."],',
    "b_tag",
)

# 2) §1.1 与 Table 1 / §1 引言重复，且其段落跨栏产生的孤立尾句跑到右栏顶部
cut(
    '    Hh("1.1  What is new, precisely", size=9.4, before=5.0, after=2.0)',
    '    Hh("1.2  Contributions", size=9.4, before=4.0, after=2.0)',
    '',
    "sec1_1",
)

# 3) 正文里的内部简称换成规范编号引用
rep(
    '      "fixed-state models (S4, S5, Zoology, Repeat After Me, Illusion of State; "\n'
    '      "see References). We do NOT re-prove those. The question we answer is which "',
    '      "fixed-state models (S4 [1], S5 [2], Zoology [13], Repeat After Me [14], "\n'
    '      "Illusion of State [15]). We do NOT re-prove those. The question we answer is which "',
    "intro_cites",
)

rep(
    '        ["Fixed-state models are weak at associative recall (Zoology 2312.04927; 2402.01032; 2404.08819)",',
    '        ["Fixed-state models are weak at associative recall (Zoology [13]; [14]; [15])",',
    "t1_row2",
)

rep(
    '        ["Selective/input-dependent parameters break the convolution view (Mamba 2312.00752)",',
    '        ["Selective/input-dependent parameters break the convolution view (Mamba [4])",',
    "t1_row3",
)

rep(
    '        ["Linear attention uses S = sum k_s v_s^T (2006.16236; GLA 2312.06635)",',
    '        ["Linear attention uses S = sum k_s v_s^T ([8]; GLA [6])",',
    "t1_row4",
)

rep(
    '        ["Fixed-state models are weak at recall; input format can carry task information (Zoology [13]; 2402.01032; 2404.08819)",',
    '        ["Fixed-state models are weak at recall; input format can carry task information (Zoology [13]; [14]; [15])",',
    "t1_row6",
)

open(p, "w", encoding="utf-8").write(s)
print("frontpage patch OK")
