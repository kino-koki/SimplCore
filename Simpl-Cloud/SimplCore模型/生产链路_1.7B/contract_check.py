#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Static contract check for the 1.7B production torch path.

No torch in the sandbox, so ``modeling_simplcore_v7.py`` cannot be executed
here.  Instead we parse its AST, expand every nn.Module / nn.Parameter into the
exact state_dict key templates it MUST produce, and compare them with the keys
emitted by ``migrate_qwen3.py`` (also AST-derived).  A mismatch means the
student checkpoint would silently load random weights -- exactly the failure
mode this project has been guarding against.
"""
from __future__ import annotations

import ast
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_SRC = os.path.join(HERE, "modeling_simplcore_v7.py")
MIG_SRC = os.path.join(HERE, "migrate_qwen3.py")
TRAIN_SRC = os.path.join(HERE, "train_simplcore.py")
VERIFY_SRC = os.path.join(HERE, "verify_arch.py")


def class_attrs(tree):
    """{class_name: [(attr, source_expr), ...]} for self.X = ... assignments."""
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        attrs = []
        seen_attr = set()
        # self.X = ... lives inside __init__, not in the class body.
        for st in ast.walk(node):
            if not isinstance(st, ast.Assign):
                continue
            val = st.value
            for tg in st.targets:
                if (isinstance(tg, ast.Attribute) and isinstance(tg.value, ast.Name)
                        and tg.value.id == "self" and tg.attr not in seen_attr):
                    seen_attr.add(tg.attr)
                    attrs.append((tg.attr, ast.unparse(val) if val else ""))
        out[node.name] = attrs
    return out


def classify(expr):
    for t in ("Linear", "Embedding", "Parameter", "ModuleList"):
        if expr.startswith("nn." + t):
            return t
    for t in ("RMSNorm", "QwenSwiGLU", "SimplCoreV7Model"):
        if expr.startswith(t):
            return t
    return "other"


def expand(cls, prefix, attrs, seen=None, depth=0):
    """Yield expected state_dict key templates; {i} is the layer index."""
    seen = seen or frozenset()
    if depth > 4 or cls in seen:
        return
    seen = seen | {cls}
    for name, expr in attrs.get(cls, []):
        t = classify(expr)
        if t in ("Linear", "Embedding", "RMSNorm"):
            yield prefix + name + ".weight"
        elif t == "Parameter":
            yield prefix + name
        elif t == "QwenSwiGLU":
            for sub in ("gate", "up", "down"):
                yield prefix + name + "." + sub + ".weight"
        elif t == "SimplCoreV7Model":
            yield from expand("SimplCoreV7Model", prefix + name + ".",
                              attrs, seen, depth + 1)
        elif t == "ModuleList":
            m = re.search(r"\b(V7Block|SimplCoreBlock)\b", expr)
            if m:
                yield from expand(m.group(1), prefix + name + ".{i}.",
                                  attrs, seen, depth + 1)


def str_of(node, env):
    """Fold the string expression used as put()'s first argument."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id, "{" + node.id + "}")
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return str_of(node.left, env) + str_of(node.right, env)
    if isinstance(node, ast.JoinedStr):
        s = ""
        for v in node.values:
            if isinstance(v, ast.Constant):
                s += str(v.value)
            elif isinstance(v, ast.FormattedValue):
                s += "{" + str_of(v.value, env).strip("{}") + "}"
        return s
    return "{" + ast.unparse(node) + "}"


def migrate_keys(tree):
    """Recover put() key templates, expanding the ffn.{ours} loop."""
    env, raw = {}, []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tg in node.targets:
                if isinstance(tg, ast.Name) and isinstance(node.value, ast.JoinedStr):
                    env[tg.id] = str_of(node.value, env)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "put" and node.args):
            raw.append(str_of(node.args[0], env))
    out = []
    for k in raw:
        if "{ours}" in k:
            for o in ("gate", "up", "down"):
                out.append(k.replace("{ours}", o))
        else:
            out.append(k)
    return out


def tuple_constants(src, name):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tg in node.targets:
                if isinstance(tg, ast.Name) and tg.id == name:
                    if isinstance(node.value, (ast.Tuple, ast.List)):
                        return [ast.literal_eval(e) for e in node.value.elts]
    return []


def main():
    mt = ast.parse(open(MODEL_SRC, encoding="utf-8").read())
    gt = ast.parse(open(MIG_SRC, encoding="utf-8").read())
    attrs = class_attrs(mt)
    expected = sorted(set(expand("SimplCoreV7ForCausalLM", "", attrs)))
    emitted = sorted(set(migrate_keys(gt)))

    missing = sorted(set(expected) - set(emitted))
    unexpected = sorted(set(emitted) - set(expected))
    tied = "lm_head.weight" in expected and "lm_head.weight" not in emitted
    real_missing = [k for k in missing if k != "lm_head.weight"]

    train_keys = tuple_constants(open(TRAIN_SRC, encoding="utf-8").read(),
                                 "TRAIN_KEYS")
    train_hits = sorted(k for k in expected
                        if any(s in k for s in train_keys))
    sgc_sel = sorted(set(
        e.value for node in ast.walk(ast.parse(
            open(VERIFY_SRC, encoding="utf-8").read()))
        if isinstance(node, ast.Tuple) for e in node.elts
        if isinstance(e, ast.Constant) and isinstance(e.value, str)
        and e.value in ("Wk", "Wv", "Wq", "Wc", "Wg", "wgate", "alpha")))
    sgc_hits = sorted(k for k in expected if any(s in k for s in sgc_sel))

    new_shapes = []
    for node in ast.walk(gt):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("zeros", "tensor")):
            new_shapes.append(ast.unparse(node))

    report = dict(
        model_source=os.path.basename(MODEL_SRC),
        expected_keys=expected, emitted_keys=emitted,
        n_expected=len(expected), n_emitted=len(emitted),
        missing_from_migration=missing,
        legitimate_tied_key_excluded=bool(tied),
        unexpected_in_migration=unexpected,
        train_keys=train_keys, train_key_hits=train_hits,
        sgc_perturbation_selectors=sgc_sel, sgc_perturbation_hits=sgc_hits,
        explicit_new_tensor_exprs=sorted(set(new_shapes))[:20],
    )
    report["passed"] = bool(not real_missing and not unexpected and tied
                            and len(train_hits) > 0 and len(sgc_hits) > 0)
    json.dump(report, open(os.path.join(HERE, "contract_check.json"), "w"),
              indent=2)
    print("[contract] expected=%d emitted=%d missing=%s unexpected=%s tied=%s "
          "train_hits=%d sgc_hits=%d PASS=%s"
          % (len(expected), len(emitted), real_missing, unexpected, tied,
             len(train_hits), len(sgc_hits), report["passed"]))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
