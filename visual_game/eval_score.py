#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reusable evaluation utilities (DocVQA-style):
- ANLS (Average Normalized Levenshtein Similarity) with thresholding
- Numeric tolerance (percent scaling + rel_tol)
- Dataset-level F1 treating "Not answerable" as negative (as in the provided reference code)

This file is intended to be imported by different tasks (e.g., Jericho QA, visual game QA).
"""

import re
from math import isclose
from collections import defaultdict
from typing import List



def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def anls_compute(groundtruth, prediction, threshold=0.5):
    dist = levenshtein_distance(groundtruth, prediction)
    length = max(len(groundtruth.upper()), len(prediction.upper()))
    value = 0.0 if length == 0 else float(dist) / float(length)
    anls = 1.0 - value
    if anls<=threshold:
        anls = 0.0
    return anls


def is_float_equal(reference, prediction, include_percentage: bool = False, is_close: float = False) -> bool:
    def get_precision(gt_ans: float) -> int:
        precision = 3
        if '.' in str(gt_ans):
            precision = len(str(gt_ans).split('.')[-1])
        return precision

    reference = float(str(reference).strip().rstrip("%").strip())
    try:
        prediction = float(str(prediction).strip().rstrip("%").strip())
    except:
        return False

    if include_percentage:
        gt_result = [reference / 100, reference, reference * 100]
    else:
        gt_result = [reference]
    for item in gt_result:
        try:
            if is_close:
                if isclose(item, prediction, rel_tol=0.01):
                    return True
            precision = max(min(get_precision(prediction), get_precision(item)), 2)
            if round(prediction, precision) == round(item, precision):
                return True
        except Exception:
            continue
    return False


def get_clean_string(s):
    s = str(s).lower().strip()
    if s.endswith("mile"):
        s.rstrip("mile").strip()
    if s.endswith("miles"):
        s.rstrip("miles").strip()
    if s.endswith("million"):
        s.rstrip("million").strip()
    # remove parenthesis
    s = re.sub(r'\s*\([^)]*\)', '', s).strip()
    # remove quotes
    s = re.sub(r"^['\"]|['\"]$", "", s).strip()
    s = s.strip().lstrip("$").strip()
    s = s.strip().rstrip("%").strip()
    return s


def is_exact_match(s):
    flag = False
    # Website
    if "https://" in s:
        flag = True
    # code file
    if s.endswith(".py") or s.endswith("ipynb"):
        flag = True
    if s.startswith("page"):
        flag = True
    # telephone number
    if re.fullmatch(r'\b\d+(-\d+|\s\d+)?\b', s):
        flag = True
    # time
    if "a.m." in s or "p.m." in s:
        flag = True
    # YYYY-MM-DD
    if re.fullmatch(r'\b\d{4}[-\s]\d{2}[-\s]\d{2}\b', s):
        flag = True
    # YYYY-MM
    if re.fullmatch(r'\b\d{4}[-\s]\d{2}\b', s):
        flag = True
    # Email address
    if re.fullmatch(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', s):
        flag = True
    return flag


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def eval_score(gt, pred, answer_type):
    import ast
    import re

    # Keep original behavior for scalar types
    if answer_type == "Int":
        def to_int_like(x) -> int:
            # 允许 "1", "1.0", " 1.00 ", 以及数值类型
            s = get_clean_string("" if x is None else str(x))
            s = str(s).strip().rstrip("%").strip()
            v = float(s)  # "1.0" -> 1.0
            if abs(v - round(v)) < 1e-6:
                return int(round(v))
            return int(v)
        try:
            gt_i = to_int_like(gt)
            pred_i = to_int_like(pred)
            score = (gt_i == pred_i)
        except Exception:
            score = 0.0
    elif answer_type == "Float":
        try:
            gt = float(get_clean_string(str(gt)))
            pred = float(get_clean_string(str(pred)))
        except Exception:
            pred = ""
        score = is_float_equal(gt, pred, include_percentage=True, is_close=True)

    elif answer_type in ["Str", "None"]:
        gt = get_clean_string(gt)
        pred = get_clean_string(pred)
        if is_exact_match(gt):
            score = (gt == pred)
        else:
            score = anls_compute(gt, pred)

    else:
        # -------- list-like answers with "max over views" --------

        def to_list(x):
            """Convert x to list. If it's a string like '[...]', parse it."""
            if isinstance(x, str) and x.strip().startswith("["):
                try:
                    y = ast.literal_eval(x)
                    x = y
                except Exception:
                    pass
            return x if isinstance(x, list) else [x]

        def clean_item(x) -> str:
            return get_clean_string("" if x is None else str(x))

        def atomic_score(g_item: str, p_item: str) -> float:
            """Score two atomic strings using your existing rules."""
            g_item = clean_item(g_item)
            p_item = clean_item(p_item)
            if not g_item and not p_item:
                return 1.0
            # number-like: float equality
            if isfloat(g_item) and isfloat(p_item):
                return 1.0 if is_float_equal(g_item, p_item, include_percentage=True, is_close=True) else 0.0
            # exact-match-like: exact
            if is_exact_match(g_item):
                return 1.0 if g_item == p_item else 0.0
            # default: ANLS similarity
            return float(anls_compute(g_item, p_item))

        def join_variants(items: List[str]) -> List[str]:
            """List->string variants. Include comma+space because your example uses it."""
            items = [clean_item(x) for x in items if clean_item(x)]
            if not items:
                return [""]
            return [
                ", ".join(items),
                ",".join(items),
                "\n".join(items),
                " ".join(items),
                " and ".join(items),
            ]

        def split_pred_compound(pred_str: str) -> List[str]:
            """
            Turn 'a, b' or 'a\nb' or 'a and b' into ['a','b'] (best-effort).
            """
            s = (pred_str or "").strip()
            if not s:
                return []
            # Split on comma, newline, semicolon, or ' and '
            parts = re.split(r"\s*(?:,|\n|;|\band\b)\s*", s, flags=re.IGNORECASE)
            parts = [clean_item(p) for p in parts if clean_item(p)]
            # Dedup but preserve order
            seen = set()
            out = []
            for p in parts:
                k = p.lower()
                if k not in seen:
                    seen.add(k)
                    out.append(p)
            return out

        gt_list = to_list(gt)
        pred_list = to_list(pred)

        gt_clean = [clean_item(x) for x in gt_list if clean_item(x)]
        pred_clean_list = [clean_item(x) for x in pred_list if clean_item(x)]

        # Handle empty
        if len(gt_clean) == 0 and len(pred_clean_list) == 0:
            return 1.0
        if len(gt_clean) == 0 or len(pred_clean_list) == 0:
            return 0.0

        # We will compute multiple candidate scores and take the max.
        candidates: List[float] = []

        # --- Candidate A: original strict list-vs-list logic (only when lengths match)
        if len(gt_clean) == len(pred_clean_list):
            gt_sorted = sorted(gt_clean)
            pred_sorted = sorted(pred_clean_list)
            if isfloat(gt_sorted[0]) or is_exact_match(gt_sorted[0]):
                candidates.append(1.0 if ("-".join(gt_sorted) == "-".join(pred_sorted)) else 0.0)
            else:
                candidates.append(min([anls_compute(g, p) for g, p in zip(gt_sorted, pred_sorted)]))

        # --- Candidate B: pred as a single item vs each gt item -> max
        if len(pred_clean_list) == 1:
            p0 = pred_clean_list[0]
            candidates.append(max(atomic_score(g, p0) for g in gt_clean))

        # (Optional symmetric) gt single vs each pred item -> max
        if len(gt_clean) == 1:
            g0 = gt_clean[0]
            candidates.append(max(atomic_score(g0, p) for p in pred_clean_list))

        # --- Candidate C: list->string join variants vs pred (string)
        # Use pred as a string (if pred is list, also try its joined forms)
        pred_as_strs: List[str] = []
        if len(pred_clean_list) == 1:
            pred_as_strs.append(pred_clean_list[0])
        else:
            pred_as_strs.extend(join_variants(pred_clean_list))

        for pstr in pred_as_strs:
            for g_join in join_variants(gt_clean):
                candidates.append(atomic_score(g_join, pstr))

        # --- Candidate D: pred compound string (e.g., "a, b") -> split and compare as set
        # If pred is a single string, try splitting it into multiple items
        if len(pred_clean_list) == 1:
            parts = split_pred_compound(pred_clean_list[0])
            if len(parts) >= 2:
                # Exact set match => full credit
                if set([x.lower() for x in parts]) == set([x.lower() for x in gt_clean]):
                    candidates.append(1.0)
                # If same length, also try original list matching on the split parts
                if len(parts) == len(gt_clean):
                    gt_sorted = sorted(gt_clean)
                    parts_sorted = sorted(parts)
                    if isfloat(gt_sorted[0]) or is_exact_match(gt_sorted[0]):
                        candidates.append(1.0 if ("-".join(gt_sorted) == "-".join(parts_sorted)) else 0.0)
                    else:
                        candidates.append(min([anls_compute(g, p) for g, p in zip(gt_sorted, parts_sorted)]))

                # Also: part-vs-item max (if user answered one of them)
                candidates.append(max(atomic_score(g, parts[0]) for g in gt_clean))

        # Final: take max score among all views
        score = max(candidates) if candidates else 0.0

    return float(score)


def eval_acc_and_f1(samples):
    evaluated_samples = [sample for sample in samples if "score" in sample]
    if not evaluated_samples:
        return 0.0, 0.0

    acc = sum([sample["score"] for sample in evaluated_samples])/len(evaluated_samples)
    try:
        recall = sum([sample["score"] for sample in evaluated_samples if sample["answer"]!="Not answerable"])/len([sample for sample in evaluated_samples if sample["answer"]!="Not answerable"])
        precision = sum([sample["score"] for sample in evaluated_samples if sample["answer"]!="Not answerable"])/len([sample for sample in evaluated_samples if sample["pred"]!="Not answerable"])
        f1 = 2*recall*precision/(recall+precision) if (recall+precision)>0.0 else 0.0
    except:
        f1 = 0.0

    return acc, f1


def show_results(samples, show_path=None):
    for sample in samples:
        sample["evidence_pages"] = eval(sample["evidence_pages"])
        sample["evidence_sources"] = eval(sample["evidence_sources"])

    with open(show_path, 'w') as f:
        acc, f1 = eval_acc_and_f1(samples)
        f.write("Overall Acc: {} | Question Number: {}\n".format(acc, len(samples)))
        f.write("Overall F1-score: {} | Question Number: {}\n".format(f1, len(samples)))
        f.write("-----------------------\n")

        #####################
        acc_single_page, _ = eval_acc_and_f1([sample for sample in samples if len(sample["evidence_pages"])==1])
        acc_multi_page, _ = eval_acc_and_f1([sample for sample in samples if len(sample["evidence_pages"])!=1 and sample["answer"]!="Not answerable"])
        acc_neg, _ = eval_acc_and_f1([sample for sample in samples if sample["answer"]=="Not answerable"])

        f.write("Single-page | Accuracy: {} | Question Number: {}\n".format(
            acc_single_page, len([sample for sample in samples if len(sample["evidence_pages"])==1])
        ))
        f.write("Cross-page | Accuracy: {} | Question Number: {}\n".format(
            acc_multi_page, len([sample for sample in samples if len(sample["evidence_pages"])!=1 and sample["answer"]!="Not answerable"])
        ))
        f.write("Unanswerable | Accuracy: {} | Question Number: {}\n".format(
            acc_neg, len([sample for sample in samples if sample["answer"]=="Not answerable"])
        ))
        f.write("-----------------------\n")

        #####################
        source_sample_dict, document_type_dict = defaultdict(list), defaultdict(list)
        for sample in samples:
            for answer_source in sample["evidence_sources"]:
                source_sample_dict[answer_source].append(sample)
            document_type_dict[sample["doc_type"]].append(sample)
        for type, sub_samples in source_sample_dict.items():
            f.write(
                "Evidence Sources: {} | Accuracy: {} | Question Number: {}\n".format(type, eval_acc_and_f1(sub_samples)[0], len(sub_samples))
            )

        f.write("-----------------------\n")
        for type, sub_samples in document_type_dict.items():
            f.write(
                "Document Type: {} | Accuracy: {} | Question Number: {}\n".format(type, eval_acc_and_f1(sub_samples)[0], len(sub_samples))
            )


# --- Optional helper for tasks that don't provide answer_type explicitly ---
def infer_answer_type(gt, pred=None):
    """
    Best-effort inference for answer_type used by eval_score().
    This does NOT change eval_score logic; it only picks which branch to use.
    """
    if isinstance(gt, list):
        return "List"
    if isinstance(gt, str) and gt.strip().startswith("["):
        return "List"

    gt_s = get_clean_string(gt)
    if isfloat(gt_s):
        try:
            v = float(gt_s)
            if v.is_integer():
                return "Int"
            return "Float"
        except:
            return "Float"
    return "Str"


def canonicalize_not_answerable(s):
    """
    Normalize variants like 'not answerable' to the canonical label 'Not answerable',
    because eval_acc_and_f1 checks that string exactly.
    """
    cs = get_clean_string(s)
    if cs == "not answerable":
        return "Not answerable"
    return s
