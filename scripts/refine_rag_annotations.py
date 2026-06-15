from __future__ import annotations

import argparse
import json
import re
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ANNOTATIONS_PATH = Path("data/eval/real_user_questions_rag_annotations.jsonl")
SUMMARY_PATH = Path("data/eval/real_user_questions_rag_annotations.summary.json")
DOC_PATH = Path("docs/26-真实用户问题测试集RAG标注.md")
METADATA_ROOT = Path("data/raw/arxiv/metadata")
SOURCE_ROOT = Path("data/raw/arxiv/source")
CHINESE_REFERENCE_OVERRIDES_PATH = Path("data/eval/chinese_rag_reference_overrides.json")
ANNOTATION_VERSION = "rag_eval_v2_factual_reference_annotations_2026-04-28"
ANNOTATION_BASIS = "factual_reference_from_metadata_and_latex_sections"

ARXIV_RE = re.compile(r"\b\d{4}\.\d{5}(?:v\d+)?\b", re.IGNORECASE)
LATEX_SECTION_RE = re.compile(
    r"\\(?P<level>section|subsection|subsubsection)\*?\{(?P<title>[^{}]+)\}",
    re.IGNORECASE,
)

METHOD_KEYS = (
    "method",
    "methodology",
    "approach",
    "proposed",
    "model",
    "framework",
    "architecture",
    "algorithm",
    "implementation",
    "formulation",
    "analysis",
    "方法",
)
EXPERIMENT_KEYS = (
    "experiment",
    "evaluation",
    "result",
    "dataset",
    "comparison",
    "ablation",
    "benchmark",
    "case study",
    "实验",
    "结果",
)
LIMIT_KEYS = (
    "limitation",
    "discussion",
    "conclusion",
    "future",
    "scope",
    "drawback",
    "threat",
    "appendix",
    "局限",
    "结论",
)

CHINESE_REFERENCE_TERM_REPLACEMENTS = (
    ("AI meeting summaries", "AI 会议摘要"),
    ("Dataset Pipeline", "数据集流水线"),
    ("artifact package", "工件包"),
    ("source intake", "源数据接入"),
    ("structured reference construction", "结构化参考答案构建"),
    ("candidate generation", "候选生成"),
    ("structured scoring", "结构化评分"),
    ("reporting", "报告生成"),
    ("claim scorers", "声明评分器"),
    ("ground truth", "标准答案"),
    ("evaluator outputs", "评估器输出"),
    ("artifacts", "工件"),
    ("aggregation", "聚合"),
    ("issue analysis", "问题分析"),
    ("statistical testing", "统计检验"),
    ("meetings", "会议"),
    ("meeting-model pairs", "会议-模型对"),
    ("judge runs", "评审运行"),
    ("mean accuracy", "平均准确率"),
    ("state-of-the-art baselines", "最先进基线"),
    ("state-of-the-art", "最先进"),
    ("baselines", "基线"),
    ("baseline", "基线"),
    ("benchmark", "基准"),
    ("Large Language Models", "大语言模型"),
    ("large language models", "大语言模型"),
    ("language model", "语言模型"),
    ("Knowledge Graph Completion", "知识图谱补全"),
    ("LLM-generated text detection", "LLM 生成文本检测"),
    ("LLM-generated text", "LLM 生成文本"),
    ("Implicit Reward Models", "隐式奖励模型"),
    ("implicit reward model", "隐式奖励模型"),
    ("zero-shot detection method", "零样本检测方法"),
    ("zero-shot", "零样本"),
    ("instruction-tuned models", "指令微调模型"),
    ("base models", "基础模型"),
    ("preference construction", "偏好构造"),
    ("preference collection", "偏好收集"),
    ("task-specific fine-tuning", "任务特定微调"),
    ("reward-based method", "基于奖励的方法"),
    ("evaluation metrics", "评估指标"),
    ("detection methods", "检测方法"),
    ("supervised methods", "监督方法"),
    ("domains", "领域"),
    ("arXiv dataset", "arXiv 数据集"),
    ("Review dataset", "Review 数据集"),
    ("LLM tokens", "LLM 词元"),
    ("KG entities", "KG 实体"),
    ("entity representations", "实体表示"),
    ("coarse-to-fine", "从粗到细"),
    ("codebook", "码本"),
    ("dense embeddings", "稠密嵌入"),
    ("hierarchical clustering", "层次聚类"),
    ("semantic tree", "语义树"),
    ("residual quantization", "残差量化"),
    ("Embedding-based methods", "基于嵌入的方法"),
    ("Granular Semantic Enhancement", "粒度语义增强"),
    ("Generative Structural Reconstruction", "生成式结构重建"),
    ("embeddings", "嵌入"),
    ("tokens", "词元"),
    ("codes", "编码"),
    ("tree", "树"),
    ("memory retrieval policy", "记忆检索策略"),
    ("operator agent", "操作员智能体"),
    ("agents", "智能体"),
    ("agent", "智能体"),
    ("operator 智能体", "操作员智能体"),
    ("episodes", "轮次"),
    ("episode", "轮次"),
    ("reflection", "反思"),
    ("memory", "记忆"),
    ("tools", "工具"),
    ("planner", "规划器"),
    ("predictions", "预测"),
    ("learning", "学习"),
    ("decision prompt", "决策提示"),
    ("prompt", "提示"),
    ("component ablation", "组件消融"),
    ("ablation", "消融"),
    ("performance", "性能"),
    ("planner evolution", "规划器演化"),
    ("per-tool selection", "按工具选择"),
    ("cold-start", "冷启动"),
    ("credit method", "归因方法"),
    ("uniform credit", "均匀归因"),
    ("backbone LLM", "骨干 LLM"),
    ("temperature", "温度"),
    ("portfolio mode", "组合模式"),
    ("tool selection", "工具选择"),
    ("tool outputs", "工具输出"),
    ("portfolio weights", "组合权重"),
    ("warm-up period", "预热期"),
    ("multi-agent latent trajectories", "多智能体潜在轨迹"),
    ("research questions", "研究问题"),
    ("multi-agent systems", "多智能体系统"),
    ("multi-智能体 latent trajectories", "多智能体潜在轨迹"),
    ("multi-智能体 systems", "多智能体系统"),
    ("key-value caches", "键值缓存"),
    ("latent communication", "潜在通信"),
    ("text-based protocols", "文本协议"),
    ("multi-agent latent trajectories", "多智能体潜在轨迹"),
    ("parameter-efficient supervised training", "参数高效监督训练"),
    ("mathematical reasoning", "数学推理"),
    ("scientific QA", "科学问答"),
    ("code generation", "代码生成"),
    ("commonsense benchmarks", "常识基准"),
    ("self-consistency analysis on", "自一致性分析："),
    ("Decoding Stability Analysis", "解码稳定性分析"),
    ("token-level perplexity", "词元级困惑度"),
    ("perplexity", "困惑度"),
    ("decoding stability", "解码稳定性"),
    ("decoding", "解码"),
    ("calibration", "校准"),
    ("self-consistency analysis", "自一致性分析"),
    ("Self-consistency analysis", "自一致性分析"),
    ("Accuracy", "准确率"),
    ("Tables", "表"),
    ("top", "上方"),
    ("bottom", "下方"),
    ("batch size", "批大小"),
    ("natural image SR", "自然图像超分辨率"),
    ("text SR", "文本超分辨率"),
    ("low-light", "低光照增强"),
    ("Remote Sensing Infrared Image Super-Resolution", "遥感红外图像超分辨率"),
    ("bicubic downsampling", "双三次下采样"),
    ("encoder-decoder", "编码器-解码器"),
    ("ethnicity", "族裔"),
    ("accent", "口音"),
    ("gender", "性别"),
    ("age", "年龄"),
    ("first language", "第一语言"),
    ("clique", "团"),
    ("independent set", "独立集"),
    ("SAT solving", "SAT 求解"),
    ("Ramsey-good graphs", "Ramsey-good 图"),
    ("doubly saturated", "双重饱和"),
    ("computer-assisted", "计算机辅助"),
    ("mathematics", "数学"),
    ("graphs", "图"),
    ("experimental", "实验"),
    ("recall", "召回"),
    ("precision", "精确率"),
    ("state-tracking", "状态跟踪"),
    ("attention", "注意力"),
    ("recurrent state updates", "循环状态更新"),
    ("hybrid architectures", "混合架构"),
    ("foundation models", "基础模型"),
    ("embodiment", "具身形态"),
    ("kinematics", "运动学"),
    ("kinematic", "运动学"),
    ("bioactivity", "生物活性"),
    ("Markush structures", "Markush 结构"),
    ("chemical-structure-grounded", "化学结构约束"),
    ("extraction", "抽取"),
    ("description logic", "描述逻辑"),
    ("refined", "精化"),
    ("oracle-sensitive", "预言机敏感"),
    ("checkbox/slider", "复选框/滑块"),
    ("operator agent", "操作员智能体"),
    ("prefill", "预填充"),
    ("adjudicator agent", "裁决智能体"),
    ("adjudicator 智能体", "裁决智能体"),
    ("web adversary", "网络对手"),
    ("adversary", "对手"),
    ("headless crawlers", "无头爬虫"),
    ("automated", "自动化"),
    ("logo", "标志"),
    ("phishing URLs", "钓鱼 URL"),
    ("phishing URL", "钓鱼 URL"),
    ("live urls", "实时 URL"),
    ("snapshot-based", "基于快照"),
    ("checklist", "检查清单"),
    ("indicators of compromise", "失陷指标"),
    ("CAPTCHA challenges", "CAPTCHA 挑战"),
    ("behavioral puzzles", "行为谜题"),
    ("false negatives", "假阴性"),
    ("Thompson Sampling bandit", "汤普森采样多臂老虎机"),
    ("sequential portfolio", "顺序组合"),
    ("KV streaming", "KV 流式传输"),
    ("KV chunks", "KV 块"),
    ("schedules", "调度"),
    ("context loading", "上下文加载"),
    ("decoder layer", "解码器层"),
    ("attention operator", "注意力算子"),
    ("Transformer models", "Transformer 模型"),
    ("edit", "编辑"),
)


@dataclass(frozen=True)
class PaperEvidence:
    paper_id: str
    title: str
    abstract: str
    categories: list[str]
    source_extract: str = ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refine RAG annotations with factual references from local paper evidence."
    )
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--doc", type=Path, default=DOC_PATH)
    parser.add_argument("--metadata-root", type=Path, default=METADATA_ROOT)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument(
        "--chinese-reference-overrides",
        type=Path,
        default=CHINESE_REFERENCE_OVERRIDES_PATH,
    )
    args = parser.parse_args()

    annotations = load_jsonl(args.annotations)
    evidence = build_evidence_index(args.metadata_root, args.source_root)
    refined = refine_annotations(
        annotations,
        evidence,
        chinese_reference_overrides=load_json_mapping(args.chinese_reference_overrides),
    )
    write_jsonl(args.annotations, refined)
    summary = build_summary(refined, args.annotations)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.doc.write_text(render_markdown(refined, summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "annotations": str(args.annotations),
                "count": len(refined),
                "rag_references_refined": count_rag_references(refined),
                "summary": str(args.summary),
                "doc": str(args.doc),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_json_mapping(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return {str(key): str(value) for key, value in payload.items()}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def refine_annotations(
    annotations: list[dict[str, Any]],
    evidence: dict[str, PaperEvidence],
    *,
    chinese_reference_overrides: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    overrides = chinese_reference_overrides or {}
    return [
        build_refined_annotation(
            annotation,
            evidence,
            chinese_reference_overrides=overrides,
        )
        for annotation in annotations
    ]


def build_refined_annotation(
    annotation: dict[str, Any],
    evidence: dict[str, PaperEvidence],
    *,
    chinese_reference_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    refined = dict(annotation)
    tags = dict(refined.get("tags") or {})
    if not is_rag_retrieval_case(tags):
        return refined

    expected_source_ids = [str(item) for item in refined.get("expected_source_ids") or []]
    explicit_ids = [str(item) for item in tags.get("explicit_arxiv_ids") or []]
    source_type = str(tags.get("source_type") or "")
    input_quality = str(tags.get("input_quality") or "")

    papers = [evidence[paper_id] for paper_id in expected_source_ids if paper_id in evidence]
    if papers:
        override = (chinese_reference_overrides or {}).get(str(refined.get("case_id") or ""))
        if override and requires_chinese_answer(refined):
            refined["reference"] = normalize_chinese_reference_text(override)
        else:
            refined["reference"] = build_factual_reference(refined, papers)
        refined["required_points"] = build_factual_required_points(
            refined,
            papers,
            source_type=source_type,
            input_quality=input_quality,
        )
    else:
        ids = explicit_ids or extract_arxiv_ids(str(refined.get("user_input", "")))
        refined["reference"] = build_missing_corpus_reference(ids)
        refined["required_points"] = build_missing_corpus_points(ids)

    tags["annotation_version"] = ANNOTATION_VERSION
    tags["annotation_basis"] = ANNOTATION_BASIS
    refined["tags"] = tags
    return refined


def is_rag_retrieval_case(tags: dict[str, Any]) -> bool:
    tracks = tags.get("evaluation_tracks") or []
    return "rag_retrieval" in {str(track) for track in tracks}


def build_factual_reference(annotation: dict[str, Any], papers: list[PaperEvidence]) -> str:
    if requires_chinese_answer(annotation):
        return build_chinese_factual_reference(annotation, papers)

    segments: list[str] = []
    for paper in papers[:3]:
        identity = f"arXiv:{paper.paper_id}《{paper.title or paper.paper_id}》"
        categories = "、".join(paper.categories[:4])
        prefix = f"{identity}"
        if categories:
            prefix += f"（类别：{categories}）"
        prefix += "。"
        abstract = truncate(clean_sentence_block(paper.abstract), 900)
        part = prefix
        if abstract:
            part += f"摘要事实：{abstract}"
        if paper.source_extract:
            part += f" 正文事实：{truncate(paper.source_extract, 900)}"
        segments.append(part)

    return truncate(" ".join(segment for segment in segments if segment), 2600)


def requires_chinese_answer(annotation: dict[str, Any]) -> bool:
    text_parts = [
        str(annotation.get("user_input") or ""),
        " ".join(str(point) for point in annotation.get("required_points") or []),
    ]
    text = "\n".join(text_parts)
    return "中文" in text or "Chinese" in text


def normalize_chinese_reference_text(reference: str) -> str:
    parts = re.split(r"(《[^》]+》)", reference)
    return "".join(
        part
        if part.startswith("《") and part.endswith("》")
        else normalize_chinese_reference_segment(part)
        for part in parts
    )


def normalize_chinese_reference_segment(reference: str) -> str:
    normalized = reference
    for source, target in CHINESE_REFERENCE_TERM_REPLACEMENTS:
        if re.fullmatch(r"[A-Za-z][A-Za-z-]*", source):
            normalized = re.sub(rf"\b{re.escape(source)}\b", target, normalized)
        else:
            normalized = normalized.replace(source, target)
    normalized = re.sub(
        r"([\u4e00-\u9fffA-Za-z0-9][\u4e00-\u9fffA-Za-z0-9-]{1,40})（\1）",
        r"\1",
        normalized,
    )
    normalized = normalized.replace("困惑度（困惑度, PPL）", "困惑度（PPL）")
    normalized = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", normalized)
    return normalized


def build_chinese_factual_reference(
    annotation: dict[str, Any],
    papers: list[PaperEvidence],
) -> str:
    tags = annotation.get("tags") or {}
    source_type = str(tags.get("source_type") or "")
    segments: list[str] = []
    for paper in papers[:3]:
        identity = f"arXiv:{paper.paper_id}《{paper.title or paper.paper_id}》"
        categories = "、".join(paper.categories[:4])
        prefix = identity
        if categories:
            prefix += f"（类别：{categories}）"
        prefix += "。"
        segments.append(prefix + chinese_reference_body(source_type, paper))

    return truncate(" ".join(segment for segment in segments if segment), 2600)


def chinese_reference_body(source_type: str, paper: PaperEvidence) -> str:
    abstract = truncate(clean_sentence_block(paper.abstract), 900)
    source_extract = truncate(paper.source_extract, 900)
    parts: list[str] = []
    if source_type == "rag_eval":
        if abstract:
            parts.append(f"相关事实：{abstract}")
        if source_extract:
            parts.append(f"正文事实：{source_extract}")
        parts.append(chinese_rag_eval_answer(paper))
    else:
        if abstract:
            parts.append(f"方法主线：{abstract}")
        if source_extract:
            parts.append(f"正文证据：{source_extract}")
        parts.append(chinese_paper_qa_answer(paper))
    return "".join(parts)


def chinese_paper_qa_answer(paper: PaperEvidence) -> str:
    focus = chinese_section_focus(paper.source_extract)
    if focus:
        return f"正文阅读重点：{focus}。"
    return "正文证据状态：本地证据未提供正文片段，章节重点和复现细节未覆盖。"


def chinese_rag_eval_answer(paper: PaperEvidence) -> str:
    basis = "上述摘要事实和正文事实" if paper.source_extract else "上述摘要和元数据事实"
    return (
        f"RAG 判定答案：相关上下文为目标论文中能够支撑{basis}的片段；"
        f"噪声上下文为非目标论文片段、只匹配关键词但不能支撑{basis}的片段，"
        f"或与该 query 无关的片段；缺失上下文为没有覆盖{basis}中关键研究对象、"
        "方法、结果或边界信息的目标论文证据。"
    )


def chinese_section_focus(source_extract: str) -> str:
    labels: list[str] = []
    if "方法/理论依据：" in source_extract or "正文依据：" in source_extract:
        labels.append("方法或理论部分")
    if "实验/结果依据：" in source_extract:
        labels.append("实验或结果部分")
    if "边界/局限依据：" in source_extract:
        labels.append("讨论、结论或局限部分")
    return "、".join(labels)


def build_factual_required_points(
    annotation: dict[str, Any],
    papers: list[PaperEvidence],
    *,
    source_type: str,
    input_quality: str,
) -> list[str]:
    paper_labels = "、".join(
        f"arXiv:{paper.paper_id}《{paper.title or paper.paper_id}》" for paper in papers
    )
    has_body_evidence = any(paper.source_extract for paper in papers)
    points: list[str] = [
        f"准确识别并围绕目标论文：{paper_labels}",
        "使用目标论文摘要事实说明研究问题、对象或任务背景",
    ]
    if has_body_evidence:
        points.append("使用目标论文正文证据说明方法、理论思路、实验、结果或结论中的至少两类")
    else:
        points.append("使用目标论文摘要和元数据事实说明方法、任务、实验或结论线索")
    points.append("区分论文事实、基于证据的合理推断，以及当前检索上下文没有覆盖的信息")

    if source_type == "paper_qa":
        points.extend(
            [
                "直接回答用户指定维度，例如方法主线、贡献、局限、可复现性或是否 incremental",
                "指出适用边界、局限、复现风险或需要人工核对的信息",
            ]
        )
    elif source_type == "cross_domain":
        points.extend(
            [
                "把源论文机制映射到目标领域任务，并说明相似点和不匹配点",
                "提出可验证实验、baseline、消融或失败条件",
            ]
        )
    elif source_type == "literature_review":
        points.extend(
            [
                "以锚点论文为起点组织背景、方法分支、代表工作和评测证据",
                "总结开放问题或后续研究机会，避免只罗列关键词",
            ]
        )
    elif source_type == "trend_analysis":
        points.extend(
            [
                "结合库内论文的主题、类别、时间或方法证据给出趋势判断",
                "区分研究问题、方法路线、benchmark、数据集或评测变化",
            ]
        )
    elif source_type == "idea_novelty":
        points.extend(
            [
                "比较用户 idea 与目标论文/相近工作的重合点和差异点",
                "指出 novelty 风险、证据缺口和可发表性不确定性",
            ]
        )
    elif source_type == "paper_search":
        points.extend(
            [
                "返回标题、摘要、章节线索、匹配理由或相关片段",
                "说明哪些章节最值得先读，以及证据不足时如何调整检索",
            ]
        )
    elif source_type == "rag_eval":
        points.extend(
            [
                "定义哪些 retrieved contexts 算相关、噪声或缺失",
                "说明 Context Precision、Context Recall、Completeness 或失败样例判定依据",
            ]
        )
    elif source_type == "study_plan":
        points.extend(
            [
                "把目标论文放入学习路径中的合适位置",
                "给出阶段安排、检查点和应产出的笔记/复现/评测结果",
            ]
        )
    elif source_type == "multi_turn_followup" and input_quality == "good":
        points.extend(
            [
                "基于检索证据判断是否适合组会、复现或继续分析",
                "保留目标论文、筛选标准和后续追问可引用的上下文",
            ]
        )

    points.extend(common_user_constraints(str(annotation.get("user_input") or "")))
    return dedupe(points)


def build_missing_corpus_reference(arxiv_ids: list[str]) -> str:
    label = (
        "、".join(f"arXiv:{paper_id}" for paper_id in arxiv_ids)
        if arxiv_ids
        else "用户点名论文"
    )
    return (
        f"当前库内没有可验证论文证据覆盖{label}。事实型 reference 只记录库内证据不足："
        "不能确认该论文的作者、摘要、方法、实验或结论，不能用相似论文替代目标论文；"
        "下一步是导入论文、调整关键词、放宽范围或人工核对。"
    )


def build_missing_corpus_points(arxiv_ids: list[str]) -> list[str]:
    label = (
        "、".join(f"arXiv:{paper_id}" for paper_id in arxiv_ids)
        if arxiv_ids
        else "用户点名论文"
    )
    return [
        f"识别目标为{label}",
        "说明当前库内没有可验证论文证据或证据不足",
        "不编造目标论文的作者、摘要、方法、实验或结论",
        "不使用无关或相似论文替代目标论文",
        "给出下一步：导入论文、换关键词、放宽范围或人工确认",
    ]


def common_user_constraints(text: str) -> list[str]:
    points: list[str] = []
    years = re.findall(r"\b20\d{2}(?:\s*[到至-]\s*20\d{2})?\b", text)
    if "中文" in text or "Chinese" in text:
        points.append("按用户要求使用中文回答")
    if "英文" in text or "English" in text:
        points.append("按用户要求保留或使用英文内容")
    if years:
        points.append("遵守用户指定的时间范围或年份：" + "、".join(dedupe(years)))
    if any(marker in text for marker in ["对比", "比较", "差异", "本质差异"]):
        points.append("明确比较对象的共同点、差异点和判断依据")
    if any(marker in text for marker in ["表格", "table"]):
        points.append("按用户要求使用表格组织比较结果")
    if any(marker in text for marker in ["JSON", "json"]):
        points.append("按用户要求输出 JSON 或字段清晰的结构化结果")
    return points


def build_evidence_index(metadata_root: Path, source_root: Path) -> dict[str, PaperEvidence]:
    evidence: dict[str, PaperEvidence] = {}
    source_paths = {source_paper_id(path): path for path in source_root.glob("**/*.tar.gz")}
    for metadata_path in metadata_root.glob("**/*.json"):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        paper_id = str(payload.get("paper_id") or metadata_path.stem)
        source_path = source_paths.get(paper_id)
        source_extract = build_tex_extract(source_path) if source_path else ""
        evidence[paper_id] = PaperEvidence(
            paper_id=paper_id,
            title=str(payload.get("title") or paper_id),
            abstract=str(payload.get("abstract") or ""),
            categories=[str(item) for item in payload.get("categories") or []],
            source_extract=source_extract,
        )
    return evidence


def source_paper_id(path: Path) -> str:
    if path.name.endswith(".tar.gz"):
        return path.name.removesuffix(".tar.gz")
    return path.stem


def build_tex_extract(source_path: Path) -> str:
    tex = read_first_tex(source_path)
    if not tex:
        return ""
    sections = split_latex_sections(tex)
    method = collect_section_snippets(sections, METHOD_KEYS, "方法/理论依据")
    experiments = collect_section_snippets(sections, EXPERIMENT_KEYS, "实验/结果依据")
    limits = collect_section_snippets(sections, LIMIT_KEYS, "边界/局限依据")
    extract = " ".join(item for item in [method, experiments, limits] if item)
    if extract:
        return extract
    return collect_general_body_snippets(sections)


def read_first_tex(source_path: Path) -> str:
    try:
        with tarfile.open(source_path, "r:gz") as archive:
            names = sorted(
                [name for name in archive.getnames() if name.endswith(".tex")],
                key=lambda name: (Path(name).name not in {"arxiv.tex", "main.tex"}, name.lower()),
            )
            contents: list[str] = []
            for name in names:
                member = archive.extractfile(name)
                if member is None:
                    continue
                data = member.read()
                try:
                    contents.append(data.decode("utf-8"))
                except UnicodeDecodeError:
                    contents.append(data.decode("latin-1", errors="ignore"))
            return "\n".join(contents)
    except (tarfile.TarError, OSError):
        return ""
    return ""


def split_latex_sections(tex: str) -> list[tuple[str, str]]:
    matches = list(LATEX_SECTION_RE.finditer(tex))
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        title = clean_latex(match.group("title"))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(tex)
        content = clean_latex(tex[start:end])
        if title and content:
            sections.append((title, content))
    return sections


def collect_section_snippets(
    sections: list[tuple[str, str]],
    keys: tuple[str, ...],
    label: str,
) -> str:
    snippets: list[str] = []
    for title, content in sections:
        haystack = title.lower()
        if not any(key.lower() in haystack for key in keys):
            continue
        snippet = first_sentences(content, max_chars=420)
        if snippet:
            snippets.append(snippet)
        if len(snippets) >= 2:
            break
    if not snippets:
        return ""
    return f"{label}：" + " ".join(snippets)


def collect_general_body_snippets(sections: list[tuple[str, str]]) -> str:
    skipped_titles = ("references", "bibliography", "acknowledg", "supplement")
    snippets: list[str] = []
    for title, content in sections:
        if any(marker in title.lower() for marker in skipped_titles):
            continue
        snippet = first_sentences(content, max_chars=420)
        if snippet:
            snippets.append(snippet)
        if len(snippets) >= 2:
            break
    if not snippets:
        return ""
    return "正文依据：" + " ".join(snippets)


def clean_latex(text: str) -> str:
    text = re.sub(r"%.*", " ", text)
    text = re.sub(r"\\begin\{[^{}]+\}|\\end\{[^{}]+\}", " ", text)
    text = re.sub(r"\\(cite|ref|label|footnote|url|href)\*?(?:\[[^\]]*\])?\{[^{}]*\}", " ", text)
    text = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{[^{}]*\}", " ", text)
    text = re.sub(r"\\caption\{", " ", text)
    text = text.replace("\\noindent", " ")
    text = text.replace("\\textbf", " ")
    text = text.replace("\\textit", " ")
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", text)
    text = text.replace("{", " ").replace("}", " ")
    text = text.replace("$", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def first_sentences(text: str, *, max_chars: int) -> str:
    cleaned = clean_sentence_block(text)
    if len(cleaned) <= max_chars:
        return cleaned
    parts = re.split(r"(?<=[.!?。！？])\s+", cleaned)
    result = ""
    for part in parts:
        if not part:
            continue
        candidate = f"{result} {part}".strip()
        if len(candidate) > max_chars:
            break
        result = candidate
    return result or truncate(cleaned, max_chars)


def clean_sentence_block(text: str) -> str:
    text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    return text


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def extract_arxiv_ids(text: str) -> list[str]:
    seen: set[str] = set()
    ids: list[str] = []
    for item in ARXIV_RE.findall(text):
        normalized = re.sub(r"v\d+$", "", item.lower())
        if normalized not in seen:
            ids.append(normalized)
            seen.add(normalized)
    return ids


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        cleaned = item.strip()
        if cleaned and cleaned not in seen:
            result.append(cleaned)
            seen.add(cleaned)
    return result


def build_summary(rows: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    tags = [row.get("tags") or {} for row in rows]
    summary: dict[str, Any] = {
        "source": "docs/25-真实用户问题测试集.md",
        "output": str(output_path),
        "count": len(rows),
        "source_type": counter(tags, "source_type"),
        "input_quality": counter(tags, "input_quality"),
        "rag_applicability": counter(tags, "rag_applicability"),
        "expected_handling": counter(tags, "expected_handling"),
        "reference_kind": counter(tags, "reference_kind"),
        "rag_eval_suitable": counter(tags, "rag_eval_suitable"),
        "corpus_check": counter(tags, "corpus_check"),
        "reference_refinement": {
            "annotation_version": ANNOTATION_VERSION,
            "basis": ANNOTATION_BASIS,
            "rag_retrieval_cases": count_rag_references(rows),
            "factual_references": sum(
                1
                for row in rows
                if is_rag_retrieval_case(row.get("tags") or {})
                and is_factual_reference_text(str(row.get("reference") or ""))
            ),
        },
    }
    existing = SUMMARY_PATH if output_path == ANNOTATIONS_PATH else None
    if existing and existing.exists():
        try:
            old = json.loads(existing.read_text(encoding="utf-8"))
            if isinstance(old.get("generation"), dict):
                summary["generation"] = old["generation"]
        except json.JSONDecodeError:
            pass
    return summary


def is_factual_reference_text(reference: str) -> bool:
    guidance_markers = [
        "理想回答应",
        "围绕目标论文",
        "先交代",
        "再概括",
        "避免编造",
        "答案覆盖目标论文身份",
        "正文阅读重点放在",
    ]
    return not any(marker in reference for marker in guidance_markers)


def counter(tags: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(str(tag.get(field)) for tag in tags if field in tag)
    return dict(sorted(counts.items()))


def count_rag_references(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if is_rag_retrieval_case(row.get("tags") or {}))


def render_markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# 真实用户问题测试集 RAG 标注（1000 条）",
        "",
        (
            "本文档是 `docs/25-真实用户问题测试集.md` 的逐条标注版。"
            "机器可读的完整标注以 "
            "`data/eval/real_user_questions_rag_annotations.jsonl` 为准。"
        ),
        "",
        "## 标注口径",
        "",
        (
            "- `reference` 用于官方 RAGAS 的 Context Recall、Noise Sensitivity、"
            "Semantic Similarity 等指标。"
        ),
        (
            "- 进入 `rag_retrieval` 评测轨道的样本使用事实型标准答案；"
            "reference 不再使用“理想回答应……”这种评分规则描述。"
        ),
        "- `required_points` 用于项目自定义 `completeness` 指标，每项尽量保持原子化。",
        "- 非 RAG 的系统操作、记忆、多轮上下文和鲁棒性样例保留行为型 reference。",
        "",
        "## 查证快照",
        "",
        "| 项目 | 值 |",
        "|---|---:|",
        f"| annotation_version | `{ANNOTATION_VERSION}` |",
    ]
    generation = summary.get("generation") if isinstance(summary.get("generation"), dict) else {}
    for key in [
        "qdrant_points_count",
        "qdrant_indexed_vectors_count",
        "qdrant_unique_papers",
        "expected_source_ids_verified_in_current_qdrant",
    ]:
        if key in generation:
            lines.append(f"| {key} | {generation[key]} |")
    lines.extend(
        [
            f"| rag_retrieval_cases | {summary['reference_refinement']['rag_retrieval_cases']} |",
            f"| factual_references | {summary['reference_refinement']['factual_references']} |",
            "",
            "## RAG 适用性分布",
            "",
            "| RAG 适用性 | 数量 |",
            "|---|---:|",
        ]
    )
    for key, value in summary.get("rag_applicability", {}).items():
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## Corpus Check 分布",
            "",
            "| corpus_check | 数量 |",
            "|---|---:|",
        ]
    )
    for key, value in summary.get("corpus_check", {}).items():
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## 逐条标注",
            "",
            (
                "| ID | 会话 | 轮次 | 类型 | 质量 | RAG适用性 | RAG可评分 | 期望处理 | "
                "参考类型 | corpus_check | expected_source_ids | reference | required_points | "
                "用户原话 |"
            ),
            "|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        tags = row.get("tags") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    escape_md(str(row.get("case_id") or "")),
                    escape_md(str(row.get("session_id") or "")),
                    escape_md(str(row.get("turn") or "")),
                    escape_md(str(tags.get("source_type") or "")),
                    escape_md(str(tags.get("input_quality") or "")),
                    escape_md(str(tags.get("rag_applicability") or "")),
                    escape_md(str(tags.get("rag_eval_suitable") or "")),
                    escape_md(str(tags.get("expected_handling") or "")),
                    escape_md(str(tags.get("reference_kind") or "")),
                    escape_md(str(tags.get("corpus_check") or "")),
                    escape_md(
                        "、".join(str(item) for item in row.get("expected_source_ids") or [])
                    ),
                    escape_md(truncate(str(row.get("reference") or ""), 420)),
                    escape_md("<br>".join(str(item) for item in row.get("required_points") or [])),
                    escape_md(str(row.get("user_input") or "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def escape_md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
