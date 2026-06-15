from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from refine_rag_annotations import (
    PaperEvidence,
    build_evidence_index,
    clean_sentence_block,
    truncate,
)


SOURCE_PATH = Path("docs/25-真实用户问题测试集.md")
OUTPUT_PATH = Path("data/eval/real_user_questions_rag_annotations.jsonl")
SUMMARY_PATH = Path("data/eval/real_user_questions_rag_annotations.summary.json")
DOC_PATH = Path("docs/26-真实用户问题测试集RAG标注.md")
METADATA_ROOT = Path("data/raw/arxiv/metadata")
SOURCE_ROOT = Path("data/raw/arxiv/source")

ANNOTATION_VERSION = "research_150_actual_answer_annotations_2026-04-29"
ANNOTATION_BASIS = "actual_answer_references_from_local_corpus_and_memory_scenarios"
ARXIV_RE = re.compile(r"\b\d{4}\.\d{5}(?:v\d+)?\b", re.IGNORECASE)


CASE_SOURCE_TYPES = {
    **{f"Q{index:04d}": "paper_qa" for index in range(1, 10)},
    **{f"Q{index:04d}": "idea_novelty" for index in range(10, 19)},
    **{f"Q{index:04d}": "trend_analysis" for index in range(19, 27)},
    **{f"Q{index:04d}": "cross_domain" for index in range(27, 35)},
    **{f"Q{index:04d}": "study_plan" for index in range(35, 43)},
    **{f"Q{index:04d}": "paper_reading" for index in range(43, 51)},
}

MANUAL_SOURCE_IDS = {
    "Q0011": ["2604.21284", "2604.21304", "2604.21345"],
    "Q0013": ["2604.21345", "2604.21380"],
    "Q0015": ["2604.21284", "2604.21304", "2604.21345"],
    "Q0016": ["2604.21477", "2604.21679"],
    "Q0021": ["2604.21304", "2604.21284", "2604.21345", "2604.21570"],
    "Q0024": ["2604.21477", "2604.21679"],
    "Q0026": ["2604.21345", "2604.21380"],
    "Q0035": ["2604.21345", "2604.21304", "2604.21570"],
    "Q0036": ["2604.21284", "2604.21304"],
    "Q0038": ["2604.21345", "2604.21380"],
    "Q0039": ["2604.21326", "2604.21304"],
    "Q0041": ["2604.21477", "2604.21679"],
    "Q0042": ["2604.21304", "2604.21345"],
}


MEMORY_REFERENCES = {
    "MEM001": [
        "已记住：论文问答默认先给结论，再给证据。",
        "已补充：论文问答引用证据时保留 arXiv ID 和论文标题，不只写简称。",
        "已补充：检索证据不足时先说明不足，再给可执行的补充检索建议。",
        "可用答案：方法主线回答应按“结论一句话 → arXiv ID 和标题 → 摘要/正文证据 → 局限或证据缺口 → 下一步检索建议”组织。",
        "已更正：结论可以短，但证据部分必须明确对应到论文片段。",
        "可用答案：模板为“结论：...；目标论文：arXiv:...《...》；证据1：...；证据2：...；推断：...；证据不足/下一步：...”。",
        "可用答案：本次可用英文回答 slides 内容，但不应把默认中文偏好改为英文。",
        "可用答案：恢复默认中文；论文问答仍按短结论、明确证据片段、证据不足先声明的结构回答。",
        "已补充：长期偏好是不把没有证据的推断写成论文事实。",
        "可用答案：已记住的论文问答偏好包括先结论后证据、保留 arXiv ID 和标题、证据不足先声明、证据片段要明确、推断不能写成事实。"
    ],
    "MEM002": [
        "已记住：idea novelty 评估优先关注已有工作覆盖了什么，不只给乐观建议。",
        "已补充：novelty 分析需要分开写 overlap、difference、evidence gap。",
        "已补充：如果 idea 太泛，应先收窄成可检索版本。",
        "可用答案：应先把 RAG 评测报告结构拆成失败样例组织、指标聚合、证据引用、诊断流程四个检索问题，再查已有覆盖。",
        "已更正：novelty 分析最后必须给出最低成本验证实验。",
        "已记住：idea novelty 固定输出项包括最低成本验证实验。",
        "可用答案：本次 workshop 可接受 exploratory idea，但默认严谨标准不变。",
        "可用答案：默认标准是先查覆盖，再列差异和证据缺口，最后给最低成本验证实验；证据不足不能下确定判断。",
        "已补充：库内论文不足时要明确标记证据缺口，不给确定判断。",
        "可用答案：idea novelty 偏好包括覆盖优先、overlap/difference/evidence gap 分离、先收窄泛化 idea、输出最低成本验证实验、证据不足要明说。"
    ],
    "MEM003": [
        "已记住：trend 分析不能只看热门词，还要覆盖方法和评测变化。",
        "已补充：趋势分析优先按问题、方法、数据集、指标四类组织。",
        "已补充：趋势结论只来自少量论文时要提醒样本不足。",
        "可用答案：agent memory 趋势模板为“研究问题变化、方法路线变化、数据/任务变化、指标变化、代表论文、样本不足提示、后续 query”。",
        "已更正：主题很新时可接受少量论文，但必须标记为早期信号。",
        "可用答案：tool-use agent safety 趋势应按攻击面、工具链机制、评测场景、验证指标组织，并标注是否只是早期信号。",
        "可用答案：本次只给一页趋势概览，但不要保存为长期默认。",
        "可用答案：下次趋势报告恢复默认结构：问题、方法、数据集、指标、代表论文、样本边界和后续 query。",
        "已补充：趋势报告最后给 3 个可继续追踪的检索 query。",
        "可用答案：trend 偏好包括方法/评测变化优先、四类结构、样本不足或早期信号标记，以及最后给 3 个后续检索 query。"
    ],
    "MEM004": [
        "已记住：cross-domain 分析最关心迁移边界和失败条件。",
        "已补充：跨领域迁移回答固定包含 source mechanism、target mismatch、validation plan。",
        "已补充：源领域和目标领域数据形态差异大时要主动指出。",
        "可用答案：MCP safety 到代码生成迁移应写源机制、代码生成目标任务、数据/权限/工具链不匹配、风险、baseline、ablation 和失败条件。",
        "已更正：cross-domain 的 validation plan 至少包含 baseline 和 ablation。",
        "已记住：baseline 和 ablation 是 cross-domain 固定结构的一部分。",
        "可用答案：本次只列风险，但不改变默认完整结构。",
        "可用答案：下次迁移合理性问题恢复完整结构，而不是只列风险。",
        "已补充：如果迁移只是类比、没有机制对应，应直接说不建议继续。",
        "可用答案：cross-domain 偏好包括迁移边界、失败条件、source mechanism、target mismatch、validation plan、baseline、ablation、数据形态差异和机制对应检查。"
    ],
    "MEM005": [
        "已记住：study plan 每天最多安排 45 分钟。",
        "已补充：学习计划要把阅读、复现和检查点分开列。",
        "已补充：工作日适合轻量阅读，周末适合复现。",
        "可用答案：一周 RAG evaluation 计划应把工作日安排为 45 分钟内阅读/笔记，周末安排复现和检查点产出。",
        "已记录为临时例外：本周五有两个小时可安排一次复现，但不改变长期时间偏好。",
        "可用答案：本周计划应保留工作日轻量节奏，只在周五加入一次两小时复现，周末仍做较重检查点。",
        "可用答案：下周默认节奏不应继续假设周五有两个小时。",
        "可用答案：paper-reading 能力训练计划应按默认 45 分钟上限安排，工作日读摘要/方法图/实验表，周末做一次完整精读复述。",
        "已补充：每个学习计划最后要有可检查产出。",
        "可用答案：study-plan 偏好包括每日 45 分钟、阅读/复现/检查点分离、工作日轻量阅读、周末复现、临时空档不长期保存、每个计划有可检查产出。"
    ],
    "MEM006": [
        "已记住：paper-reading 默认先看摘要、方法图、实验表，再读细节。",
        "已补充：精读时要逐段说明段落在论文论证中的作用。",
        "已补充：有公式时先解释变量含义，再解释公式目的。",
        "可用答案：精读流程为摘要定位问题、方法图梳理模块、实验表看证据，再逐段解释论证作用和公式变量。",
        "已更正：理论论文可先看定理和证明结构，再看实验或例子。",
        "可用答案：理论论文精读应先定位定理、假设、证明主线和关键引理，再看例子或实验。",
        "可用答案：本次只读 conclusion 是临时需求，不保存为长期偏好。",
        "可用答案：下次精读恢复默认流程，不只读 conclusion。",
        "已补充：精读结束时给 3 个复述检查问题。",
        "可用答案：paper-reading 偏好包括先摘要/方法图/实验表、逐段解释论证作用、公式先变量后目的、理论论文先定理证明结构、临时 conclusion 不长期保存、结尾给 3 个复述问题。"
    ],
    "MEM007": [
        "已记住：所有 research 功能默认使用中文回答。",
        "已补充：用户明确要求英文时，单次可用英文，但不改变默认语言。",
        "已补充：所有功能回答都要区分事实、推断和建议。",
        "可用答案：trend 问题应中文回答，并按事实证据、趋势推断、后续建议分区组织。",
        "已更正：study plan 可以更直接，不需要每一步都写事实、推断和建议。",
        "可用答案：study-plan 更偏执行清单；其他功能仍保持事实、推断、建议分离。",
        "可用答案：本次很短回答是临时需求，不覆盖默认详细程度。",
        "可用答案：下次 idea novelty 或 cross-domain 恢复默认证据结构。",
        "已补充：无法验证时必须明确说没有验证，不要写得像已经确认。",
        "可用答案：通用偏好包括默认中文、英文仅单次覆盖、事实/推断/建议分离、study-plan 可更直接、临时短答不长期保存、无法验证要明说。"
    ],
    "MEM008": [
        "已记住：长期研究主题是 research agent evaluation。",
        "已补充：短期关注 agent memory compression，持续到本周五。",
        "已补充：推荐论文或做趋势分析时优先长期主题，再考虑本周短期主题。",
        "可用答案：trend query 可围绕 research agent evaluation 的任务和指标变化；idea novelty query 可围绕 agent memory compression 如何改进 research agent evaluation。",
        "已更正：本周内短期主题可排第一，下周自动回到长期主题。",
        "可用答案：当前仍在本周，study plan 主题可优先设为 agent memory compression，但需连接到长期 research agent evaluation。",
        "可用答案：今天看推荐系统论文是临时需求，不保存为长期兴趣。",
        "可用答案：下周默认推荐不应优先推荐推荐系统论文。",
        "已补充：负偏好是不喜欢只有 leaderboard 提升、没有机制分析的论文。",
        "可用答案：长期主题是 research agent evaluation；短期主题是本周五前的 agent memory compression；本周短期可优先，下周回长期；推荐系统是临时；过滤掉只有榜单提升无机制分析的论文。"
    ],
    "MEM009": [
        "已记住：论文问答偏好 hybrid 检索结果，同时提示 sparse 是否漏掉关键词。",
        "已补充：idea novelty 中 dense 和 sparse 找到的相近论文不一致时要提醒。",
        "已补充：trend 分析不必逐条列所有检索结果，但要说明检索覆盖范围。",
        "已补充：cross-domain 不只依赖语义相似，还要检查机制是否一致。",
        "已更正：paper-reading 不需要对比检索策略，只围绕目标论文上下文。",
        "可用答案：论文问答 hybrid 优先并提示 sparse 补充；novelty 标记 dense/sparse 分歧；trend 说明覆盖范围；cross-domain 检查机制一致；paper-reading 不做策略对比。",
        "可用答案：今天调试 RAG 时临时优先 sparse，不保存为长期默认。",
        "可用答案：下次正常论文问答恢复 hybrid 优先，并提示 sparse 可能补充的关键词证据。",
        "已补充：用户没有指定论文时，先检索候选再回答，不凭记忆直接答。",
        "可用答案：检索和路由偏好包括 ask 用 hybrid 优先、novelty 关注 dense/sparse 分歧、trend 说明覆盖范围、cross-domain 看机制一致、paper-reading 围绕目标论文、无指定论文先检索候选。"
    ],
    "MEM010": [
        "已记住：不希望系统把临时会话需求写成长期记忆。",
        "已补充：用户说“请记住”时，默认可以写入长期偏好。",
        "已补充：用户说“今天临时”时，只用于当前会话。",
        "已补充：记忆冲突时优先采用最新明确更正。",
        "已更正：明显长期偏好即使没说“请记住”，也可以先询问是否保存。",
        "可用答案：遇到偏好变化时，应区分明确长期、临时、本轮执行和冲突；明显长期但未授权保存时先询问。",
        "可用答案：今天所有回答缩成一句话是临时需求，不长期保存。",
        "可用答案：明天 paper-reading 恢复默认精读偏好。",
        "已补充：新偏好和旧偏好冲突时，要指出冲突并确认是否覆盖。",
        "可用答案：长期记忆规则是“请记住”默认保存，“今天临时”不保存，明显长期偏好先询问，冲突采用最新明确更正但先指出并确认，临时一句话输出不影响明天默认 paper-reading。"
    ],
}


def main() -> None:
    rows = parse_markdown_table(SOURCE_PATH)
    if len(rows) != 150:
        raise SystemExit(f"Expected 150 rows, found {len(rows)}")

    evidence = build_evidence_index(METADATA_ROOT, SOURCE_ROOT)
    annotations = [build_annotation(row, evidence) for row in rows]
    validate_annotations(annotations)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in annotations)
        + "\n",
        encoding="utf-8",
    )
    summary = build_summary(annotations)
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    DOC_PATH.write_text(render_markdown(annotations, summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def parse_markdown_table(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.startswith("| Q"):
            continue
        parts = [part.strip() for part in raw_line.strip().strip("|").split("|")]
        if len(parts) != 4:
            raise ValueError(f"Unexpected table row shape: {raw_line}")
        case_id, session_id, turn, user_input = parts
        rows.append(
            {
                "case_id": case_id,
                "session_id": session_id,
                "turn": int(turn),
                "user_input": user_input,
            }
        )
    return rows


def build_annotation(row: dict[str, Any], evidence: dict[str, PaperEvidence]) -> dict[str, Any]:
    case_id = str(row["case_id"])
    text = str(row["user_input"])
    session_id = str(row["session_id"])
    turn = int(row["turn"])
    source_type = CASE_SOURCE_TYPES.get(case_id, "memory_context")
    explicit_ids = extract_arxiv_ids(text)
    source_ids = explicit_ids or MANUAL_SOURCE_IDS.get(case_id, [])
    available_ids = [paper_id for paper_id in source_ids if paper_id in evidence]
    missing_ids = [paper_id for paper_id in explicit_ids if paper_id not in evidence]
    input_quality = "unsupported" if missing_ids else "good"

    if source_type == "memory_context":
        reference = build_memory_reference(session_id, turn)
        required_points = build_memory_required_points(session_id, turn)
        expected_source_ids: list[str] = []
    else:
        papers = [evidence[paper_id] for paper_id in available_ids]
        reference = build_research_reference(
            case_id=case_id,
            source_type=source_type,
            text=text,
            papers=papers,
            missing_ids=missing_ids,
        )
        required_points = build_research_required_points(
            source_type=source_type,
            papers=papers,
            missing_ids=missing_ids,
        )
        expected_source_ids = available_ids

    tags = {
        "annotation_basis": ANNOTATION_BASIS,
        "annotation_version": ANNOTATION_VERSION,
        "corpus_check": corpus_check(available_ids, missing_ids, source_type),
        "evaluation_tracks": evaluation_tracks(source_type),
        "expected_handling": expected_handling(source_type, input_quality),
        "explicit_arxiv_ids": explicit_ids,
        "input_quality": input_quality,
        "rag_applicability": rag_applicability(source_type, input_quality),
        "rag_eval_suitable": source_type != "memory_context" and input_quality == "good",
        "reference_kind": reference_kind(source_type, input_quality),
        "requires_conversation_context": source_type == "memory_context",
        "requires_memory_context": source_type == "memory_context",
        "source_doc": str(SOURCE_PATH),
        "source_type": source_type,
        "system_query_type": system_query_type(source_type),
        "verification_basis": "local_corpus_metadata_and_testset_memory_state",
    }
    return {
        "case_id": case_id,
        "expected_source_ids": expected_source_ids,
        "reference": reference,
        "required_points": required_points,
        "session_id": session_id,
        "tags": tags,
        "turn": turn,
        "user_input": text,
    }


def build_research_reference(
    *,
    case_id: str,
    source_type: str,
    text: str,
    papers: list[PaperEvidence],
    missing_ids: list[str],
) -> str:
    if missing_ids and not papers:
        label = "、".join(f"arXiv:{paper_id}" for paper_id in missing_ids)
        return (
            f"可用答案：当前库内没有可验证证据覆盖 {label}。不能编造该论文的作者、摘要、"
            "方法、实验或结论，也不能用相似论文替代目标论文。下一步应导入该论文、改用库内"
            "候选论文，或让用户放宽主题范围后再检索。"
        )

    evidence_text = " ".join(paper_answer_block(paper) for paper in papers[:3])
    if source_type == "paper_qa":
        return (
            "可用答案：先直接回答用户指定问题，再给证据。"
            f"{evidence_text} 结论必须只来自上述摘要和正文片段；没有检索覆盖的作者、实验细节、"
            "章节位置或数值结果要标为证据不足。"
        )
    if source_type == "idea_novelty":
        return (
            "可用答案：先把用户 idea 收窄为可检索表述，再判断已有覆盖、差异和证据缺口。"
            f"{evidence_text} novelty 风险来自与上述论文的问题设定、方法机制或评测流程重合；"
            "可保留的新意应落在目标场景、失败分析、交互流程或验证指标的差异上。最低成本验证实验是"
            "选一个小规模库内论文/请求集，比较 baseline、用户方案和去掉关键组件的 ablation。"
        )
    if source_type == "trend_analysis":
        return (
            "可用答案：趋势应按研究问题、方法路线、数据/任务、指标四类组织。"
            f"{evidence_text} 基于这些证据，可把趋势写成：研究对象从单篇论文问答扩展到 paper agent、"
            "评测流水线、工具安全和形式化规格；方法上强调结构化流水线、可追踪证据、失败样例和可验证指标；"
            "如果只覆盖少量库内论文，应标注为早期信号而非全领域结论。"
        )
    if source_type == "cross_domain":
        return (
            "可用答案：迁移分析应固定包含 source mechanism、target mismatch、validation plan。"
            f"{evidence_text} 迁移合理性取决于源论文机制是否能映射到目标任务；不匹配点通常包括数据形态、"
            "反馈信号、风险边界和评测指标。验证计划至少包含一个原领域或朴素 baseline、一个目标领域 baseline、"
            "以及去掉关键机制的 ablation；如果只是标题类比而没有机制对应，应建议暂停。"
        )
    if source_type == "study_plan":
        return (
            "可用答案：学习计划应按阶段给出阅读、复现和检查点。"
            f"{evidence_text} 第一阶段读摘要、问题设定和方法图；第二阶段读实验/评测部分并整理指标；"
            "第三阶段做一个最小复现或评测模板。每一天应有可检查产出，例如一页笔记、对比表、复述问题或小实验结果。"
        )
    if source_type == "paper_reading":
        return (
            "可用答案：精读应围绕目标论文逐段解释，而不是泛泛综述。"
            f"{evidence_text} 应先说明该段在论文论证中的作用，再解释关键变量、模块、实验表或威胁模型；"
            "结尾给出 3 个复述检查问题，并标注哪些细节当前检索上下文没有覆盖。"
        )
    raise ValueError(f"Unsupported source_type: {source_type} for {case_id}")


def paper_answer_block(paper: PaperEvidence) -> str:
    categories = "、".join(paper.categories[:4])
    identity = f"目标证据：arXiv:{paper.paper_id}《{paper.title}》"
    if categories:
        identity += f"（类别：{categories}）"
    abstract = truncate(clean_sentence_block(paper.abstract), 520)
    body = truncate(clean_sentence_block(paper.source_extract), 520)
    parts = [identity + "。"]
    if abstract:
        parts.append(f"摘要事实：{abstract}")
    if body:
        parts.append(f"正文事实：{body}")
    return "".join(parts)


def build_memory_reference(session_id: str, turn: int) -> str:
    answers = MEMORY_REFERENCES.get(session_id)
    if not answers or turn < 1 or turn > len(answers):
        raise ValueError(f"Missing memory reference for {session_id} turn {turn}")
    answer = answers[turn - 1]
    if answer.startswith("可用答案："):
        return answer
    return "可用答案：" + answer


def build_research_required_points(
    *,
    source_type: str,
    papers: list[PaperEvidence],
    missing_ids: list[str],
) -> list[str]:
    if missing_ids and not papers:
        label = "、".join(f"arXiv:{paper_id}" for paper_id in missing_ids)
        return [
            f"识别目标为 {label}",
            "说明当前库内没有可验证论文证据",
            "不编造论文事实，不用相似论文替代目标论文",
            "给出导入论文、放宽范围或调整检索词的下一步",
        ]

    paper_labels = [f"arXiv:{paper.paper_id}《{paper.title}》" for paper in papers]
    points = [
        "准确识别并围绕目标证据：" + "、".join(paper_labels),
        "使用库内摘要事实说明问题背景或研究对象",
        "使用库内正文片段说明方法、实验、评测、结果或局限",
        "区分论文事实、基于证据的推断和证据不足内容",
    ]
    feature_points = {
        "paper_qa": ["直接回答用户指定维度", "指出证据边界和复现/局限风险"],
        "idea_novelty": ["比较 overlap、difference、evidence gap", "给出最低成本验证实验"],
        "trend_analysis": ["按问题、方法、数据/任务、指标组织趋势", "标注样本不足或早期信号"],
        "cross_domain": ["说明 source mechanism 和 target mismatch", "给出 baseline、ablation 和失败条件"],
        "study_plan": ["给出阶段安排、检查点和可检查产出", "把锚点论文放入合适学习阶段"],
        "paper_reading": ["逐段说明段落在论文论证中的作用", "给出复述检查问题和证据缺口"],
    }
    points.extend(feature_points[source_type])
    return points


def build_memory_required_points(session_id: str, turn: int) -> list[str]:
    return [
        f"沿用 {session_id} 前序轮次中的用户偏好和更正",
        "区分长期偏好、临时需求和明确更正",
        "本轮回答要给出可直接执行的内容，而不是只描述评分规则",
        "如果新旧偏好冲突，应采用最新明确更正并保留临时需求边界",
    ]


def extract_arxiv_ids(text: str) -> list[str]:
    seen: set[str] = set()
    ids: list[str] = []
    for item in ARXIV_RE.findall(text):
        normalized = re.sub(r"v\d+$", "", item.lower())
        if normalized not in seen:
            ids.append(normalized)
            seen.add(normalized)
    return ids


def corpus_check(available_ids: list[str], missing_ids: list[str], source_type: str) -> str:
    if source_type == "memory_context":
        return "no_expected_source_ids_by_design"
    if missing_ids and not available_ids:
        return "explicit_arxiv_id_not_in_current_corpus"
    if available_ids:
        return "expected_source_ids_verified_in_local_metadata"
    return "topic_query_no_fixed_source_ids"


def evaluation_tracks(source_type: str) -> list[str]:
    if source_type == "memory_context":
        return ["memory_behavior", "answer_quality"]
    return ["rag_retrieval", "answer_quality"]


def expected_handling(source_type: str, input_quality: str) -> str:
    if input_quality == "unsupported":
        return "explain_missing_corpus_evidence"
    if source_type == "memory_context":
        return "answer_with_memory_context"
    return "answer_with_retrieved_evidence"


def rag_applicability(source_type: str, input_quality: str) -> str:
    if source_type == "memory_context":
        return "requires_memory_context"
    if input_quality == "unsupported":
        return "requires_clarification_or_ingest"
    return "required"


def reference_kind(source_type: str, input_quality: str) -> str:
    if source_type == "memory_context":
        return "actual_memory_behavior_answer"
    if input_quality == "unsupported":
        return "missing_corpus_actual_answer"
    return "actual_corpus_grounded_answer"


def system_query_type(source_type: str) -> str | None:
    return {
        "paper_qa": "qa",
        "idea_novelty": "idea_novelty",
        "trend_analysis": "trend",
        "cross_domain": "cross_domain",
        "study_plan": "study_plan",
        "paper_reading": "paper_reading",
        "memory_context": "qa",
    }.get(source_type)


def validate_annotations(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 150:
        raise ValueError(f"Expected 150 annotations, got {len(rows)}")
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        expected_id = f"Q{index:04d}"
        if row["case_id"] != expected_id:
            raise ValueError(f"Expected {expected_id}, got {row['case_id']}")
        if row["case_id"] in seen:
            raise ValueError(f"Duplicate case_id: {row['case_id']}")
        seen.add(row["case_id"])
        if not str(row.get("reference") or "").startswith("可用答案："):
            raise ValueError(f"Reference is not an actual answer: {row['case_id']}")
        if not row.get("required_points"):
            raise ValueError(f"Missing required_points: {row['case_id']}")


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tags = [row["tags"] for row in rows]
    return {
        "annotation_basis": ANNOTATION_BASIS,
        "annotation_version": ANNOTATION_VERSION,
        "count": len(rows),
        "corpus_check": count_tags(tags, "corpus_check"),
        "expected_handling": count_tags(tags, "expected_handling"),
        "input_quality": count_tags(tags, "input_quality"),
        "output": str(OUTPUT_PATH),
        "rag_applicability": count_tags(tags, "rag_applicability"),
        "rag_eval_suitable": count_tags(tags, "rag_eval_suitable"),
        "reference_kind": count_tags(tags, "reference_kind"),
        "source": str(SOURCE_PATH),
        "source_type": count_tags(tags, "source_type"),
        "updated_at": datetime.now(UTC).isoformat(),
    }


def count_tags(tags: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(tag.get(key)) for tag in tags).items()))


def render_markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# 真实用户问题测试集 RAG 标注（150 条）",
        "",
        "本文档是 `docs/25-真实用户问题测试集.md` 的逐条标注版。机器可读完整标注以 "
        "`data/eval/real_user_questions_rag_annotations.jsonl` 为准。",
        "",
        "## 标注口径",
        "",
        "- `reference` 是可直接作为评测标准答案使用的实际答案，不使用“理想回答应……”式规则描述。",
        "- RAG 样例的 reference 基于本地论文元数据和正文片段；库内缺失论文明确给出缺失证据答案。",
        "- Memory 样例的 reference 基于同一会话前序轮次的记忆状态、临时偏好和冲突处理规则。",
        "- `required_points` 用于 completeness 检查，保留原子化覆盖点。",
        "",
        "## 查证快照",
        "",
        "| 项目 | 值 |",
        "|---|---:|",
        f"| annotation_version | `{summary['annotation_version']}` |",
        f"| count | {summary['count']} |",
        "",
        "## 类型分布",
        "",
        "| 类型 | 数量 |",
        "|---|---:|",
    ]
    for key, value in summary["source_type"].items():
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
    for key, value in summary["corpus_check"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## 逐条标注",
            "",
            "| ID | 会话 | 轮次 | 类型 | 质量 | RAG适用性 | RAG可评分 | 期望处理 | 参考类型 | corpus_check | expected_source_ids | reference | required_points | 用户原话 |",
            "|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        tags = row["tags"]
        lines.append(
            "| "
            + " | ".join(
                [
                    escape_md(row["case_id"]),
                    escape_md(row["session_id"]),
                    str(row["turn"]),
                    escape_md(tags["source_type"]),
                    escape_md(tags["input_quality"]),
                    escape_md(tags["rag_applicability"]),
                    escape_md(str(tags["rag_eval_suitable"])),
                    escape_md(tags["expected_handling"]),
                    escape_md(tags["reference_kind"]),
                    escape_md(tags["corpus_check"]),
                    escape_md("、".join(row["expected_source_ids"])),
                    escape_md(truncate(row["reference"], 420)),
                    escape_md("<br>".join(row["required_points"])),
                    escape_md(row["user_input"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def escape_md(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
