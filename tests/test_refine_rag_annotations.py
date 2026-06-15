from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.refine_rag_annotations import (
    PaperEvidence,
    build_evidence_index,
    build_refined_annotation,
    build_tex_extract,
    normalize_chinese_reference_text,
    refine_annotations,
)


def test_build_tex_extract_prefers_method_experiment_result_and_limit_sections(tmp_path: Path):
    source_path = tmp_path / "2604.00001.tar.gz"
    tex = r"""
\section{Introduction}
The paper introduces a retrieval method for grounded answers.
\section{Methodology}
The method uses a dense retriever, a sparse retriever, and a reranker.
\subsection{Implementation details}
The implementation trains the reranker on hard negatives.
\section{Experiments}
The experiments compare BM25, dense retrieval, and hybrid retrieval.
\section{Limitations}
The limitation is that all experiments use one corpus.
"""
    with tarfile.open(source_path, "w:gz") as archive:
        tex_path = tmp_path / "arxiv.tex"
        tex_path.write_text(tex, encoding="utf-8")
        archive.add(tex_path, arcname="arxiv.tex")

    extract = build_tex_extract(source_path)

    assert "方法/理论依据：The method uses a dense retriever" in extract
    assert "实验/结果依据：The experiments compare BM25" in extract
    assert "边界/局限依据：The limitation is that all experiments use one corpus" in extract
    assert "The paper introduces a retrieval method" not in extract


def test_build_tex_extract_falls_back_to_general_body_sections(tmp_path: Path):
    source_path = tmp_path / "2604.00002.tar.gz"
    tex = r"""
\section{Introduction}
This paper studies grounded retrieval for long scientific documents.
\section{Background}
The task requires evidence-preserving chunk selection.
"""
    with tarfile.open(source_path, "w:gz") as archive:
        tex_path = tmp_path / "arxiv.tex"
        tex_path.write_text(tex, encoding="utf-8")
        archive.add(tex_path, arcname="arxiv.tex")

    extract = build_tex_extract(source_path)

    assert extract.startswith("正文依据：")
    assert "This paper studies grounded retrieval" in extract


def test_build_tex_extract_reads_split_section_tex_files(tmp_path: Path):
    source_path = tmp_path / "2604.00003.tar.gz"
    main_tex = r"\input{sections/Approach}\input{sections/Experiments}"
    approach_tex = r"\section{Approach} The approach groups spectral bands before encoding."
    experiments_tex = r"\section{Results} Experiments show improved accuracy on PACE data."
    with tarfile.open(source_path, "w:gz") as archive:
        main_path = tmp_path / "main.tex"
        main_path.write_text(main_tex, encoding="utf-8")
        archive.add(main_path, arcname="main.tex")
        section_dir = tmp_path / "sections"
        section_dir.mkdir()
        approach_path = section_dir / "Approach.tex"
        approach_path.write_text(approach_tex, encoding="utf-8")
        archive.add(approach_path, arcname="sections/Approach.tex")
        experiments_path = section_dir / "Experiments.tex"
        experiments_path.write_text(experiments_tex, encoding="utf-8")
        archive.add(experiments_path, arcname="sections/Experiments.tex")

    extract = build_tex_extract(source_path)

    assert "方法/理论依据：The approach groups spectral bands" in extract
    assert "实验/结果依据：Experiments show improved accuracy" in extract


def test_build_refined_annotation_uses_factual_reference_for_rag_case():
    annotation = {
        "case_id": "Q0001",
        "user_input": "Explain the method of arXiv:2604.00001《Grounded Retrieval》.",
        "expected_source_ids": ["2604.00001"],
        "required_points": ["old"],
        "tags": {
            "evaluation_tracks": ["rag_retrieval", "answer_quality"],
            "source_type": "paper_qa",
            "reference_kind": "corpus_grounded_reference",
        },
    }
    evidence = {
        "2604.00001": PaperEvidence(
            paper_id="2604.00001",
            title="Grounded Retrieval",
            abstract=(
                "This paper proposes a hybrid retrieval method for grounded question answering."
            ),
            categories=["cs.IR", "cs.CL"],
            source_extract=(
                "方法/理论依据：The method combines dense retrieval and reranking. "
                "实验/结果依据：Experiments compare BM25 and dense retrieval."
            ),
        )
    }

    refined = build_refined_annotation(annotation, evidence)

    assert refined["reference"].startswith("arXiv:2604.00001《Grounded Retrieval》")
    assert "This paper proposes a hybrid retrieval method" in refined["reference"]
    assert "方法/理论依据" in refined["reference"]
    assert "理想回答应" not in refined["reference"]
    assert "标准答案" not in refined["reference"]
    assert "应" not in refined["reference"]
    assert any("摘要事实" in point for point in refined["required_points"])
    assert any("正文证据" in point for point in refined["required_points"])


def test_build_refined_annotation_uses_chinese_reference_for_chinese_request():
    annotation = {
        "case_id": "Q0001",
        "user_input": "请用中文解释 arXiv:2604.00001《Grounded Retrieval》的方法主线。",
        "expected_source_ids": ["2604.00001"],
        "required_points": ["old"],
        "tags": {
            "evaluation_tracks": ["rag_retrieval", "answer_quality"],
            "source_type": "paper_qa",
            "reference_kind": "corpus_grounded_reference",
        },
    }
    evidence = {
        "2604.00001": PaperEvidence(
            paper_id="2604.00001",
            title="Grounded Retrieval",
            abstract=(
                "This paper proposes a hybrid retrieval method for grounded question answering."
            ),
            categories=["cs.IR", "cs.CL"],
            source_extract=(
                "方法/理论依据：The method combines dense retrieval and reranking. "
                "实验/结果依据：Experiments compare BM25 and dense retrieval."
            ),
        )
    }

    refined = build_refined_annotation(annotation, evidence)

    assert "中文参考答案" not in refined["reference"]
    assert "方法主线：This paper proposes" in refined["reference"]
    assert "正文证据：方法/理论依据：The method combines" in refined["reference"]
    assert "正文阅读重点：方法或理论部分、实验或结果部分" in refined["reference"]
    assert "围绕目标论文" not in refined["reference"]
    assert "先交代" not in refined["reference"]
    assert "再概括" not in refined["reference"]
    assert "避免编造" not in refined["reference"]
    assert "按用户要求使用中文回答" in refined["required_points"]


def test_build_refined_annotation_uses_chinese_reference_override_for_chinese_request():
    annotation = {
        "case_id": "Q0001",
        "user_input": "请用中文解释 arXiv:2604.00001《Grounded Retrieval》的方法主线。",
        "expected_source_ids": ["2604.00001"],
        "required_points": ["old"],
        "tags": {
            "evaluation_tracks": ["rag_retrieval", "answer_quality"],
            "source_type": "paper_qa",
            "reference_kind": "corpus_grounded_reference",
        },
    }
    evidence = {
        "2604.00001": PaperEvidence(
            paper_id="2604.00001",
            title="Grounded Retrieval",
            abstract="This paper proposes a hybrid retrieval method.",
            categories=["cs.IR"],
        )
    }

    refined = build_refined_annotation(
        annotation,
        evidence,
        chinese_reference_overrides={
            "Q0001": "arXiv:2604.00001《Grounded Retrieval》。方法主线：该论文提出混合检索方法。"
        },
    )

    assert (
        refined["reference"]
        == "arXiv:2604.00001《Grounded Retrieval》。方法主线：该论文提出混合检索方法。"
    )
    assert "This paper proposes" not in refined["reference"]


def test_normalize_chinese_reference_text_translates_common_english_terms():
    normalized = normalize_chinese_reference_text(
        "arXiv:2604.00001《Large Language Models for Knowledge Graph Completion》。"
        "方法主线：Large Language Models support Knowledge Graph Completion. "
        "The agent uses source intake and structured reference construction."
    )

    assert "《Large Language Models for Knowledge Graph Completion》" in normalized
    assert "大语言模型" in normalized
    assert "知识图谱补全" in normalized
    assert "智能体" in normalized
    assert "agent" not in normalized
    assert "年龄nt" not in normalized
    assert "源数据接入" in normalized
    assert "结构化参考答案构建" in normalized


def test_normalize_chinese_reference_text_cleans_partial_agent_terms():
    normalized = normalize_chinese_reference_text(
        "方法主线：operator agent coordinates multi-agent systems and "
        "multi-agent latent trajectories. 潜在通信（latent communication） "
        "困惑度（perplexity, PPL）"
    )

    assert "操作员智能体" in normalized
    assert "多智能体系统" in normalized
    assert "多智能体潜在轨迹" in normalized
    assert "潜在通信（潜在通信）" not in normalized
    assert "潜在通信" in normalized
    assert "困惑度（PPL）" in normalized
    assert "operator 智能体" not in normalized
    assert "multi-智能体" not in normalized


def test_build_refined_annotation_does_not_infer_chinese_request_from_evidence_text():
    annotation = {
        "case_id": "Q0001",
        "user_input": "Explain arXiv:2604.00001《Grounded Retrieval》.",
        "expected_source_ids": ["2604.00001"],
        "required_points": ["old"],
        "tags": {
            "evaluation_tracks": ["rag_retrieval", "answer_quality"],
            "source_type": "paper_qa",
            "reference_kind": "corpus_grounded_reference",
        },
    }
    evidence = {
        "2604.00001": PaperEvidence(
            paper_id="2604.00001",
            title="Grounded Retrieval",
            abstract="This paper evaluates English and Chinese retrieval benchmarks.",
            categories=["cs.IR"],
        )
    }

    refined = build_refined_annotation(annotation, evidence)

    assert "中文参考答案" not in refined["reference"]
    assert "English and Chinese retrieval benchmarks" in refined["reference"]


def test_build_refined_annotation_does_not_require_body_evidence_when_only_metadata_exists():
    annotation = {
        "case_id": "Q0004",
        "user_input": "帮我读 arXiv:2604.00004《Metadata Only》。",
        "expected_source_ids": ["2604.00004"],
        "required_points": ["old"],
        "tags": {
            "evaluation_tracks": ["rag_retrieval", "answer_quality"],
            "source_type": "paper_qa",
            "reference_kind": "corpus_grounded_reference",
        },
    }
    evidence = {
        "2604.00004": PaperEvidence(
            paper_id="2604.00004",
            title="Metadata Only",
            abstract="This paper is available only through metadata in the local test fixture.",
            categories=["cs.IR"],
        )
    }

    refined = build_refined_annotation(annotation, evidence)

    assert "正文事实" not in refined["reference"]
    assert not any("正文证据" in point for point in refined["required_points"])
    assert any("摘要和元数据事实" in point for point in refined["required_points"])


def test_build_evidence_index_matches_tar_gz_source_by_paper_id(tmp_path: Path):
    metadata_root = tmp_path / "metadata" / "cs.AI"
    source_root = tmp_path / "source" / "cs.AI"
    metadata_root.mkdir(parents=True)
    source_root.mkdir(parents=True)
    (metadata_root / "2604.00001.json").write_text(
        json.dumps(
            {
                "paper_id": "2604.00001",
                "title": "Grounded Retrieval",
                "abstract": "This paper proposes grounded retrieval.",
                "categories": ["cs.IR"],
            }
        ),
        encoding="utf-8",
    )
    tex_path = tmp_path / "arxiv.tex"
    tex_path.write_text(
        r"\section{Methodology} The method combines dense and sparse retrieval.",
        encoding="utf-8",
    )
    with tarfile.open(source_root / "2604.00001.tar.gz", "w:gz") as archive:
        archive.add(tex_path, arcname="arxiv.tex")

    evidence = build_evidence_index(metadata_root.parent, source_root.parent)

    assert evidence["2604.00001"].source_extract.startswith("方法/理论依据")


def test_refine_annotations_updates_all_rag_retrieval_cases_and_leaves_non_rag_behavioral():
    annotations = [
        {
            "case_id": "Q0001",
            "user_input": "帮我读 arXiv:2604.00001《Grounded Retrieval》。",
            "expected_source_ids": ["2604.00001"],
            "reference": "理想回答应基于检索到的库内证据回答。",
            "required_points": ["old"],
            "tags": {
                "evaluation_tracks": ["rag_retrieval", "answer_quality"],
                "source_type": "paper_qa",
                "reference_kind": "corpus_grounded_reference",
            },
        },
        {
            "case_id": "Q0002",
            "user_input": "检索 arXiv:2501.99999《Missing Paper》。",
            "expected_source_ids": [],
            "reference": "理想回答应基于检索到的库内证据回答。",
            "required_points": ["old"],
            "tags": {
                "evaluation_tracks": ["rag_retrieval", "answer_quality"],
                "source_type": "paper_search",
                "reference_kind": "missing_corpus_reference",
                "explicit_arxiv_ids": ["2501.99999"],
            },
        },
        {
            "case_id": "Q0003",
            "user_input": "如果 cloudflared URL 变了，前端需要改哪里？",
            "expected_source_ids": [],
            "reference": "理想回答应依据 ScholarMind 项目文档给出可执行步骤。",
            "required_points": ["old"],
            "tags": {
                "evaluation_tracks": ["system_ops", "answer_quality"],
                "source_type": "system_ops",
                "reference_kind": "project_docs_verified",
            },
        },
    ]
    evidence = {
        "2604.00001": PaperEvidence(
            paper_id="2604.00001",
            title="Grounded Retrieval",
            abstract="This paper proposes a hybrid retrieval method.",
            categories=["cs.IR"],
            source_extract="方法/理论依据：The method combines dense and sparse retrieval.",
        )
    }

    refined = refine_annotations(annotations, evidence)

    assert "理想回答应" not in refined[0]["reference"]
    assert "当前库内没有可验证论文证据" in refined[1]["reference"]
    assert "标准答案" not in refined[1]["reference"]
    assert "理想回答应依据 ScholarMind" in refined[2]["reference"]
    assert json.dumps(refined, ensure_ascii=False)
