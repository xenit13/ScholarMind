from pathlib import Path


def test_request_detail_does_not_render_rag_data_section():
    app_js = Path("static/js/app.js").read_text(encoding="utf-8")

    assert "// --- Dimension 3: RAG Data" not in app_js
    assert "ragEvents" not in app_js
    assert "hasRag" not in app_js
    assert "rag_metrics" not in app_js


def test_request_detail_hides_unused_memory_section():
    app_js = Path("static/js/app.js").read_text(encoding="utf-8")

    memory_section = app_js.split("// --- Dimension 2: Memory Data", 1)[1].split(
        "// --- Dimension 3: Evaluation", 1
    )[0]

    assert "if (hasMemoryV2Data) {" in memory_section
    assert "No V2 memory data available for this request." not in memory_section


def test_request_overview_health_score_reads_top_level_field_first():
    app_js = Path("static/js/app.js").read_text(encoding="utf-8")

    overview_section = app_js.split("// --- Dimension 1: Request Overview ---", 1)[1].split(
        "// --- Dimension 2: Memory Data", 1
    )[0]

    health_score_line = (
        "const healthScore = evalData.execution_health_score ?? eh.execution_health_score;"
    )
    assert health_score_line in overview_section
    assert "{ label: 'Health Score', value: healthScore, fmt: 'score' }" in overview_section
    assert "{ label: 'Health Score', value: eh.execution_health_score" not in overview_section
    assert "Faithfulness" not in overview_section


def test_dashboard_has_memory_and_answer_cards_without_rag_card():
    index_html = Path("static/index.html").read_text(encoding="utf-8")
    app_js = Path("static/js/app.js").read_text(encoding="utf-8")

    memory_card = index_html.split('<div class="stat-title">Memory Score</div>', 1)[1].split(
        "</div>\n              <div class=\"stat-card\">",
        1,
    )[0]
    answer_card = index_html.split('<div class="stat-title">Answer Score</div>', 1)[1].split(
        "</div>\n            </div>",
        1,
    )[0]
    stats_section = app_js.split("function renderAdminStats(resp)", 1)[1].split(
        "const avgLatencyEl",
        1,
    )[0]

    assert 'id="stat-memory-score"' in memory_card
    assert 'id="stat-answer-quality"' in answer_card
    assert 'id="stat-rag-score"' not in index_html
    assert 'data-sort="rag_score"' not in index_html
    assert "setScoreCard('stat-memory-score', d.avg_memory_score);" in stats_section
    assert "setScoreCard('stat-answer-quality', d.avg_answer_quality_score);" in stats_section
    assert "avg_rag_score" not in app_js


def test_frontend_does_not_expose_removed_research_workflows():
    index_html = Path("static/index.html").read_text(encoding="utf-8")
    app_js = Path("static/js/app.js").read_text(encoding="utf-8")
    combined = index_html + app_js

    forbidden_snippets = [
        "/research",
        "Paper Q&A",
        "Idea Novelty",
        "Trend Analysis",
        "Cross-Domain",
        "Study Plan",
        "Paper Reading",
        "paper_reading",
        "idea_novelty",
        "cross_domain",
    ]
    present = sorted(snippet for snippet in forbidden_snippets if snippet in combined)

    assert present == []
