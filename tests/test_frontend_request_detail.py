from pathlib import Path


def test_request_detail_rag_data_shows_caller_agent_without_events_table():
    app_js = Path("static/js/app.js").read_text(encoding="utf-8")

    rag_section = app_js.split("// --- Dimension 3: RAG Data", 1)[1].split(
        "// --- Dimension 4: Evaluation", 1
    )[0]

    assert "Caller Agent" in rag_section
    assert "renderDimEventsTable(ragEvents" not in rag_section
    assert "{ key: 'caller_agent', label: 'Caller' }" not in rag_section


def test_request_detail_hides_unused_rag_and_memory_sections():
    app_js = Path("static/js/app.js").read_text(encoding="utf-8")

    memory_section = app_js.split("// --- Dimension 2: Memory Data", 1)[1].split(
        "// --- Dimension 3: RAG Data", 1
    )[0]
    rag_section = app_js.split("// --- Dimension 3: RAG Data", 1)[1].split(
        "// --- Dimension 4: Evaluation", 1
    )[0]

    assert "if (hasMemoryV2Data) {" in memory_section
    assert "No V2 memory data available for this request." not in memory_section
    assert "if (hasRag) {" in rag_section


def test_request_overview_health_score_reads_top_level_field_first():
    app_js = Path("static/js/app.js").read_text(encoding="utf-8")

    overview_section = app_js.split("// --- Dimension 1: Request Overview ---", 1)[1].split(
        "// --- Dimension 2: Memory Data", 1
    )[0]

    assert "const healthScore = evalData.execution_health_score ?? eh.execution_health_score;" in overview_section
    assert "{ label: 'Health Score', value: healthScore, fmt: 'score' }" in overview_section
    assert "{ label: 'Health Score', value: eh.execution_health_score" not in overview_section


def test_dashboard_memory_and_answer_cards_use_matching_dom_ids():
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
    assert "setScoreCard('stat-memory-score', d.avg_memory_score);" in stats_section
    assert "setScoreCard('stat-answer-quality', d.avg_answer_quality_score);" in stats_section
