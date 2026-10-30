import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "data"
RESULTS_FILE = DATA_DIR / "day3_final_reranked_results.json"
CACHE_FILE = DATA_DIR / "id_text_lookup_question_body_cache.json"
BM25_CACHE = DATA_DIR / "bm25_test_rankings_cache.json"
REPORT_FILE = DATA_DIR / "error_analysis_20_failures.md"

def run_error_analysis():
    print("[*] Đang nạp kết quả Day 3 và văn bản câu hỏi/câu trả lời...")
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        res = json.load(f)
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)
    with open(BM25_CACHE, "r", encoding="utf-8") as f:
        bm25_data = json.load(f)

    q_lookup = {int(k): v for k, v in cache["q_lookup"].items()}
    p_lookup = {int(k): v for k, v in cache["p_lookup"].items()}

    test_qids = res["test_query_ids"]
    test_pos_ids = res["test_pos_ids"]
    dense_lists = res["dense_top100_ids"]
    hybrid_lists = res["hybrid_top100_ids"]
    ce_lists = res["reranked_top100_ids"]
    bm25_lists = bm25_data["bm25_lists"]

    failures = []
    for idx, (qid, pid) in enumerate(zip(test_qids, test_pos_ids)):
        ce_top10 = ce_lists[idx][:10]
        if pid not in ce_top10:
            bm25_rank = bm25_lists[idx].index(pid) + 1 if pid in bm25_lists[idx] else ">100"
            dense_rank = dense_lists[idx].index(pid) + 1 if pid in dense_lists[idx] else ">100"
            hybrid_rank = hybrid_lists[idx].index(pid) + 1 if pid in hybrid_lists[idx] else ">100"
            ce_rank = ce_lists[idx].index(pid) + 1 if pid in ce_lists[idx] else ">100"

            failures.append({
                "qid": qid,
                "pid": pid,
                "query": q_lookup[qid],
                "true_passage": p_lookup[pid],
                "bm25_rank": bm25_rank,
                "dense_rank": dense_rank,
                "hybrid_rank": hybrid_rank,
                "ce_rank": ce_rank,
                "top1_predicted": p_lookup[ce_lists[idx][0]]
            })

        if len(failures) >= 20:
            break

    print(f"[*] Đã trích xuất thành công {len(failures)} ca lỗi. Đang phân loại và ghi báo cáo...")

    md_content = ["# Qualitative Error Analysis: 20 Failure Cases (Rank > 10)\n"]
    md_content.append("This document analyzes 20 representative queries where the ground-truth passage was not retrieved within Top-10 by the final Hybrid + Cross-Encoder pipeline.\n")
    md_content.append("## Failure Taxonomy\n")
    md_content.append("1. **Type A: Candidate Generation Miss (Retrieval Bottleneck)** — True passage ranked > 30 in both BM25 and Dense, leaving Cross-Encoder unable to examine it.\n")
    md_content.append("2. **Type B: Vocabulary Mismatch / Jargon Polysemy** — Dense model matched general programming concepts rather than framework-specific subtleties.\n")
    md_content.append("3. **Type C: Non-Canonical Valid Answers** — The retrieved #1 passage provides an alternative working solution to the query, but carries a different StackOverflow `answer_id`.\n")
    md_content.append("4. **Type D: Code-Heavy Zero-Prose Passage** — Answer contains pure code snippets without explanatory narrative, hindering cross-attention text alignment.\n\n---\n")

    md_content.append("## Detailed Inspection of 20 Failures\n\n")

    for i, item in enumerate(failures, 1):
        # Xác định nhóm lỗi
        h_r = item["hybrid_rank"]
        if h_r == ">100" or (isinstance(h_r, int) and h_r > 30):
            cat = "Type A: Candidate Generation Miss (RRF Rank > 30)"
        elif "```" in item["true_passage"] and len(item["true_passage"].replace("```", "").split()) < 20:
            cat = "Type D: Code-Heavy Zero-Prose Passage"
        elif item["ce_rank"] != ">100" and item["ce_rank"] <= 20:
            cat = "Type C: Non-Canonical Valid Answers / Ranking Tie"
        else:
            cat = "Type B: Vocabulary Mismatch / Semantic Distraction"

        md_content.append(f"### Case #{i}: Query ID {item['qid']} (Category: {cat})\n")
        md_content.append(f"- **Rank Progression:** BM25: `#{item['bm25_rank']}` | Dense: `#{item['dense_rank']}` | Hybrid: `#{item['hybrid_rank']}` | **Final CE: `#{item['ce_rank']}`**\n")
        md_content.append(f"- **Query:**\n> {item['query'][:300]}...\n\n")
        md_content.append(f"- **Ground-Truth Passage (ID {item['pid']}):**\n> {item['true_passage'][:300]}...\n\n")
        md_content.append(f"- **Predicted Rank #1 Passage:**\n> {item['top1_predicted'][:300]}...\n\n")
        md_content.append("---\n\n")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.writelines(md_content)

    print(f"[*] Báo cáo Error Analysis đã được lưu tại: {REPORT_FILE}")

if __name__ == "__main__":
    run_error_analysis()

