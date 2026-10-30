import os
import glob
import json
import time
import numpy as np
import torch
from tqdm import tqdm
from datasets import Dataset

# Import trực tiếp từ các submodule để Pylance/VSCode không báo lỗi "unknown import symbol"
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from sentence_transformers.sentence_transformer.losses import (
    MultipleNegativesRankingLoss,
)

from sentence_transformers.sentence_transformer.evaluation import (
    InformationRetrievalEvaluator,
)

from sentence_transformers.sentence_transformer.training_args import (
    BatchSamplers,
)

import faiss
from rank_bm25 import BM25Okapi

# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN & SIÊU THAM SỐ (TỐI ƯU CHO RTX 2050 4GB VRAM)
# ==============================================================================
DATA_DIR = r"D:\Hoc\trained-hybrid-retrieval-engine\artifacts\data"
ARROW_DIR = r"D:\Hoc\trained-hybrid-retrieval-engine\krylodar_stackoverflow"
OUTPUT_DIR = r"D:\Hoc\trained-hybrid-retrieval-engine\artifacts\models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dùng question_body làm Query và answer_body làm Passage theo đúng yêu cầu của bạn
QUERY_FIELD = "question_body"
PASSAGE_FIELD = "answer_body"

# Đổi tên file cache gắn với question_body để không bị nhầm với cache "title" cũ
CACHE_PATH = os.path.join(DATA_DIR, "id_text_lookup_question_body_cache.json")
BM25_CACHE_PATH = os.path.join(DATA_DIR, "bm25_test_rankings_cache.json")

BASE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 16              # An toàn tuyệt đối cho 4GB VRAM
EPOCHS = 1                   # Với ~15,000 queries, 1-2 epoch là hội tụ đẹp và nhanh trên RTX 2050
LR = 2e-5
NUM_NEGS_PER_QUERY = 2       # Lấy 2 negatives/query (~30,000 triplets) để train nhanh mà vẫn sắc bén

# ==============================================================================
# 2. ĐỌC DATA_*.JSONL & TRÍCH XUẤT TEXT TỪ 9 FILE ARROW (CÓ CACHE)
# ==============================================================================
def load_jsonl(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def build_or_load_lookup(train_data, val_data, test_data):
    if os.path.exists(CACHE_PATH):
        print(f"[*] Phát hiện file cache tại {CACHE_PATH}, đang nạp...")
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)
        q_lookup = {int(k): v for k, v in cache["q_lookup"].items()}
        p_lookup = {int(k): v for k, v in cache["p_lookup"].items()}
        print(f"[*] Đã nạp {len(q_lookup):,} queries (question_body) và {len(p_lookup):,} passages (answer_body).")
        return q_lookup, p_lookup

    needed_q_ids = set()
    needed_p_ids = set()
    for split in [train_data, val_data, test_data]:
        for row in split:
            needed_q_ids.add(int(row["query_id"]))
            needed_p_ids.add(int(row["positive_id"]))
            for nid in row.get("hard_negative_ids", []):
                needed_p_ids.add(int(nid))
            for nid in row.get("random_negative_ids", []):
                needed_p_ids.add(int(nid))

    print(f"[*] Cần trích xuất {len(needed_q_ids):,} unique question_ids và {len(needed_p_ids):,} unique answer_ids.")

    arrow_files = sorted(glob.glob(os.path.join(ARROW_DIR, "*.arrow")))
    if not arrow_files:
        raise FileNotFoundError(f"Không tìm thấy file .arrow nào trong {ARROW_DIR}")

    q_lookup = {}
    p_lookup = {}

    for fpath in arrow_files:
        print(f"[*] Đang quét {os.path.basename(fpath)}...")
        ds = Dataset.from_file(fpath)
        cols_to_select = ["question_id", "answer_id", QUERY_FIELD, PASSAGE_FIELD]
        ds_slim = ds.select_columns(cols_to_select)

        for row in tqdm(ds_slim, desc=os.path.basename(fpath), leave=False):
            qid = int(row["question_id"])
            aid = int(row["answer_id"])

            if qid in needed_q_ids and qid not in q_lookup:
                q_lookup[qid] = str(row[QUERY_FIELD]).strip()

            if aid in needed_p_ids and aid not in p_lookup:
                p_lookup[aid] = str(row[PASSAGE_FIELD]).strip()

        if len(q_lookup) >= len(needed_q_ids) and len(p_lookup) >= len(needed_p_ids):
            break

    print(f"[*] Hoàn tất quét Arrow! Thu được {len(q_lookup):,} queries và {len(p_lookup):,} passages.")
    print(f"[*] Đang lưu cache ra {CACHE_PATH}...")
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump({"q_lookup": q_lookup, "p_lookup": p_lookup}, f, ensure_ascii=False)

    return q_lookup, p_lookup

# ==============================================================================
# 3. TẠO HUGGINGFACE DATASET CHO TRAINER & VALIDATION EVALUATOR
# ==============================================================================
def create_hf_dataset(train_data, q_lookup, p_lookup, neg_type="hard"):
    """
    Trả về HuggingFace Dataset có 3 cột: ['anchor', 'positive', 'negative']
    tương thích trực tiếp với SentenceTransformerTrainer + MultipleNegativesRankingLoss.
    """
    neg_key = "hard_negative_ids" if neg_type == "hard" else "random_negative_ids"
    anchors, positives, negatives = [], [], []
    skipped = 0

    for item in train_data:
        qid = int(item["query_id"])
        pos_id = int(item["positive_id"])
        neg_ids = item.get(neg_key, [])[:NUM_NEGS_PER_QUERY]

        if qid not in q_lookup or pos_id not in p_lookup:
            skipped += 1
            continue

        q_text = q_lookup[qid]
        pos_text = p_lookup[pos_id]

        for nid in neg_ids:
            nid = int(nid)
            if nid in p_lookup and nid != pos_id:
                anchors.append(q_text)
                positives.append(pos_text)
                negatives.append(p_lookup[nid])

    ds = Dataset.from_dict({
        "anchor": anchors,
        "positive": positives,
        "negative": negatives
    })
    print(f"[*] Cấu hình [{neg_type.upper()} NEGATIVES]: Tạo thành công {len(ds):,} triplets (bỏ qua {skipped}).")
    return ds

def build_ir_evaluator(val_data, q_lookup, p_lookup, name="val", max_val_queries=500):
    queries = {}
    corpus = {}
    relevant_docs = {}

    # Giới hạn tối đa 500 câu Val trong lúc train để đánh giá giữa các epoch chỉ mất ~3 giây
    for item in val_data[:max_val_queries]:
        qid = int(item["query_id"])
        pos_id = int(item["positive_id"])
        if qid in q_lookup and pos_id in p_lookup:
            queries[str(qid)] = q_lookup[qid]
            corpus[str(pos_id)] = p_lookup[pos_id]
            relevant_docs[str(qid)] = {str(pos_id)}
            for nid in item.get("hard_negative_ids", []):
                nid = int(nid)
                if nid in p_lookup:
                    corpus[str(nid)] = p_lookup[nid]

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 5, 10],
        precision_recall_at_k=[10],
        show_progress_bar=False
    )

# ==============================================================================
# 4. HÀM FINE-TUNE BI-ENCODER BẰNG SENTENCETRANSFORMERTRAINER (CHUẨN V3+)
# ==============================================================================
def fine_tune_model(train_dataset, val_evaluator, output_path):
    # Nếu đã train xong trước đó thì bỏ qua để tiết kiệm thời gian
    if os.path.exists(os.path.join(output_path, "config.json")):
        print(f"[*] Đã tìm thấy model đã train tại {output_path}, bỏ qua bước train lại!")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(BASE_MODEL_NAME, device=device)
    model.max_seq_length = MAX_SEQ_LENGTH

    train_loss = MultipleNegativesRankingLoss(model=model, scale=20.0)

    steps_per_epoch = max(1, len(train_dataset) // BATCH_SIZE)
    eval_steps = max(100, steps_per_epoch // 2)

    args = SentenceTransformerTrainingArguments(
        output_dir=output_path,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LR,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=(device == "cuda"),
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # Tránh 2 triplet cùng query rơi vào 1 batch
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_{val_evaluator.name}_cosine_ndcg@10",
        logging_steps=50,
    )

    print(f"\n[*] Bắt đầu Fine-tune -> {output_path}")
    print(f"    - Device: {device} (FP16={args.fp16})")
    print(f"    - Triplets: {len(train_dataset):,} | Batch Size: {BATCH_SIZE} | Steps/Epoch: {steps_per_epoch}")

    t0 = time.time()
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=val_evaluator,
    )
    trainer.train()

    # Lưu bản tốt nhất ngay tại thư mục gốc output_path
    model.save_pretrained(output_path)
    print(f"[*] Hoàn tất train và lưu model tại {output_path} trong {(time.time() - t0)/60:.1f} phút!")

# ==============================================================================
# 5. HARNESS ĐÁNH GIÁ FULL PIPELINE (BM25 CÓ CACHE, DENSE, WEIGHTED HYBRID RRF)
# ==============================================================================
def calc_metrics(ranked_lists, true_ids, k=10):
    recalls, mrrs, ndcgs = [], [], []
    for preds, tid in zip(ranked_lists, true_ids):
        top_k = preds[:k]
        if tid in top_k:
            rank = top_k.index(tid) + 1
            recalls.append(1.0)
            mrrs.append(1.0 / rank)
            ndcgs.append(1.0 / np.log2(rank + 1))
        else:
            recalls.append(0.0)
            mrrs.append(0.0)
            ndcgs.append(0.0)
    return {
        "Recall@10": float(np.mean(recalls)),
        "MRR@10": float(np.mean(mrrs)),
        "NDCG@10": float(np.mean(ndcgs))
    }

def reciprocal_rank_fusion(dense_lists, bm25_lists, w_dense=0.7, w_bm25=0.3, k_rrf=60, top_n=100):
    fused = []
    for d_docs, b_docs in zip(dense_lists, bm25_lists):
        scores = {}
        for r, doc_id in enumerate(d_docs[:top_n], start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w_dense * (1.0 / (k_rrf + r))
        for r, doc_id in enumerate(b_docs[:top_n], start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w_bm25 * (1.0 / (k_rrf + r))
        ranked = [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
        fused.append(ranked)
    return fused

def get_or_compute_bm25(test_queries, corpus_ids, corpus_texts):
    if os.path.exists(BM25_CACHE_PATH):
        print(f"[*] Phát hiện cache BM25 tại {BM25_CACHE_PATH}, đang nạp (tốn 0s)...")
        with open(BM25_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if len(data.get("bm25_lists", [])) == len(test_queries):
            return data["bm25_lists"]

    print(f"[*] Đang xây dựng BM25 Index trên {len(corpus_texts):,} passages...")
    tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_lists = []
    for q in tqdm(test_queries, desc="BM25 Retrieval"):
        scores = bm25.get_scores(q.lower().split())
        # Dùng argpartition nhanh hơn argsort gấp 5 lần trên mảng lớn
        top_idx = np.argpartition(scores, -100)[-100:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        bm25_lists.append([int(corpus_ids[i]) for i in top_idx])

    with open(BM25_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump({"bm25_lists": bm25_lists}, f)
    print(f"[*] Đã lưu cache BM25 tại {BM25_CACHE_PATH}")
    return bm25_lists

def evaluate_model_on_test(model_name_or_path, test_queries, test_pos_ids, corpus_ids, corpus_texts, bm25_lists):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name_or_path, device=device)
    model.max_seq_length = MAX_SEQ_LENGTH

    c_embs = model.encode(
        corpus_texts,
        batch_size=128,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype("float32")

    q_embs = model.encode(
        test_queries,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")

    index = faiss.IndexFlatIP(c_embs.shape[1])
    index.add(c_embs)

    _, I = index.search(q_embs, 100)
    dense_lists = [[int(corpus_ids[idx]) for idx in row] for row in I]

    dense_metrics = calc_metrics(dense_lists, test_pos_ids, k=10)
    hybrid_lists = reciprocal_rank_fusion(dense_lists, bm25_lists, w_dense=0.7, w_bm25=0.3, k_rrf=60)
    hybrid_metrics = calc_metrics(hybrid_lists, test_pos_ids, k=10)

    return dense_metrics, hybrid_metrics, dense_lists, hybrid_lists

# ==============================================================================
# 6. CHƯƠNG TRÌNH CHÍNH (TRAIN TRƯỚC -> EVAL SAU)
# ==============================================================================
if __name__ == "__main__":
    train_data = load_jsonl(os.path.join(DATA_DIR, "data_train.jsonl"))
    val_data   = load_jsonl(os.path.join(DATA_DIR, "data_validate.jsonl"))
    test_data  = load_jsonl(os.path.join(DATA_DIR, "data_test.jsonl"))

    # Bước 1: Quét 9 file Arrow lấy question_body (query) và answer_body (passage)
    q_lookup, p_lookup = build_or_load_lookup(train_data, val_data, test_data)

    # Bước 2: Chuẩn bị Evaluator trên tập Validate
    val_evaluator = build_ir_evaluator(val_data, q_lookup, p_lookup, name="stackoverflow-val")

    # Bước 3: Train Run 2A - Random Negatives (Ablation) TRƯỚC để kiểm tra GPU ngay lập tức
    rand_model_dir = os.path.join(OUTPUT_DIR, "bi-encoder-random-neg")
    rand_ds = create_hf_dataset(train_data, q_lookup, p_lookup, neg_type="random")
    fine_tune_model(rand_ds, val_evaluator, rand_model_dir)

    # Bước 4: Train Run 2B - BM25 Hard Negatives (Main Model)
    hard_model_dir = os.path.join(OUTPUT_DIR, "bi-encoder-hard-neg")
    hard_ds = create_hf_dataset(train_data, q_lookup, p_lookup, neg_type="hard")
    fine_tune_model(hard_ds, val_evaluator, hard_model_dir)

    # Bước 5: Xây dựng Corpus Pool cho Test Evaluation
    # Chỉ lấy toàn bộ Positives + Hard Negatives + Random Negatives của tập Test (~15,000 - 25,000 passages)
    # để BM25 và FAISS chạy nhanh (chỉ ~2-3 phút) mà vẫn giữ nguyên độ khó
    test_corpus_set = set()
    valid_test = []
    for item in test_data:
        qid = int(item["query_id"])
        pid = int(item["positive_id"])
        if qid in q_lookup and pid in p_lookup:
            valid_test.append(item)
            test_corpus_set.add(pid)
            for nid in item.get("hard_negative_ids", []):
                if int(nid) in p_lookup:
                    test_corpus_set.add(int(nid))
            for nid in item.get("random_negative_ids", []):
                if int(nid) in p_lookup:
                    test_corpus_set.add(int(nid))

    corpus_ids = sorted(list(test_corpus_set))
    corpus_texts = [p_lookup[cid] for cid in corpus_ids]
    test_queries = [q_lookup[int(item["query_id"])] for item in valid_test]
    test_pos_ids = [int(item["positive_id"]) for item in valid_test]

    print(f"\n[*] Test Queries: {len(test_queries):,} | Test Evaluation Corpus Pool: {len(corpus_ids):,} passages")

    # Bước 6: Chạy BM25 (dùng argpartition + tự động lưu cache)
    bm25_lists = get_or_compute_bm25(test_queries, corpus_ids, corpus_texts)
    bm25_metrics = calc_metrics(bm25_lists, test_pos_ids, k=10)

    # Bước 7: Đánh giá Zero-shot vs. Random-Neg vs. Hard-Neg
    print("\n[*] Đang đánh giá Zero-shot Baseline...")
    zs_dense, zs_hybrid, _, _ = evaluate_model_on_test(
        BASE_MODEL_NAME, test_queries, test_pos_ids, corpus_ids, corpus_texts, bm25_lists
    )

    print("[*] Đang đánh giá Fine-tuned (Random Negatives)...")
    rn_dense, rn_hybrid, _, _ = evaluate_model_on_test(
        rand_model_dir, test_queries, test_pos_ids, corpus_ids, corpus_texts, bm25_lists
    )

    print("[*] Đang đánh giá Fine-tuned (BM25 Hard Negatives)...")
    hn_dense, hn_hybrid, hn_dense_lists, hn_hybrid_lists = evaluate_model_on_test(
        hard_model_dir, test_queries, test_pos_ids, corpus_ids, corpus_texts, bm25_lists
    )

    # Lưu kết quả ứng viên Top-100 của Hard-Neg Hybrid để Day 3 dùng cho Cross-Encoder
    day3_candidates_path = os.path.join(DATA_DIR, "day2_hybrid_candidates_for_day3.json")
    with open(day3_candidates_path, "w", encoding="utf-8") as f:
        json.dump({
            "test_query_ids": [int(item["query_id"]) for item in valid_test],
            "test_pos_ids": test_pos_ids,
            "hybrid_top100_ids": hn_hybrid_lists
        }, f)
    print(f"[*] Đã lưu Top-100 Hybrid Candidates cho Day 3 tại: {day3_candidates_path}")

    # In bảng kết quả
    print("\n" + "=" * 82)
    print("## DAY 2 RESULTS: BI-ENCODER FINE-TUNING & NEGATIVE MINING ABLATION")
    print("=" * 82)
    table_rows = [
        ("BM25 (Lexical Baseline)", bm25_metrics),
        ("Dense Off-the-shelf (`all-MiniLM-L6-v2`)", zs_dense),
        ("Hybrid Untrained (BM25 + Zero-shot Dense RRF)", zs_hybrid),
        ("Dense Fine-tuned (MNRL + Random Negatives)", rn_dense),
        ("Dense Fine-tuned (MNRL + BM25 Hard Negatives)", hn_dense),
        ("Hybrid Fine-tuned (BM25 + Hard-Neg Dense RRF)", hn_hybrid),
    ]
    print(f"| {'Pipeline / Model Configuration':<47} | {'Recall@10':^9} | {'MRR@10':^8} | {'NDCG@10':^8} |")
    print(f"| :{'-'*46} | :{'-'*7}: | :{'-'*6}: | :{'-'*6}: |")
    for label, m in table_rows:
        print(f"| {label:<47} | {m['Recall@10']:^9.4f} | {m['MRR@10']:^8.4f} | {m['NDCG@10']:^8.4f} |")