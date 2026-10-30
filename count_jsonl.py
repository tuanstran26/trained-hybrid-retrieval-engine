import os
import json

DATA_DIR = r"D:\Hoc\trained-hybrid-retrieval-engine\artifacts\data"
FILES = ["data_train.jsonl", "data_validate.jsonl", "data_test.jsonl"]

print(f"{'Tên file':<22} | {'Số Queries':>12} | {'Tổng Hard Negs':>15} | {'Tổng Rand Negs':>15} | {'Unique Passages':>16}")
print("-" * 92)

total_queries = 0
total_hard = 0
total_rand = 0
all_unique_passages = set()

for fname in FILES:
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath):
        print(f"{fname:<22} | {'(Không tìm thấy file)':>66}")
        continue

    q_count = 0
    h_count = 0
    r_count = 0
    split_passages = set()

    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            q_count += 1
            
            pos_id = int(item["positive_id"])
            h_ids = [int(x) for x in item.get("hard_negative_ids", [])]
            r_ids = [int(x) for x in item.get("random_negative_ids", [])]

            h_count += len(h_ids)
            r_count += len(r_ids)

            split_passages.add(pos_id)
            split_passages.update(h_ids)
            split_passages.update(r_ids)

    total_queries += q_count
    total_hard += h_count
    total_rand += r_count
    all_unique_passages.update(split_passages)

    print(f"{fname:<22} | {q_count:>12,} | {h_count:>15,} | {r_count:>15,} | {len(split_passages):>16,}")

print("-" * 92)
print(f"{'TỔNG CỘNG (3 files):':<22} | {total_queries:>12,} | {total_hard:>15,} | {total_rand:>15,} | {len(all_unique_passages):>16,}")