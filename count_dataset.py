import os
import glob
from datasets import Dataset, concatenate_datasets

ARROW_DIR = r"D:\Hoc\trained-hybrid-retrieval-engine\krylodar_stackoverflow"

arrow_files = sorted(glob.glob(os.path.join(ARROW_DIR, "*.arrow")))
total_rows = 0

print(f"{'Tên file':<45} | {'Số lượng dòng':>15}")
print("-" * 63)

datasets_list = []
for fpath in arrow_files:
    ds = Dataset.from_file(fpath)
    num_rows = len(ds)
    total_rows += num_rows
    datasets_list.append(ds)
    print(f"{os.path.basename(fpath):<45} | {num_rows:>15,}")

print("-" * 63)
print(f"{'TỔNG CỘNG (9 files):':<45} | {total_rows:>15,}")

# (Tùy chọn) Đếm thêm số lượng unique question_id và answer_id nếu cần đưa vào README
full_ds = concatenate_datasets(datasets_list)
unique_questions = len(set(full_ds["question_id"]))
unique_answers = len(set(full_ds["answer_id"]))
print(f"\n[*] Unique Questions: {unique_questions:,}")
print(f"[*] Unique Answers  : {unique_answers:,}")