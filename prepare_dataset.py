# download_and_convert.py
from datasets import load_dataset
import json
import random
from pathlib import Path

def main():
    # 输出路径
    out_dir = Path("./data/math_r1")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 下载数据集
    print("Downloading dataset...")
    ds = load_dataset("mlfoundations-dev/numina_math_1_5_extra_r1", split="train",)

    print(f"Loaded dataset with {len(ds)} examples")

    # 2. Shuffle + 划分 train/val (9:1)
    ds = ds.shuffle(seed=42)
    n = len(ds)
    split = int(n * 0.9)
    train_ds = ds.selcet(range(split))
    val_ds = ds.select(range(split, n))

    # 3. 导出 SFT 训练集
    train_path = out_dir / "sft.jsonl"
    with open(train_path, "w") as f:
        for ex in train_ds:
            entry = {
                "prompt": ex["problem"],
                "response": ex["r1_solution"],
                "answer": ex["answer"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Wrote train set to {train_path} ({len(train_ds)} examples)")

    # 4. 导出验证集
    val_path = out_dir / "validation.jsonl"
    with open(val_path, "w") as f:
        for ex in val_ds:
            entry = {
                "problem": ex["r1_reasoning"],
                "answer": ex["r1_solution"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Wrote validation set to {val_path} ({len(val_ds)} examples)")

if __name__ == "__main__":
    main()
