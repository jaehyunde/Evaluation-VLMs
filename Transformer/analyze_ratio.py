"""
analyze_ratio_sweep.py
-----------------------------------
- 특정 W&B sweep의 결과를 모아서
  alpha별 test F1-score 평균/표준편차 계산
- x축=alpha, y축=mean F1 인 그래프 저장
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import wandb

# 🔴 여기를 네 sweep ID로 수정해야 함
# 예: "autoshin0322-goethe-university-frankfurt/RatioTrain/hcqxew0r"
SWEEP_PATH = "autoshin0322-goethe-university-frankfurt/RatioTrain/hcqxew0r"

OUT_DIR = "output_ratio"
os.makedirs(os.path.join(OUT_DIR, "reports"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "plots"), exist_ok=True)


def main():
    api = wandb.Api()

    sweep = api.sweep(SWEEP_PATH)
    runs = sweep.runs

    records = []
    for run in runs:
        cfg = run.config
        summary = run.summary

        alpha = cfg.get("alpha", None)
        run_id = cfg.get("run_id", None)

        # 여기서는 test_f1을 기준으로 평균 계산 (필요하면 val_f1로 바꿔도 됨)
        test_f1 = summary.get("test_f1", None)

        if alpha is None or test_f1 is None:
            continue

        records.append({
            "run_name": run.name,
            "alpha": float(alpha),
            "run_id": int(run_id) if run_id is not None else None,
            "test_f1": float(test_f1),
        })

    if not records:
        print("⚠️ 수집된 run이 없습니다. SWEEP_PATH를 확인하세요.")
        return

    df = pd.DataFrame(records)
    df.sort_values(["alpha", "run_id"], inplace=True)

    # alpha별 평균/표준편차 계산
    grp = df.groupby("alpha")["test_f1"].agg(["mean", "std", "count"]).reset_index()
    grp.rename(columns={
        "mean": "f1_mean",
        "std": "f1_std",
        "count": "n_runs"
    }, inplace=True)

    # CSV 저장
    all_path = os.path.join(OUT_DIR, "reports", "alpha_f1_all_runs.csv")
    summary_path = os.path.join(OUT_DIR, "reports", "alpha_f1_summary.csv")

    df.to_csv(all_path, index=False)
    grp.to_csv(summary_path, index=False)

    print("\n✅ 모든 run 기록")
    print(df)
    print("\n✅ alpha별 평균/표준편차")
    print(grp)
    print(f"\n📄 저장: {all_path}")
    print(f"📄 저장: {summary_path}")

    # 그래프 그리기 (x축=alpha, y축=mean F1)
    plt.figure()
    plt.errorbar(
        grp["alpha"],
        grp["f1_mean"],
        yerr=grp["f1_std"],
        fmt="-o",
    )
    plt.xlabel("alpha")
    plt.ylabel("test F1-score (mean ± std)")
    plt.title("Test F1 vs alpha")
    plt.grid(True)

    plot_path = os.path.join(OUT_DIR, "plots", "alpha_vs_test_f1.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"🖼 그래프 저장: {plot_path}")


if __name__ == "__main__":
    main()
