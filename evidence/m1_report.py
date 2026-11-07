"""Tables and the plot of Meeting 1, from the files that the runs saved.

uv run python evidence/m1_report.py val
uv run python evidence/m1_report.py test
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from synthdata.config import CLASSES, EVIDENCE_DIR, RUNS_DIR, TARGET, TARGET_INDEX

RUNS = {"m1-real": "real", "m1-oversample": "oversample", "m1-augment": "augment"}
COLOURS = {"real": "#2a78d6", "oversample": "#eb6834", "augment": "#1baf7a"}
INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
OTHERS = [breed for breed in CLASSES if breed != TARGET]


def load(split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results_name = "results.json" if split == "val" else "test-results.json"
    metrics_name = "metrics.json" if split == "val" else "test-metrics.json"
    predictions_name = "predictions.csv" if split == "val" else "test-predictions.csv"
    seeds, confusions, predictions = [], [], []
    for run, method in RUNS.items():
        results = json.loads((RUNS_DIR / run / results_name).read_text())
        for entry in results["seeds"]:
            seed = entry["seed"]
            folder = RUNS_DIR / run / f"seed-{seed}"
            matrix = json.loads((folder / metrics_name).read_text())["confusion"]
            added_to_mau = sum(
                row[TARGET_INDEX]
                for index, row in enumerate(matrix)
                if index != TARGET_INDEX
            )
            found = matrix[TARGET_INDEX][TARGET_INDEX]
            seeds.append(
                {
                    "method": method,
                    "seed": seed,
                    "mau_recall": entry["egyptian_mau_recall"],
                    "macro_f1": entry["macro_f1"],
                    "accuracy": entry["accuracy"],
                    "other_recall": sum(entry["recall"][b] for b in OTHERS) / 7,
                    "mau_precision": found / (found + added_to_mau),
                    "false_mau": added_to_mau,
                }
            )
            for index, breed in enumerate(CLASSES):
                confusions.append(
                    {
                        "method": method,
                        "seed": seed,
                        "breed": breed,
                        "mau_called": matrix[TARGET_INDEX][index],
                        "called_mau": matrix[index][TARGET_INDEX],
                        "recall": entry["recall"][breed],
                    }
                )
            table = pd.read_csv(folder / predictions_name)
            table = table[table["breed"] == TARGET].assign(method=method, seed=seed)
            predictions.append(table)
    return pd.DataFrame(seeds), pd.DataFrame(confusions), pd.concat(predictions)


def paired_differences(seeds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pairs = [
        ("oversample", "real"),
        ("augment", "real"),
        ("augment", "oversample"),
    ]
    for first, second in pairs:
        for metric in ("mau_recall", "macro_f1", "other_recall"):
            a = seeds[seeds["method"] == first].set_index("seed")[metric]
            b = seeds[seeds["method"] == second].set_index("seed")[metric]
            difference = a - b
            rows.append(
                {
                    "comparison": f"{first} - {second}",
                    "metric": metric,
                    "mean": difference.mean(),
                    "min": difference.min(),
                    "max": difference.max(),
                    "seeds_up": int((difference > 0).sum()),
                    "seeds_tied": int((difference == 0).sum()),
                    "seeds_down": int((difference < 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def plot(seeds: pd.DataFrame, split: str) -> None:
    methods = list(RUNS.values())
    panels = [("mau_recall", "Egyptian Mau recall"), ("macro_f1", "Macro-F1")]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.6), facecolor=SURFACE)
    for axis, (metric, title) in zip(axes, panels, strict=True):
        axis.set_facecolor(SURFACE)
        for seed, group in seeds.groupby("seed"):
            shift = (seed - 2) * 0.05
            group = group.set_index("method").loc[methods]
            x = [index + shift for index in range(len(methods))]
            axis.plot(x, group[metric], color=GRID, linewidth=1.5, zorder=1)
            axis.scatter(
                x,
                group[metric],
                s=64,
                c=[COLOURS[method] for method in methods],
                edgecolors=SURFACE,
                linewidths=2,
                zorder=2,
            )
        for index, method in enumerate(methods):
            mean = seeds.loc[seeds["method"] == method, metric].mean()
            axis.hlines(
                mean, index - 0.2, index + 0.2, color=INK, linewidth=2, zorder=3
            )
            axis.annotate(
                f"mean {mean:.3f}",
                (index + 0.23, mean),
                va="center",
                fontsize=9,
                color=INK,
            )
        axis.set_xticks(range(len(methods)), methods, color=SECONDARY)
        axis.set_xlim(-0.4, len(methods) - 0.2)
        axis.set_title(title, loc="left", color=INK, fontsize=12)
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", colors=MUTED, length=0)
        axis.tick_params(axis="x", length=0)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color(BASELINE)
    name = "validation" if split == "val" else "test"
    figure.suptitle(
        f"Meeting 1 baselines on the {name} photos: one dot per seed, "
        "lines join the same seed",
        x=0.01,
        ha="left",
        color=SECONDARY,
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(EVIDENCE_DIR / f"m1-{split}-results.png", dpi=160)


def main() -> None:
    split = sys.argv[1]
    seeds, confusions, predictions = load(split)
    order = list(RUNS.values())

    wide = seeds.pivot(index="seed", columns="method")[["mau_recall", "macro_f1"]]
    summary = (
        seeds.groupby("method")
        .agg(
            mau_recall=("mau_recall", "mean"),
            mau_recall_sd=("mau_recall", "std"),
            macro_f1=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            accuracy=("accuracy", "mean"),
            other_recall=("other_recall", "mean"),
            mau_precision=("mau_precision", "mean"),
            false_mau=("false_mau", "mean"),
        )
        .loc[order]
    )
    differences = paired_differences(seeds)
    breeds = confusions.pivot_table(
        index="breed", columns="method", values="recall", aggfunc="mean"
    )[order].loc[CLASSES]
    mau_called = confusions.pivot_table(
        index="breed", columns="method", values="mau_called", aggfunc="mean"
    )[order].loc[CLASSES]
    called_mau = confusions.pivot_table(
        index="breed", columns="method", values="called_mau", aggfunc="mean"
    )[order].loc[CLASSES]
    predictions["right"] = predictions["predicted"] == TARGET
    photos = predictions.pivot_table(
        index="image", columns="method", values="right", aggfunc="sum"
    )[order]
    photos["all"] = photos.sum(axis=1)
    photos = photos.sort_values("all")

    summary.to_csv(EVIDENCE_DIR / f"m1-{split}-summary.csv")
    seeds.to_csv(EVIDENCE_DIR / f"m1-{split}-seeds.csv", index=False)
    differences.to_csv(EVIDENCE_DIR / f"m1-{split}-differences.csv", index=False)
    breeds.to_csv(EVIDENCE_DIR / f"m1-{split}-breed-recall.csv")
    photos.to_csv(EVIDENCE_DIR / f"m1-{split}-mau-photos.csv")
    plot(seeds, split)

    pd.set_option("display.width", 160, "display.max_columns", 30)
    pd.set_option("display.float_format", "{:.3f}".format)
    sections = [
        ("per seed", wide),
        ("summary", summary),
        ("paired differences", differences),
        ("recall per breed", breeds),
        ("Mau photos predicted as (mean per seed)", mau_called),
        ("photos of each breed predicted as Mau", called_mau),
        ("right predictions per Mau photo, out of 5 per method", photos),
    ]
    for title, table in sections:
        print(f"\n## {title}\n{table.to_string()}")


if __name__ == "__main__":
    main()
