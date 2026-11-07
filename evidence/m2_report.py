"""Tables and the plot of Meeting 2, from the files that the runs saved.

uv run python evidence/m2_report.py val
uv run python evidence/m2_report.py test
uv run python evidence/m2_report.py sheet   # same seeds side by side in every pool
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw

from synthdata.config import CLASSES, EVIDENCE_DIR, RUNS_DIR, TARGET, TARGET_INDEX
from synthdata.pools import POOLS_DIR

VAL_RUNS = {
    "m1-oversample": "oversample",
    "m2-base": "base",
    "m2-guidance-3": "guidance 3",
    "m2-guidance-12": "guidance 12",
    "m2-steps-15": "steps 15",
    "m2-one-prompt": "one prompt",
    "m2-sdxl-turbo": "sdxl-turbo",
}
TEST_RUNS = {
    "m1-real": "real",
    "m1-oversample": "oversample",
    "m1-augment": "augment",
    "m2-base": "base",
}
REFERENCE = "oversample"
POOLS = [run for run in VAL_RUNS if run.startswith("m2-")]
INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
REFERENCE_COLOUR, SYNTHETIC_COLOUR, OTHER_COLOUR = "#eb6834", "#2a78d6", "#898781"
OTHERS = [breed for breed in CLASSES if breed != TARGET]


def colour(method: str) -> str:
    if method == REFERENCE:
        return REFERENCE_COLOUR
    if method in ("real", "augment"):
        return OTHER_COLOUR
    return SYNTHETIC_COLOUR


def load(split: str, runs: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results_name = "results.json" if split == "val" else "test-results.json"
    metrics_name = "metrics.json" if split == "val" else "test-metrics.json"
    predictions_name = "predictions.csv" if split == "val" else "test-predictions.csv"
    seeds, confusions, predictions = [], [], []
    for run, method in runs.items():
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
                    "bengal_recall": entry["recall"]["Bengal"],
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


def paired_differences(seeds: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for first, second in pairs:
        for metric in ("mau_recall", "macro_f1", "other_recall", "bengal_recall"):
            a = seeds[seeds["method"] == first].set_index("seed")[metric]
            b = seeds[seeds["method"] == second].set_index("seed")[metric]
            difference = (a - b).round(6)
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


def plot(seeds: pd.DataFrame, methods: list[str], split: str) -> None:
    panels = [
        ("mau_recall", "Egyptian Mau recall"),
        ("macro_f1", "Macro-F1"),
        ("bengal_recall", "Bengal recall"),
    ]
    width = max(12, 2.2 * len(methods) * 1.5)
    figure, axes = plt.subplots(1, 3, figsize=(width, 4.8), facecolor=SURFACE)
    for axis, (metric, title) in zip(axes, panels, strict=True):
        axis.set_facecolor(SURFACE)
        for seed, group in seeds.groupby("seed"):
            shift = (seed - 2) * 0.05
            group = group.set_index("method").loc[methods]
            x = [index + shift for index in range(len(methods))]
            axis.scatter(
                x,
                group[metric],
                s=40,
                c=[colour(method) for method in methods],
                edgecolors=SURFACE,
                linewidths=1.5,
                zorder=2,
            )
        reference = seeds.loc[seeds["method"] == REFERENCE, metric].mean()
        axis.axhline(reference, color=REFERENCE_COLOUR, linewidth=1, linestyle="--")
        for index, method in enumerate(methods):
            mean = seeds.loc[seeds["method"] == method, metric].mean()
            axis.hlines(mean, index - 0.22, index + 0.22, color=INK, linewidth=2, zorder=3)
            axis.annotate(
                f"{mean:.3f}",
                (index, mean),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=INK,
            )
        axis.set_xticks(range(len(methods)), methods, color=SECONDARY, rotation=30, ha="right")
        axis.set_xlim(-0.5, len(methods) - 0.5)
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
        f"Meeting 2 on the {name} photos: one dot per seed, black bar = mean, "
        f"dashed line = {REFERENCE} mean",
        x=0.01,
        ha="left",
        color=SECONDARY,
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(EVIDENCE_DIR / f"m2-{split}-results.png", dpi=160)


def sheet() -> None:
    """The same generation seeds in every pool, one row per pool."""
    samples = [0, 1, 2, 3, 15, 43, 77, 103]
    size, label = 160, 22
    image = Image.new(
        "RGB", (label * 6 + size * len(samples), size * len(POOLS) + label), "white"
    )
    draw = ImageDraw.Draw(image)
    left = label * 6
    for column, sample in enumerate(samples):
        draw.text((left + column * size + 4, 4), f"sample {sample:04d}", fill=INK)
    for row, pool in enumerate(POOLS):
        top = label + row * size
        draw.text((4, top + size // 2), pool.removeprefix("m2-"), fill=INK)
        records = {
            int(r["sample_id"].rsplit("-", 1)[1]): r
            for r in map(json.loads, (POOLS_DIR / pool / "manifest.jsonl").open())
        }
        for column, sample in enumerate(samples):
            record = records[sample]
            path = POOLS_DIR / pool / record["file"]
            box = (left + column * size, top)
            if path.exists():
                image.paste(Image.open(path).convert("RGB").resize((size, size)), box)
            if record["status"] != "ok":
                draw.rectangle(
                    [box[0], box[1], box[0] + size - 1, box[1] + size - 1],
                    outline="red",
                    width=4,
                )
    image.save(EVIDENCE_DIR / "m2-pools-same-seeds.jpg", quality=88)


def main() -> None:
    split = sys.argv[1]
    if split == "sheet":
        sheet()
        return
    runs = VAL_RUNS if split == "val" else TEST_RUNS
    order = list(runs.values())
    seeds, confusions, predictions = load(split, runs)

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
            bengal_recall=("bengal_recall", "mean"),
            mau_precision=("mau_precision", "mean"),
            false_mau=("false_mau", "mean"),
        )
        .loc[order]
    )
    pairs = [(method, REFERENCE) for method in order if method != REFERENCE]
    if split == "val":
        pairs += [(method, "base") for method in order if method not in ("base", REFERENCE)]
    differences = paired_differences(seeds, pairs)
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
    photos = photos.sort_values(order[::-1])

    summary.to_csv(EVIDENCE_DIR / f"m2-{split}-summary.csv")
    seeds.to_csv(EVIDENCE_DIR / f"m2-{split}-seeds.csv", index=False)
    differences.to_csv(EVIDENCE_DIR / f"m2-{split}-differences.csv", index=False)
    breeds.to_csv(EVIDENCE_DIR / f"m2-{split}-breed-recall.csv")
    photos.to_csv(EVIDENCE_DIR / f"m2-{split}-mau-photos.csv")
    plot(seeds, order, split)

    pd.set_option("display.width", 200, "display.max_columns", 40)
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
