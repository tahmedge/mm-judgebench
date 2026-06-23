#!/usr/bin/env python3
"""Evaluate A/B judge predictions against an exported dataset CSV."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def parse_winner(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value).strip()
    if not text:
        return ""
    upper = text.upper()
    if upper in {"A", "B"}:
        return upper

    stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
    stripped = re.sub(r"\s*```$", "", stripped).strip()
    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            winner = str(data.get("winner", "")).strip().upper()
            if winner in {"A", "B"}:
                return winner
    except json.JSONDecodeError:
        pass

    match = re.search(r'"winner"\s*:\s*"([ABab])"', text)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b(?:winner|answer)\b\s*[:=]?\s*([ABab])\b", text, re.I)
    if match:
        return match.group(1).upper()
    match = re.match(r"^\s*([ABab])(?=[^A-Za-z]|$)", text)
    return match.group(1).upper() if match else ""


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "winner" not in df.columns:
        raise ValueError("Predictions CSV must contain a 'winner' column.")
    df["prediction"] = df["winner"].map(parse_winner)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Exported original.csv or reversed.csv.")
    parser.add_argument("--predictions", required=True, type=Path, help="CSV with id/language/winner predictions.")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_results"))
    args = parser.parse_args()

    dataset = pd.read_csv(args.dataset)
    predictions = load_predictions(args.predictions)
    required = {"id", "language", "label"}
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")
    missing = {"id", "language"} - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions missing columns: {sorted(missing)}")

    merged = dataset[["id", "language", "label"]].merge(
        predictions,
        on=["id", "language"],
        how="left",
        suffixes=("", "_prediction_file"),
    )
    merged["label"] = merged["label"].astype(str).str.strip().str.upper()
    merged["correct"] = merged["prediction"].eq(merged["label"])
    merged["parse_success"] = merged["prediction"].isin(["A", "B"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_dir / "merged_predictions.csv", index=False)

    overall = pd.DataFrame(
        [
            {
                "scope": "overall",
                "n": len(merged),
                "answered": int(merged["parse_success"].sum()),
                "correct": int(merged["correct"].sum()),
                "accuracy": float(merged["correct"].mean()) if len(merged) else 0.0,
                "parse_rate": float(merged["parse_success"].mean()) if len(merged) else 0.0,
            }
        ]
    )
    by_language = (
        merged.groupby("language", dropna=False)
        .agg(
            n=("id", "size"),
            answered=("parse_success", "sum"),
            correct=("correct", "sum"),
            accuracy=("correct", "mean"),
            parse_rate=("parse_success", "mean"),
        )
        .reset_index()
    )
    overall.to_csv(args.output_dir / "overall.csv", index=False)
    by_language.to_csv(args.output_dir / "by_language.csv", index=False)

    if "model" in merged.columns:
        by_model = (
            merged.groupby("model", dropna=False)
            .agg(
                n=("id", "size"),
                answered=("parse_success", "sum"),
                correct=("correct", "sum"),
                accuracy=("correct", "mean"),
                parse_rate=("parse_success", "mean"),
            )
            .reset_index()
        )
        by_model.to_csv(args.output_dir / "by_model.csv", index=False)

    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
