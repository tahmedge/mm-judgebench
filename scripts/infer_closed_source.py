#!/usr/bin/env python3
"""Run closed-source VLM judge inference for MM-JudgeBench.

Supports OpenAI, Anthropic Claude, and Google Gemini SDKs. API keys are read
from OPENAI_API_KEY, ANTHROPIC_API_KEY, and GEMINI_API_KEY.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import os
import re
from pathlib import Path

import pandas as pd


SYSTEM_PROMPT = (
    "You are a strict and fair judge for vision-language tasks. "
    "You will be shown an image and a user question, plus two candidate answers A and B. "
    "Decide which answer is better based on the following criteria. "
    "- Correctness with respect to the image and question. "
    "- Completeness and level of detail. "
    "- Relevance and clarity (no unnecessary verbosity). "
    "- Avoiding hallucinations or unsupported claims. "
    "Return ONLY a JSON object with this schema: "
    "{\"winner\":\"A | B\",\"reasoning\":\"brief reason\"}"
)


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def load_rows(dataset_ref: str, config: str | None, split: str, limit: int) -> pd.DataFrame:
    dataset_path = Path(dataset_ref).expanduser()
    if dataset_path.suffix.lower() == ".csv" and dataset_path.exists():
        df = pd.read_csv(dataset_path)
    else:
        if not config:
            raise ValueError("Pass --config when loading from Hugging Face.")
        from datasets import load_dataset

        df = load_dataset(dataset_ref, config, split=split).to_pandas()
    return df.head(limit).copy() if limit > 0 else df


def resolve_image(
    row: pd.Series,
    *,
    dataset_ref: str,
    csv_dataset_path: Path | None,
    image_root: Path | None,
) -> Path:
    benchmark_dir = csv_dataset_path.parent.parent if csv_dataset_path else None
    roots = [image_root] if image_root else []
    if csv_dataset_path:
        roots.extend([benchmark_dir, benchmark_dir / "images"])
    local_dataset_root = Path(dataset_ref).expanduser()
    if local_dataset_root.exists() and local_dataset_root.is_dir():
        roots.append(local_dataset_root)

    if "image_path" in row and isinstance(row["image_path"], str) and row["image_path"].strip():
        rel = row["image_path"].strip()
        for root in roots:
            candidate = root / rel
            if candidate.exists():
                return candidate
        if not csv_dataset_path and "/" in dataset_ref:
            from huggingface_hub import hf_hub_download

            return Path(hf_hub_download(repo_id=dataset_ref, filename=rel, repo_type="dataset"))

    sample_id = str(row["id"])
    for root in roots:
        exact = root / sample_id
        if exact.exists():
            return exact
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"):
            candidate = root / f"{Path(sample_id).stem}{ext}"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"No image found for id={sample_id}")


def user_prompt(row: pd.Series) -> str:
    query = "" if pd.isna(row.get("query", "")) else str(row.get("query", "")).strip()
    return (
        f"Language: {row.get('language', '')}\n"
        f"Question: {query}\n\n"
        f"Answer A:\n{row['response1']}\n\n"
        f"Answer B:\n{row['response2']}\n\n"
        "Which answer is better?"
    )


def parse_output(text: str) -> tuple[str, str]:
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
        reasoning = data.get("reasoning", "") or data.get("rationale", "")
        return str(data.get("winner", "")).upper(), str(reasoning)
    except json.JSONDecodeError:
        match = re.search(r'"winner"\s*:\s*"([ABab])"', text)
        return (match.group(1).upper() if match else "", text)


def call_openai(*, model: str, image: Path, prompt: str, temperature: float, max_tokens: int) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url(image)}},
                ],
            },
        ],
    )
    return response.choices[0].message.content or ""


def call_anthropic(*, model: str, image: Path, prompt: str, temperature: float, max_tokens: int) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    mime = mimetypes.guess_type(str(image))[0] or "image/png"
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime,
                            "data": base64.b64encode(image.read_bytes()).decode("ascii"),
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return "".join(block.text for block in response.content if getattr(block, "type", "") == "text")


def call_google(*, model: str, image: Path, prompt: str, temperature: float, max_tokens: int) -> str:
    import google.generativeai as genai
    from PIL import Image

    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    gemini = genai.GenerativeModel(
        model,
        system_instruction=SYSTEM_PROMPT,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "response_mime_type": "application/json",
        },
    )
    response = gemini.generate_content([prompt, Image.open(image)])
    return getattr(response, "text", "") or ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="tahmedge/MM-JudgeBench", help="Hugging Face dataset repo/path or a local CSV file.")
    parser.add_argument("--config", default=None, help="Hugging Face config, e.g. m-vl-rewardbench.")
    parser.add_argument("--split", default="original", help="Hugging Face split to load.")
    parser.add_argument("--provider", required=True, choices=["openai", "anthropic", "google"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    dataset = load_rows(args.dataset, args.config, args.split, args.limit)
    csv_dataset_path = Path(args.dataset).expanduser() if Path(args.dataset).expanduser().suffix.lower() == ".csv" else None
    args.output.parent.mkdir(parents=True, exist_ok=True)

    callers = {"openai": call_openai, "anthropic": call_anthropic, "google": call_google}
    write_header = not args.output.exists()
    with args.output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "language", "provider", "model", "raw_response", "winner", "reasoning"],
        )
        if write_header:
            writer.writeheader()
        for _, row in dataset.iterrows():
            image = resolve_image(
                row,
                dataset_ref=args.dataset,
                csv_dataset_path=csv_dataset_path,
                image_root=args.image_root,
            )
            raw = callers[args.provider](
                model=args.model,
                image=image,
                prompt=user_prompt(row),
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            winner, reasoning = parse_output(raw)
            writer.writerow(
                {
                    "id": row["id"],
                    "language": row["language"],
                    "provider": args.provider,
                    "model": args.model,
                    "raw_response": raw,
                    "winner": winner,
                    "reasoning": reasoning,
                }
            )
            handle.flush()


if __name__ == "__main__":
    main()
