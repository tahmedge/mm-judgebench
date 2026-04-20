# MM-JudgeBench: Lost in Translation — Do LVLM Judges Generalize Across Languages?

[![Paper](https://img.shields.io/badge/Paper-ACL%202026%20Findings-blue)](https://github.com/tahmedge/mm-judgebench/tree/main)

> **Accepted at ACL 2026 Findings** 🎉

Official repository for the paper **"Lost in Translation: Do LVLM Judges Generalize Across Languages?"**

**MM-JudgeBench** is the first large-scale benchmark for **multilingual and multimodal judge model evaluation**, covering over **60K pairwise preference instances across 25 typologically diverse languages**.

*The dataset will be released soon. Stay tuned!*

---

## 📖 Overview

Automatic evaluators such as reward models and LVLM-as-a-judge frameworks play a central role in aligning and evaluating large vision–language models (LVLMs). However, existing evaluation benchmarks are almost exclusively **English-centric**, leaving a critical gap: *how well do these evaluators generalize across languages?*

MM-JudgeBench closes this gap by unifying two complementary evaluation subsets within a single multilingual framework, enabling systematic analysis of LVLM judges across diverse linguistic and multimodal settings.

---

## ✨ Key Contributions

- 🌍 **MM-JudgeBench**: The first large-scale benchmark for multilingual + multimodal LVLM judge evaluation, spanning **25 typologically diverse languages** and **60K+ preference instances**.
- 🧪 **Large-scale empirical study**: Evaluation of **22 state-of-the-art LVLMs** (16 open-source + 6 proprietary), revealing cross-lingual variations invisible under English-only evaluation.
- 🛠️ **Multilingual training set**: A **100K-sample** multilingual training subset derived from MM-RewardBench to support domain-adaptive fine-tuning of reward models.
- 🔍 **Bias & robustness analysis**: Investigation of position bias, length bias, instruction-following behavior, and cross-lingual robustness across languages.

---

## 🧩 Benchmark Composition

MM-JudgeBench integrates **two complementary evaluation subsets** plus a training resource:

| Subset | Source | Samples | Purpose |
|---|---|---|---|
| **M-VL-RewardBench** | VL-RewardBench | ~31K | General vision–language preference judgment |
| **M-OpenCQA** | OpenCQA | ~30K | Chart-centric visual–text reasoning |
| **M-MM-RewardBench** *(training set)* | MM-RewardBench | ~100K | Domain adaptation / fine-tuning |

### 🌐 Supported Languages (25)

Arabic, Bengali, Chinese, Czech, Dutch, English, French, German, Greek, Hebrew, Hindi, Indonesian, Italian, Japanese, **Kazakh** *(low-resource)*, Korean, Persian, Polish, Portuguese, Romanian, Russian, Spanish, Turkish, Ukrainian, Vietnamese.

---

## 🛠️ Methodology

The benchmark was constructed in three stages:

1. **Translation Model Selection** — Gemini-3-Pro was selected as the translator after comparative evaluation using **LaBSE** and **CometKiwi** quality metrics against Gemini-2.5-Pro / Flash / Flash-Lite.
2. **Data Construction** — A single-prompt JSON translation strategy reduced API calls by 24×, followed by a **threshold-based quality filter** (LaBSE & CometKiwi ≥ 0.75), human back-translation review, and re-translation where needed.
3. **Evaluation** — Pairwise accuracy, positional bias, and length bias computed across 22 LVLM judges, with both original and reversed answer orderings for robustness.

---

## 🤖 Models Evaluated

**Closed-source (6):** GPT-5, GPT-5-Mini, GPT-5-Nano, Gemini-2.5-Flash, Gemini-2.5-Flash-Lite, Claude-4.5-Haiku, Grok-4.1-Fast

**Open-source (16):** Qwen3-VL (2B / 4B / 8B / 30B-A3B / 32B), InternVL-3.5 (1B / 2B / 4B / 8B / 14B), Gemma-3 (4B / 12B / 27B), Pixtral-12B, LLaVA-Critic-7B

---

## 📊 Major Results

### M-VL-RewardBench (Top Performers)

| Model | Avg Accuracy | Cross-lingual Variance |
|---|---|---|
| **GPT-5** | **81.3%** | **0.2** |
| GPT-5-Mini | 78.1% | 0.4 |
| Gemini-2.5-Flash | 76.7% | 0.5 |
| GPT-5-Nano | 73.2% | 1.2 |
| Grok-4.1-Fast | 71.3% | 0.7 |
| **Qwen3-VL-32B** (best open) | **68.8%** | 3.3 |

### M-OpenCQA (Open Models)

| Model | Avg Accuracy | Variance |
|---|---|---|
| **Qwen3-VL-32B** | **67.4%** | 1.4 |
| InternVL-3.5-14B | 66.3% | 2.1 |
| Qwen3-VL-8B | 64.5% | 0.5 |
| Qwen3-VL-30B-A3B | 63.7% | 0.9 |

### Key Findings

- 🏆 **GPT-5 dominates** with the highest accuracy (81.3%) and lowest cross-lingual variance (0.2).
- ⚠️ **Efficiency-optimized variants collapse**: Gemini-2.5-Flash-Lite (40.8%) and Claude-4.5-Haiku (56.4%) show severe multilingual degradation despite strong English performance.
- 💪 **Qwen3-VL is the strongest open-source family**, scaling consistently from 54.3% (2B) to 68.8% (32B), surpassing several closed models.
- 🌐 **Low-resource languages suffer**: Kazakh yields the worst per-model accuracy in most LVLMs (10/15 times on M-OpenCQA).
- 🎯 **Bias amplifies cross-lingually**: Positional bias nearly **2× higher in non-English languages** for InternVL-3.5 and Gemma-3.
- 📉 **Specialized ≠ better**: LLaVA-Critic-7B (trained for reward modeling) scores below 50% on M-VL-RewardBench.
- 🚀 **Fine-tuning helps**: Qwen3-VL-8B fine-tuned on M-MM-RewardBench gains **+14%** over direct prompting and **+10%** over rationale-augmented prompting.
- 🧠 **Reasoning helps**: Removing rationale generation degrades accuracy by 2–4%.
- 🔄 **Robust to translation choice**: Model rankings remain consistent across Gemini-3-Pro, Gemini-2.5-Flash, and Gemini-2.5-Flash-Lite translators.

---

## 🔬 Analyses Included

- Cross-lingual amplification of positional bias
- Length bias across closed and open models
- Output format instruction-following compliance
- Translation sensitivity analysis (Gemini-3-Pro vs. 2.5-Flash vs. Flash-Lite)
- Task-level breakdown (Hallucination / Mathematical / General Multimodal)
- Script-group analysis (Latin, Cyrillic, Greek, Arabic, Hebrew, Devanagari, CJK)
- Resource-tier analysis (High / Mid / Low-resource languages)
- Human evaluation of translation quality (back-translation + native-speaker review)
- Cross-judge consistency validation with Gemini-2.5-Pro (Spearman ρ = 0.93, Pearson r = 0.94)

---

## 📁 Repository Structure

```
mm-judgebench/
├── data/
│   ├── m-vl-rewardbench/      # 31K multilingual samples
│   ├── m-opencqa/             # 30K multilingual samples
│   └── m-mm-rewardbench/      # 100K training samples
├── eval/                      # Evaluation scripts
├── prompts/                   # Translation & judging prompts
└── README.md
```

## 📝 Citation

If you use MM-JudgeBench in your research, please cite:

```bibtex
@inproceedings{laskar2026lost,
  title     = {Lost in Translation: Do LVLM Judges Generalize Across Languages?},
  author    = {Laskar, Md Tahmid Rahman and Islam, Mohammed Saidul and Nayeem, Mir Tafseer
               and Bhuiyan, Md Amran and Rahman, Mizanur and Joty, Shafiq
               and Hoque, Enamul and Huang, Jimmy Xiangji},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  year      = {2026}
}
```

---

## 👥 Authors

Md Tahmid Rahman Laskar¹, Mohammed Saidul Islam¹, Mir Tafseer Nayeem², Md Amran Bhuiyan¹, Mizanur Rahman¹, Shafiq Joty³, Enamul Hoque¹, Jimmy Xiangji Huang¹

¹ York University  ² University of Alberta  ³ Salesforce AI Research

📧 Contact: `{tahmedge, enamulh, jhuang}@yorku.ca`

---

## 🙏 Acknowledgments

This research is supported by NSERC Canada, the York Research Chairs (YRC) program, the Canada Foundation for Innovation (CFI), Google's Gemini Academic Program, CUPE 3903 Research Grant, and Digital Research Alliance of Canada.

---

## ⚖️ Ethics & Limitations

MM-JudgeBench is designed to **expose, not obscure**, multilingual failure modes of LVLM judges. Practitioners should avoid deploying judge models in high-stakes or user-facing scenarios without multilingual validation and human oversight. All data are derived from publicly available benchmarks and contain no personally identifiable information. Some linguistic or cultural nuances may be lost due to automated translation — though our human evaluation confirms high translation quality (55.7% "Perfect", 43.8% "Good", only 0.5% "Bad"), and a Bengali native-speaker study rated 73% "Perfect" and 27% "Good" with zero "Bad" samples.

---

## 🔗 Links

- 📄 **Paper:** [ACL 2026 Findings]()
- 💻 **Code & Data:** [https://github.com/tahmedge/mm-judgebench](https://github.com/tahmedge/mm-judgebench)
