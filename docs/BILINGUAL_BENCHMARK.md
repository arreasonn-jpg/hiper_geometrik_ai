# CKPT-000 bilingual benchmark coverage

CKPT-000 now has two independently vendored, source-hash-verified,
human-annotated dependency-treebank protocols. They share the same narrow task:
**basic morphosyntactic dependency-arc candidate verification**. A positive is a
human-annotated basic arc; a negative is a deterministic distinct-head
corruption in the same sentence.

| Language | Dataset | Upstream revision | License | Main command |
|---|---|---|---|---|
| Turkish (`tr`) | Turkish Web Treebank v1 | `40838e5cbe3f2882d4e768a3d782e6219e50b52a` | Apache-2.0 | `python -m hga twt-sonuc` |
| English (`en`) | UD English EWT v1 | `4a4d77f599ea53cc405f85d0cec4b2f14f81d42b` | CC-BY-SA-4.0 | `python -m hga english-ewt` |

## English architectural controls

`english-ewt` runs five seeds by default and reports mean ± population standard
deviation for dense, vanilla Transformer, BERT-style bidirectional, and
GPT-style causal architecture controls. The curated result is:
[`ENGLISH_EWT_BASELINES.md`](ENGLISH_EWT_BASELINES.md), with full machine
report [`english_ewt_baselines.json`](english_ewt_baselines.json).

```bash
python -m hga english-ewt --seeds 1,2,3,4,5 \
  --out docs/english_ewt_baselines.json \
  --markdown docs/ENGLISH_EWT_BASELINES.md
```

The English source includes the complete official train/dev/test files and a
hash-deterministic 512/128/128 sentence subset is selected *inside* those
pre-existing splits for the CPU benchmark. `PROVENANCE.json` verifies all source
and license SHA-256 values before parsing.

## What this does and does not close

This closes the repository-level absence of a non-Turkish, English, reproducible
benchmark and makes the existing Turkish result a **bilingual two-corpus
coverage claim**. It does **not** establish:

- multilingual pretraining, cross-lingual transfer, translation, or language
  generation;
- SOTA dependency parsing or a fair comparison to pretrained BERT/GPT models;
- a broad multi-language benchmark beyond Turkish and English;
- semantic relation extraction or general reasoning.

The BERT-style and GPT-style arms are explicitly small models trained from
random initialization on the same structured arc features. “BERT-style” and
“GPT-style” name their attention direction and pooling/causal construction,
not released pretrained BERT or GPT checkpoints.
