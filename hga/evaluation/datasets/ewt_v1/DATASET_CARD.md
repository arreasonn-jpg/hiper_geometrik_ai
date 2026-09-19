# HGA English Benchmark — UD English EWT v1

This directory redistributes the **UD English EWT** CoNLL-U source files from
pinned upstream revision `4a4d77f599ea53cc405f85d0cec4b2f14f81d42b`. The
vendored `train.conllu`, `dev.conllu`, and `test.conllu` files are
byte-identical to `en_ewt-ud-train.conllu`, `en_ewt-ud-dev.conllu`, and
`en_ewt-ud-test.conllu` from that revision. `PROVENANCE.json` records the
source and license SHA-256 values.

## License and attribution

The source corpus is distributed under **CC-BY-SA-4.0**. The verbatim upstream
license text is included as `LICENSE-CC-BY-SA-4.0.txt`. This dataset keeps its
own license and attribution; it does not change the repository code license.
The upstream copyright notice is recorded in `PROVENANCE.json`. Please cite:

> Natalia Silveira, Timothy Dozat, Marie-Catherine de Marneffe, Samuel Bowman,
> Miriam Connor, John Bauer and Christopher D. Manning. 2014. *A Gold Standard
> Dependency Corpus for English*. LREC 2014.

For fuller corpus authorship and UD release information, consult the upstream
repository and its `README.md` at the pinned revision.

## Task and protocol

The benchmark task is **basic morphosyntactic dependency-arc candidate
verification**:

- Every annotated basic `(dependent, relation, head)` arc is a positive.
- A deterministic different legal head in the same sentence is a negative.
- The official UD train/dev/test file boundary is never crossed.
- To keep five-seed CPU runs practical, the suite selects an immutable subset
  within each official split: the first 512/128/128 sentence IDs after sorting
  by SHA-256 of `sent_id` (train/dev/test). The full upstream files remain
  vendored and integrity-checked before subset selection.

The associated command is:

```bash
python -m hga english-ewt --seeds 1,2,3,4,5 \
  --out docs/english_ewt_baselines.json \
  --markdown docs/ENGLISH_EWT_BASELINES.md
```

It compares dense, vanilla Transformer, **BERT-style bidirectional**, and
**GPT-style causal** architecture families. BERT-style and GPT-style here mean
small random-initialized architectures trained from scratch on this task; they
are not pretrained BERT/GPT releases and do not establish an LLM comparison.
