# HGA preprint draft packet

`main.tex` is an **honest preprint draft**, not an uploaded arXiv work and not
a peer-reviewed publication. It is included so the CKPT-000 code/data/claim
boundary can be reviewed as one document.

Before an external submission, the project owner must supply or approve:

1. real author names, affiliations, contact email, and authorship consent;
2. a target license and repository archival identifier (for example Zenodo DOI);
3. the exact hardware/runtime manifest used for reported final runs;
4. independent reviewer or co-author feedback;
5. an arXiv account/category and explicit authorization to submit.

The draft intentionally contains no invented author, reviewer, institutional,
SOTA, or scaling-law claim. It labels the English BERT-style/GPT-style arms as
random-initialized small architectural controls, and it states that the full
HGA VC/pseudo-dimension remains unproved.

Build locally with a TeX distribution:

```bash
cd paper
latexmk -pdf main.tex
```

The generated PDF is deliberately ignored by Git; source, benchmark protocol,
and pinned source hashes remain the reproducible record.
