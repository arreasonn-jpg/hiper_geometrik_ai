# Publication / external-review gate

The repository can prepare a submission artifact but cannot itself create
co-authors, obtain ethical/legal approval, upload to arXiv, or obtain peer
review. A CKPT-000 publication claim is blocked until every owner-controlled
item below is completed.

## Owner-controlled requirements

- [ ] Confirm real author list, affiliations, contact email, and authorship
      consent.
- [ ] Review `paper/main.tex`; replace the deliberate placeholder author line.
- [ ] Choose a submission license and archive exact source/benchmark revision
      (for example with an archival DOI).
- [ ] Re-run final experiments in an owner-approved environment and attach
      `EXP-NNNN` manifests plus hardware metadata.
- [ ] Obtain at least one independent technical review of theory, data-license
      treatment, and experimental conclusions.
- [ ] Submit to the chosen venue/arXiv category with project-owner approval.

## Repository-complete evidence

- [x] Reproducible experiment manifest protocol and five-seed suite.
- [x] Pinned Turkish TWT and English UD EWT source provenance/hashes.
- [x] Bilinear/rank/collapse framework with no fabricated VC claim.
- [x] Dense, Transformer, BERT-style, and GPT-style architectural controls.
- [x] An explicit scaling probe whose narrow range blocks a scaling-law claim.
- [x] Preprint source draft with limitations and no invented author metadata.

A checkbox in the first section must only be marked by an authorized human.
