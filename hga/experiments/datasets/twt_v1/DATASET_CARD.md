# HGA Real Turkish Benchmark — Turkish Web Treebank v1

This directory redistributes the **Turkish Web Treebank (TWT)** at upstream
revision `40838e5cbe3f2882d4e768a3d782e6219e50b52a`. The two CoNLL-U files are
byte-identical to that revision. See `PROVENANCE.json` for source hashes,
retrieval date, deterministic split/quarantine rules and task construction.

## Source and annotation

TWT contains 4,851 real Turkish sentences sampled from Turkish web pages and
Turkish Wikipedia. Kayadelen, Öztürel and Bohnet manually annotated
segmentation, morphology, POS, basic dependency heads and 44 dependency
relations. Published corpus totals are 66,466 words and 81,370 inflectional
group tokens.

Citation:

> Tolga Kayadelen, Adnan Öztürel and Bernd Bohnet. 2020. A Gold Standard
> Dependency Treebank for Turkish. LREC 2020, pp. 5158–5165.
> https://aclanthology.org/2020.lrec-1.634/

## License

The upstream corpus is distributed under **Apache-2.0**. A verbatim license
copy is included as `LICENSE-APACHE-2.0.txt`. The derived HGA split, quarantine,
negative-candidate construction and benchmark implementation are modifications;
the source CoNLL-U annotations themselves are not modified.

## Benchmark task

The task is **basic morphosyntactic dependency-arc candidate verification**.
Each human-annotated `(dependent token, dependency relation, head token)` arc is
a positive. A deterministic alternative-head corruption is a negative because
a basic dependency tree assigns each token exactly one annotated head/relation.
The evaluator reports Accuracy, precision, recall, F1, false acceptance rate,
false rejection rate and selective coverage, including entity-, relation-,
composition- and wording-disjoint slices.

“Entity” is only the benchmark's token identity key (lemma, falling back to
surface form); it is **not** a named-entity label. TWT dependency relations are
morphosyntactic relations, **not semantic knowledge triples**. No semantic
relation-extraction or general-language claim may be made from this task.

## Deterministic split and leakage quarantine

The initial split follows upstream code independently within both files:
`index % 10 < 8` train, `index % 10 == 8` dev and `index % 10 == 9` test.
Two source sentences remain in the raw files but are excluded from effective
evaluation by fixed IDs: one exact train/dev duplicate and one train/test
near-duplicate above the fixed lemma-set Jaccard threshold of 0.80. Effective
counts are 3,881 train, 484 dev and 484 test sentences.

The benchmark verifies raw file SHA-256 values before parsing and emits dataset,
configuration, split and candidate SHA-256 values in every report.
