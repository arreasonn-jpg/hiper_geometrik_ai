# STALE and Real-World Knowledge Lifecycle

`real-artifact-knowledge-lifecycle-v1` adds time and source-revision semantics
to provenance-aware knowledge. Its central rule is:

> **STALE is not FALSE.** It means freshness could not be established or a
> dependency is no longer active. `RETRACTED` requires an explicit withdrawal.

## States

- `ACTIVE`: source hash was validated within its declared freshness window.
- `STALE`: max age expired, source was unavailable, or a dependency ceased to
  be active. Excluded from default query, retained for audit.
- `SUPERSEDED`: a newer content hash/revision for the same source URL replaced
  this record. Immutable history is retained.
- `RETRACTED`: explicit source withdrawal; reason is mandatory at the API.

## Transitions and invariants

Records require source URL, SHA-256, revision, explicit retrieval/validation
time and positive max age. The clock is always passed by the caller: scientific
tests never depend on wall-clock time. Same-hash revalidation can restore a
STALE record. Changed content cannot be passed off as revalidation: it must be
ingested as a new revision, which marks the previous head SUPERSEDED. A
SUPERSEDED or RETRACTED record cannot be silently reactivated.

Derived records list their source record IDs. STALE/SUPERSEDED/RETRACTED state
propagates transitively as STALE to active dependents. Every transition enters
an append-only SHA-256 parent-chained event ledger.

## Real-artifact benchmark

The Research Suite Verification section uses the two vendored, byte-verified
Turkish Web Treebank files, their Apache-2.0 provenance URLs and pinned upstream
revision. It deterministically exercises:

1. active ingest;
2. freshness expiry;
3. unchanged hash revalidation;
4. controlled changed-revision supersession;
5. controlled source outage;
6. explicit controlled withdrawal;
7. dependency propagation and event-chain verification.

The files and provenance are authentic real-world artifacts. Events after
initial ingest are controlled interventions; the report explicitly does **not**
claim that upstream TWT actually changed or withdrew a source. No live HTTP
request is made, keeping the five-seed benchmark reproducible.
