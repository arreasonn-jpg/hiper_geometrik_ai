"""HGA CLI komutu (Faz 2 refactor)."""
from __future__ import annotations

from hga.cli.common import _yaz_rapor


def _bellek_streaming(target_records=100_000_000, sample_records=10_000,
                      shard_records=1_000_000, checkpoint_interval=1_000,
                      seeds=None, out=None, markdown=None):
    """P0-8: 100M-ready streaming/checkpoint dry-run harness."""
    from hga.experiments.memory_hierarchy import (
        memory_streaming_harness_markdown,
        run_memory_streaming_harness,
    )
    tohum = int((seeds or "1").split(",")[0])
    rapor = run_memory_streaming_harness(
        target_records=int(target_records),
        sample_records=int(sample_records),
        shard_records=int(shard_records),
        checkpoint_interval=int(checkpoint_interval),
        seed=tohum,
    )
    md = memory_streaming_harness_markdown(rapor)
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)

def _bellek_hiyerarsi(profile="smoke", seeds=None, out=None, markdown=None):
    """P0-8: Hot/Warm/Cold/Archive hiyerarşik bellek benchmarkı."""
    from hga.experiments.memory_hierarchy import (
        memory_hierarchy_markdown,
        run_memory_hierarchy_benchmark,
    )
    tohum = int((seeds or "1").split(",")[0])
    rapor = run_memory_hierarchy_benchmark(profile=profile, seed=tohum)
    md = memory_hierarchy_markdown(rapor)
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)
