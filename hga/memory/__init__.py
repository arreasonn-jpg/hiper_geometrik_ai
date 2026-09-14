# -*- coding: utf-8 -*-
"""Memory katmanı: seyrek deneyim slotları + experience replay + entegrasyon + torch köprüsü."""
from .ablation import AblasyonDeneyi
from .benchmark import (
    ActiveMemoryStressPoint,
    ActiveMemoryStressReport,
    MemoryBenchmarkResult,
    MemoryCapacitySweepReport,
    MemoryStressReport,
    run_active_memory_stress,
    run_memory_benchmark,
    run_memory_capacity_sweep,
    run_memory_stress,
)
from .dynamic_kv import (
    CompactionReport,
    DynamicKVMemory,
    DynamicKVRecord,
    DynamicKVSnapshot,
    EvictionEvent,
    MigrationReport,
)
from .entegrasyon import BellekEntegrasyonu
from .genelleme_ablasyonu import GenellemeAblasyonu
from .gorev_ablasyonu import GorevAblasyonu
from .hierarchical import (
    DEFAULT_CAPACITY,
    TIERS,
    HierarchicalMemory,
    TierStats,
    write_many,
)
from .interference import (
    DYNAMIC_KV,
    FIRST_WINS,
    LAST_WINS,
    POLITIKALAR,
    InterferenceCase,
    InterferenceReport,
    PolicyComparisonReport,
    ScalingReport,
    carpisan_anahtar_bul,
    run_fixed_vs_dynamic_scaling,
    run_interference_test,
    run_policy_comparison,
)
from .kopru import TorchKoprusu, bilesen_token, torch_var_mi
from .lifecycle import DynamicKVLifecycleReport, run_dynamic_kv_lifecycle
from .neural_kopru import NeuralKopru
from .replay import DeneyimTekrari
from .sparse_memory import DeneyimSlotlari, parmak_izi

__all__ = [
    "HierarchicalMemory", "TierStats", "TIERS", "DEFAULT_CAPACITY",
    "write_many",
    "DeneyimSlotlari",
    "parmak_izi",
    "MemoryBenchmarkResult",
    "MemoryCapacitySweepReport",
    "MemoryStressReport",
    "ActiveMemoryStressPoint", "ActiveMemoryStressReport",
    "run_active_memory_stress",
    "run_memory_benchmark",
    "run_memory_capacity_sweep",
    "run_memory_stress",
    "DeneyimTekrari",
    "BellekEntegrasyonu",
    "DynamicKVMemory", "DynamicKVRecord", "DynamicKVSnapshot",
    "EvictionEvent", "CompactionReport", "MigrationReport",
    "DynamicKVLifecycleReport", "run_dynamic_kv_lifecycle",
    "TorchKoprusu",
    "NeuralKopru",
    "AblasyonDeneyi",
    "GorevAblasyonu",
    "GenellemeAblasyonu",
    "torch_var_mi",
    "bilesen_token",
    "FIRST_WINS", "LAST_WINS", "DYNAMIC_KV", "POLITIKALAR",
    "InterferenceCase", "InterferenceReport", "PolicyComparisonReport",
    "ScalingReport", "carpisan_anahtar_bul", "run_interference_test",
    "run_policy_comparison", "run_fixed_vs_dynamic_scaling",
]
