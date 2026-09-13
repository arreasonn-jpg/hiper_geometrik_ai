# -*- coding: utf-8 -*-
"""HGA Generalization Test v1 Regression Suite."""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from experiments.experience_loop.run_generalization_v1 import run_hga_generalization_test_v1


def test_hga_generalization_v1_all_checks():
    report = run_hga_generalization_test_v1()
    assert report.unseen_positive_ok is True
    assert report.negative_rejection_ok is True
    assert report.conflict_detection_ok is True
    assert report.multihop_chaining_ok is True
    assert report.leakage_free_ok is True
