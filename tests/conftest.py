# -*- coding: utf-8 -*-
"""Testler için proje kökünü sys.path'e ekler."""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)
