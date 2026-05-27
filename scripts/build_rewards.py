#!/usr/bin/env python3
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
runpy.run_module("duoroute.build_data", run_name="__main__")
