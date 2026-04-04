"""
phase1_local/inspect_hospital_d.py
------------------------------------
Run this ONCE to inspect the nested .mat structure.
Output tells us exactly how to fix hospital_d.py.

Run:
    python phase1_local/inspect_hospital_d.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy.io import loadmat

DATA_DIR = "data/hospital_d"

# Find the .mat file
mat_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".mat")]
print(f"Mat files found: {mat_files}\n")

mat_path = os.path.join(DATA_DIR, mat_files[0])
data = loadmat(mat_path)

# Top-level keys
print("=== Top-level keys ===")
for k, v in data.items():
    if not k.startswith("__"):
        print(f"  '{k}': shape={v.shape}, dtype={v.dtype}")

# Inspect first record inside 'p'
print("\n=== First record: data['p'][0, 0] ===")
record = data["p"][0, 0]
print(f"  Type: {type(record)}")
print(f"  dtype: {record.dtype}")

# If it's a numpy void (struct), list its fields
if hasattr(record, "dtype") and record.dtype.names:
    print(f"  Fields: {record.dtype.names}")
    for field in record.dtype.names:
        val = record[field]
        shape = val.shape if hasattr(val, "shape") else type(val)
        dtype = val.dtype if hasattr(val, "dtype") else ""
        # Print first few values if small
        if hasattr(val, "flatten") and val.size < 10:
            print(
                f"    '{field}': shape={shape}, dtype={dtype}, values={val.flatten()}"
            )
        else:
            print(f"    '{field}': shape={shape}, dtype={dtype}")
else:
    print(f"  Raw value: {record}")

# Try second record too
print("\n=== Second record: data['p'][0, 1] ===")
record2 = data["p"][0, 1]
if hasattr(record2, "dtype") and record2.dtype.names:
    print(f"  Fields: {record2.dtype.names}")
    for field in record2.dtype.names:
        val = record2[field]
        shape = val.shape if hasattr(val, "shape") else type(val)
        dtype = val.dtype if hasattr(val, "dtype") else ""
        if hasattr(val, "flatten") and val.size < 10:
            print(
                f"    '{field}': shape={shape}, dtype={dtype}, values={val.flatten()}"
            )
        else:
            print(f"    '{field}': shape={shape}, dtype={dtype}")

print("\n=== Copy this output and share it ===")
