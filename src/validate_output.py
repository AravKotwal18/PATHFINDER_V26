#!/usr/bin/env python3
"""Validate TUM output structure and alignment against an input IMU file."""
import argparse, numpy as np
from main import read_csv_any, timestamp_seconds
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("trajectory",nargs="?",default="estimated_trajectory.txt")
    args=ap.parse_args()
    t,_,_=read_csv_any(args.input); expected=timestamp_seconds(t)
    rows=[]
    with open(args.trajectory) as f:
        for line in f:
            s=line.strip()
            if not s or s.startswith("#"): continue
            vals=[float(x) for x in s.split()]
            if len(vals)!=8: raise SystemExit(f"FAIL: expected 8 columns, got {len(vals)}")
            rows.append(vals)
    a=np.asarray(rows)
    if len(a)!=len(expected): raise SystemExit(f"FAIL: expected {len(expected)} poses, got {len(a)}")
    if not np.allclose(a[:,0],expected,atol=2e-6): raise SystemExit("FAIL: timestamps are misaligned")
    norms=np.linalg.norm(a[:,4:8],axis=1)
    if not np.all(np.isfinite(a)) or not np.allclose(norms,1,atol=2e-4):
        raise SystemExit("FAIL: non-finite pose or non-unit quaternion")
    print(f"PASS: {len(a)} poses; timestamps aligned; all quaternions normalized.")
if __name__=="__main__": main()
