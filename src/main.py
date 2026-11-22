#!/usr/bin/env python3
"""
PATHFINDER Sentinel 6D
Adaptive inertial trajectory reconstruction from degraded IMU telemetry.

Input: EuRoC imu0/data.csv (timestamp[ns], gyro xyz [rad/s], accel xyz [m/s^2])
Output: TUM pose rows: timestamp[s] tx ty tz qx qy qz qw

No external positioning, waypoint, or ground-truth data is used.
The estimator is deliberately offline and deterministic.
"""
from __future__ import annotations
import argparse, csv, io, json, math, os, sys, zipfile, time
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

G = 9.80665
EPS = 1e-12

def q_norm(q):
    n = float(np.linalg.norm(q))
    return q / n if n > EPS else np.array([0.,0.,0.,1.])

def q_mul(a, b):
    ax,ay,az,aw=a; bx,by,bz,bw=b
    return np.array([aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx,
                     aw*bz+ax*by-ay*bx+az*bw,
                     aw*bw-ax*bx-ay*by-az*bz])

def q_conj(q):
    return np.array([-q[0],-q[1],-q[2],q[3]])

def q_rotate(q, v):
    # Rotate a 3-vector from body to world coordinates.
    return q_mul(q_mul(q, np.array([v[0],v[1],v[2],0.])), q_conj(q))[:3]

def q_from_two_vectors(a, b):
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    na=np.linalg.norm(a); nb=np.linalg.norm(b)
    if na<EPS or nb<EPS: return np.array([0.,0.,0.,1.])
    a=a/na; b=b/nb
    d=float(np.dot(a,b))
    if d < -0.999999:
        axis=np.cross(a,[1.,0.,0.])
        if np.linalg.norm(axis)<1e-6: axis=np.cross(a,[0.,1.,0.])
        axis/=np.linalg.norm(axis)
        return np.array([axis[0],axis[1],axis[2],0.])
    c=np.cross(a,b)
    return q_norm(np.array([c[0],c[1],c[2],1.+d]))

def q_from_omega(omega, dt):
    angle=float(np.linalg.norm(omega))*dt
    if angle < 1e-10:
        return q_norm(np.array([omega[0]*dt/2,omega[1]*dt/2,omega[2]*dt/2,1.]))
    axis=omega/np.linalg.norm(omega)
    s=math.sin(angle/2)
    return np.array([axis[0]*s,axis[1]*s,axis[2]*s,math.cos(angle/2)])

def read_csv_any(path):
    """Read a CSV or zip containing imu0/data.csv; map columns by header names."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            candidates=[n for n in zf.namelist() if n.endswith("imu0/data.csv")]
            if not candidates: candidates=[n for n in zf.namelist() if n.endswith(".csv")]
            if not candidates: raise ValueError("ZIP contains no IMU CSV")
            raw=zf.read(candidates[0]).decode("utf-8-sig")
    else:
        with open(path,"r",encoding="utf-8-sig") as f: raw=f.read()
    lines=[ln for ln in raw.splitlines() if ln.strip()]
    if not lines: raise ValueError("Input CSV is empty")
    first=lines[0].lstrip("#").strip()
    has_header=any(ch.isalpha() for ch in first)
    if has_header:
        header=[x.strip().lower().replace('"','') for x in first.split(",")]
        rows=list(csv.reader(lines[1:]))
        arr=np.asarray([[float(v) for v in r[:len(header)]] for r in rows if len(r)>=len(header)],dtype=np.float64)
        def find(keys, fallback):
            for i,h in enumerate(header):
                if any(k in h for k in keys): return i
            return fallback
        ti=find(["timestamp"],0)
        # Explicit EuRoC ordering: w_RS_S_* then a_RS_S_*.
        gi=[find([f"w_rs_s_{a}",f"gyro_{a}",f"g{a}"],1+j) for j,a in enumerate("xyz")]
        ai=[find([f"a_rs_s_{a}",f"accel_{a}",f"a{a}"],4+j) for j,a in enumerate("xyz")]
        t=arr[:,ti]; gyro=arr[:,gi]; accel=arr[:,ai]
        # If headers are generic / don't expose semantic names, honor EuRoC positional order.
        if not any(("w_rs_s" in h or "gyro" in h) for h in header):
            gyro=arr[:,1:4]; accel=arr[:,4:7]
    else:
        arr=np.asarray([[float(v) for v in ln.split(",")] for ln in lines],dtype=np.float64)
        if arr.shape[1] != 7: raise ValueError(f"Expected 7 columns, received {arr.shape[1]}")
        t=arr[:,0]; gyro=arr[:,1:4]; accel=arr[:,4:7]
    if len(t)<2: raise ValueError("At least two IMU samples are required")
    if not np.all(np.isfinite(t)):
        raise ValueError("Non-finite timestamps cannot be mapped to TUM output rows")
    # Never silently drop an IMU frame: forward-fill invalid sensor components.
    for arr in (gyro, accel):
        for j in range(3):
            last=0.0
            for i in range(len(arr)):
                if np.isfinite(arr[i,j]): last=float(arr[i,j])
                else: arr[i,j]=last
    if len(t)<2: raise ValueError("At least two IMU samples are required")
    return t,gyro,accel

def timestamp_seconds(t):
    med=float(np.median(np.diff(t)[np.diff(t)>0])) if np.any(np.diff(t)>0) else 0.005
    scale=1e-9 if med>1e6 else (1e-6 if med>1e3 else (1e-3 if med>1.0 else 1.0))
    return t*scale

@dataclass
class Config:
    gravity: float=G
    attitude_gain: float=0.035
    max_gap_s: float=0.25
    stationary_gyro: float=0.045
    stationary_accel_tol: float=0.22
    gyro_bias_gain: float=0.015
    accel_smoothing: float=0.18
    blind_zero_threshold: float=1e-8
    blind_confirm_s: float=0.35
    blind_velocity_damping: float=0.18
    max_linear_accel: float=18.0

def estimate(t_raw, gyro, accel, cfg=Config()):
    """Robust offline-friendly forward estimator; returns timestamps, positions, quaternions, diagnostics."""
    ts=timestamp_seconds(t_raw)
    n=len(ts)
    d=np.diff(ts)
    positive=d[d>1e-6]
    nominal=float(np.median(positive)) if len(positive) else 0.005
    nominal=float(np.clip(nominal,0.001,0.05))
    # Initialization from the first short, least-dynamic window.
    init_n=min(n,max(5,int(0.5/max(nominal,1e-3))))
    a0=np.median(accel[:init_n],axis=0)
    q=q_from_two_vectors(a0,np.array([0.,0.,cfg.gravity]))
    gyro_bias=np.zeros(3)
    pos=np.zeros(3); vel=np.zeros(3)
    positions=np.zeros((n,3)); quats=np.zeros((n,4))
    quats[0]=q
    smooth_world=np.zeros(3)
    zero_run=0.0; blind=False; blind_start=-1
    stationary_count=0; gap_count=0; spike_count=0
    last_acc_world=np.zeros(3)
    for i in range(1,n):
        raw_dt=float(ts[i]-ts[i-1])
        if raw_dt < 0: raw_dt=0.0
        if raw_dt > nominal*1.8: gap_count+=1
        dt=min(raw_dt,2.0)
        # duplicate timestamps: preserve output row and pose without integrating.
        if dt <= 1e-8:
            positions[i]=pos; quats[i]=q
            continue
        gnorm=float(np.linalg.norm(gyro[i]))
        anorm=float(np.linalg.norm(accel[i]))
        is_zero=(gnorm<cfg.blind_zero_threshold and anorm<cfg.blind_zero_threshold)
        zero_run = zero_run + dt if is_zero else 0.0
        if zero_run >= cfg.blind_confirm_s and not blind:
            blind=True; blind_start=i
        if is_zero or blind:
            # An exactly-zero 6-axis vector is physically implausible in flight.
            # Coast during the short confirmation interval too, rather than
            # interpreting zero acceleration as -g after gravity subtraction.
            decay=math.exp(-cfg.blind_velocity_damping*dt)
            vel*=decay
            pos+=vel*dt
            positions[i]=pos; quats[i]=q
            continue
        # Robustly reject isolated extreme samples; keep gyro bounded to avoid quaternion flips.
        w=gyro[i]-gyro_bias
        wn=np.linalg.norm(w)
        if wn>12.0:
            w*=12.0/wn; spike_count+=1
        # Integrate attitude with midpoint angular rate.
        w_prev=gyro[i-1]-gyro_bias
        w_mid=0.5*(w_prev+w)
        q=q_norm(q_mul(q,q_from_omega(w_mid,dt)))
        # Accelerometer gravity correction only when magnitude is gravity-like.
        # Predicted specific-force direction in body coordinates is R^-1 * world_up.
        if abs(anorm-cfg.gravity) < 2.2 and anorm>1e-6:
            measured=accel[i]/anorm
            predicted=q_rotate(q_conj(q),np.array([0.,0.,1.]))
            err=np.cross(measured,predicted)
            corr=q_from_omega(-cfg.attitude_gain*err,1.0)
            q=q_norm(q_mul(q,corr))
        # Stationary detection enables zero-velocity and gyro-bias updates.
        stationary=(abs(anorm-cfg.gravity)<cfg.stationary_accel_tol and
                    np.linalg.norm(gyro[i])<cfg.stationary_gyro)
        if stationary:
            stationary_count+=1
            vel*=0.15
            gyro_bias=(1-cfg.gyro_bias_gain)*gyro_bias+cfg.gyro_bias_gain*gyro[i]
        # Rotate measured acceleration into world and remove gravity.
        aw=q_rotate(q,accel[i])-np.array([0.,0.,cfg.gravity])
        # Clip physically implausible spikes, then smooth to reduce high-frequency noise.
        norm=float(np.linalg.norm(aw))
        if norm>cfg.max_linear_accel:
            aw*=cfg.max_linear_accel/norm; spike_count+=1
        alpha=1.0-math.exp(-dt/max(cfg.accel_smoothing,1e-3))
        smooth_world=(1-alpha)*smooth_world+alpha*aw
        if stationary:
            smooth_world*=0.2
        # For long missing-data intervals, avoid treating the gap as a single large
        # acceleration impulse: coast with damped velocity and resume at next sample.
        if raw_dt>cfg.max_gap_s:
            vel*=math.exp(-0.12*min(raw_dt,5.0))
            pos+=vel*raw_dt
        else:
            pos+=vel*dt+0.5*smooth_world*dt*dt
            vel+=smooth_world*dt
        if stationary:
            vel*=0.10
        positions[i]=pos; quats[i]=q
    # Normalize all output quaternions and ensure sign continuity.
    for i in range(n):
        quats[i]=q_norm(quats[i])
        if i and np.dot(quats[i],quats[i-1])<0: quats[i]*=-1
    diag={
        "samples":int(n),"duration_s":float(ts[-1]-ts[0]),
        "nominal_rate_hz":float(1.0/nominal),
        "timestamp_gaps":int(gap_count),"rejected_or_clipped_events":int(spike_count),
        "stationary_samples":int(stationary_count),
        "blind_detected":bool(blind),"blind_start_index":int(blind_start),
        "output_rows":int(n),"algorithm":"Sentinel-6D adaptive inertial estimator",
        "notes":"IMU-only translation and yaw are not fully observable; trajectory is an estimate, not a globally corrected ground truth."
    }
    return ts,positions,quats,diag

def write_tum(path, ts, pos, quat):
    with open(path,"w",encoding="utf-8") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for t,p,q in zip(ts,pos,quat):
            f.write("{:.9f} {:.7f} {:.7f} {:.7f} {:.8f} {:.8f} {:.8f} {:.8f}\n".format(
                float(t),*p.tolist(),*q.tolist()))

def main():
    ap=argparse.ArgumentParser(description="PATHFINDER Sentinel 6D IMU trajectory estimator")
    ap.add_argument("input",help="EuRoC CSV or ZIP containing imu0/data.csv")
    ap.add_argument("-o","--output",default="estimated_trajectory.txt")
    ap.add_argument("--diagnostics",default="metadata.json")
    args=ap.parse_args()
    start=time.perf_counter()
    t,w,a=read_csv_any(args.input)
    ts,p,q,diag=estimate(t,w,a)
    write_tum(args.output,ts,p,q)
    diag["runtime_ms"]=round((time.perf_counter()-start)*1000,3)
    diag["input"]=os.path.basename(args.input)
    with open(args.diagnostics,"w",encoding="utf-8") as f: json.dump(diag,f,indent=2)
    print(f"PATHFINDER Sentinel 6D: wrote {len(ts)} poses -> {args.output}")
    print(f"duration={diag['duration_s']:.2f}s | gaps={diag['timestamp_gaps']} | blind={diag['blind_detected']} | runtime={diag['runtime_ms']:.2f}ms")
if __name__=="__main__": main()
