#!/usr/bin/env python3
"""PATHFINDER Sentinel 6D - robotics-style trajectory monitoring dashboard.

The estimator remains separate. This dashboard visualizes its TUM output and,
optionally, a ground-truth CSV for offline comparison. The office scene is an
illustrative simulated environment, not a map reconstructed from IMU data.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np


def load_tum(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                raise ValueError(f"{path}:{line_no}: expected 8 TUM columns")
            row = [float(v) for v in parts]
            if not np.all(np.isfinite(row)):
                raise ValueError(f"{path}:{line_no}: non-finite value")
            rows.append(row)
    if not rows:
        raise ValueError(f"No poses in {path}")
    a = np.asarray(rows, dtype=float)
    norms = np.linalg.norm(a[:, 4:8], axis=1)
    if np.any(norms < 1e-12):
        raise ValueError("Zero quaternion in estimated trajectory")
    a[:, 4:8] /= norms[:, None]
    return a


def load_euroc_groundtruth(path: Path) -> np.ndarray:
    """Read EuRoC state_groundtruth_estimate0/data.csv into TUM [t,p,qxyzw]."""
    import csv
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].lstrip().startswith("#") or row[0].strip().lower().startswith("timestamp"):
                continue
            if len(row) < 8:
                continue
            try:
                ns = float(row[0])
                px, py, pz = map(float, row[1:4])
                qw, qx, qy, qz = map(float, row[4:8])
            except ValueError:
                continue
            rows.append([ns * 1e-9, px, py, pz, qx, qy, qz, qw])
    if not rows:
        raise ValueError(f"No EuRoC ground-truth rows found in {path}")
    a = np.asarray(rows, dtype=float)
    a[:, 0] -= a[0, 0]
    return a


def quat_forward(q):
    x, y, z, w = q
    return np.array([1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y)])


def cuboid(ax, x, y, z, dx, dy, dz, color="#b7b7b7", alpha=1.0, edge="#555555"):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    p = np.array([
        [x,y,z],[x+dx,y,z],[x+dx,y+dy,z],[x,y+dy,z],
        [x,y,z+dz],[x+dx,y,z+dz],[x+dx,y+dy,z+dz],[x,y+dy,z+dz]
    ])
    faces = [[p[j] for j in face] for face in
             [(0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]]
    poly = Poly3DCollection(faces, facecolors=color, edgecolors=edge,
                            linewidths=0.45, alpha=alpha)
    ax.add_collection3d(poly)


def build_dashboard(est, gt=None, interval=50, repeat=True):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Rectangle, Circle, FancyArrowPatch

    black = "#111111"
    panel = "#1b1b1b"
    white = "#f4f1eb"
    grey = "#aaa49a"
    grid = "#48443e"
    amber = "#e05252"  # clear signal red for robot marker and heading
    warm_floor = "#bdb5a7"
    wall = "#ded8cd"
    wood = "#92775a"
    furniture = "#77736d"

    plt.rcParams.update({
        "figure.facecolor": black, "axes.facecolor": panel,
        "savefig.facecolor": black, "text.color": white,
        "axes.labelcolor": white, "xtick.color": grey, "ytick.color": grey,
        "font.family": "DejaVu Sans", "font.size": 8,
    })

    fig = plt.figure(figsize=(16, 9), facecolor=black)
    fig.subplots_adjust(left=.035, right=.985, bottom=.06, top=.89, wspace=.22, hspace=.28)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.55, 1.0, .82], height_ratios=[1.12, .88])
    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    axmap = fig.add_subplot(gs[0, 1])
    axcompare = fig.add_subplot(gs[1, 1])
    axinfo = fig.add_subplot(gs[:, 2])
    axinfo.set_axis_off()

    fig.text(.04, .955, "PATHFINDER  /  SENTINEL 6D", fontsize=15, weight="semibold", color=white)
    fig.text(.04, .925, "INERTIAL NAVIGATION  •  TRAJECTORY MONITOR", fontsize=8, color=grey)
    fig.text(.985, .955, "LOCAL DEMONSTRATION VIEW", ha="right", fontsize=8, color=grey)

    t = est[:, 0] - est[0, 0]
    xyz = est[:, 1:4]
    n = len(est)
    duration = float(t[-1]) if n > 1 else 0.0

    # 3D simulated office. Warm neutral materials; geometry is illustrative,
    # while the moving pose/path is read from the estimator output.
    ax3.set_title("3D ESTIMATED PATH  /  ROBOT POSITION", loc="left", color=white, pad=12, fontsize=9)
    ax3.set_xlim(-4, 4); ax3.set_ylim(-4, 4); ax3.set_zlim(0, 3)
    ax3.set_xlabel("X (m)", labelpad=2); ax3.set_ylabel("Y (m)", labelpad=2); ax3.set_zlabel("Z (m)", labelpad=2)
    ax3.tick_params(colors=grey, labelsize=7, pad=0)
    ax3.view_init(elev=31, azim=-54)
    try: ax3.set_box_aspect((1, 1, .58))
    except Exception: pass
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.pane.set_facecolor((.10,.10,.10,1))
        axis.pane.set_edgecolor("#777168")
        axis._axinfo["grid"]["color"] = (.40,.38,.34,.40)
        axis._axinfo["grid"]["linewidth"] = .45

    # Use the axis floor grid only; avoid a filled polygon that can depth-sort
    # in front of the furniture in Matplotlib's 3D renderer.
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    # Rear and side walls with restrained architectural finish.
    back_wall = Poly3DCollection([[(-4,4,0),(4,4,0),(4,4,3),(-4,4,3)]],
                                 facecolors=wall, edgecolors="#91897d", alpha=.28)
    side_wall = Poly3DCollection([[(-4,-4,0),(-4,4,0),(-4,4,3),(-4,-4,3)]],
                                 facecolors=wall, edgecolors="#91897d", alpha=.16)
    ax3.add_collection3d(back_wall); ax3.add_collection3d(side_wall)
    # Floor grid and skirting.
    for g in np.arange(-4,4.01,1):
        ax3.plot([g,g],[-4,4],[.012,.012],color="#91897d",lw=.35,alpha=.65)
        ax3.plot([-4,4],[g,g],[.012,.012],color="#91897d",lw=.35,alpha=.65)
    ax3.plot([-4,4],[3.96,3.96],[0,0],color="#766f64",lw=1.1)
    ax3.plot([-3.96,-3.96],[-4,4],[0,0],color="#766f64",lw=1.1)

    # Empty navigation test area: no desks, cabinets, partitions or obstacles.
    # The three synchronized views show only the room shell and estimated path.
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    for g in np.arange(-4,4.01,1):
        ax3.plot([g,g],[-4,4],[.012,.012],color="#77736b",lw=.42,alpha=.52)
        ax3.plot([-4,4],[g,g],[.012,.012],color="#77736b",lw=.42,alpha=.52)
    ax3.plot([-4,4],[3.96,3.96],[0,0],color="#aaa49a",lw=1.0)
    ax3.plot([-3.96,-3.96],[-4,4],[0,0],color="#aaa49a",lw=1.0)
    # trajectory is mapped into the illustrative scene coordinate frame for display only.
    pmin, pmax = xyz.min(axis=0), xyz.max(axis=0)
    center = (pmin+pmax)/2
    span_xy = max(float(np.max(pmax[:2]-pmin[:2])), 1e-6)
    # Scale the floor-plane path independently from vertical drift. A pure-IMU
    # estimate can accumulate very large Z error; allowing it to set the XY
    # scale would collapse the visible path to a dot.
    scale_xy = 4.25 / span_xy
    scene = np.empty_like(xyz)
    scene[:,0] = (xyz[:,0]-center[0])*scale_xy + .2
    scene[:,1] = (xyz[:,1]-center[1])*scale_xy - .1
    zlo, zhi = np.percentile(xyz[:,2], [5, 95])
    if zhi-zlo < 1e-6:
        scene[:,2] = .35
    else:
        z_norm = np.clip((xyz[:,2]-zlo)/(zhi-zlo), 0.0, 1.0)
        scene[:,2] = .18 + 1.75*z_norm
    # show entire estimate in dim grey; playback is brighter white
    ax3.plot(scene[:,0],scene[:,1],scene[:,2],color="#76664f",linewidth=1.15,alpha=.75)
    travelled, = ax3.plot([],[],[],color=amber,linewidth=2.6)
    robot_dot = ax3.scatter([scene[0,0]],[scene[0,1]],[scene[0,2]],s=155,c=amber,edgecolors="#fff0c2",linewidths=1.25,depthshade=False)
    # Draw one clean 3D heading: a single shaft plus one triangular tip.
    # Avoid 3D quiver's split/overlapping arrowhead rendering.
    def path_direction(points, idx, window=3):
        lo = max(0, idx-window)
        hi = min(len(points)-1, idx+window)
        vec = points[hi] - points[lo]
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            # Fall back to the nearest non-zero local displacement.
            for step in range(1, min(12, len(points))):
                j = min(len(points)-1, idx+step)
                vec = points[j] - points[idx]
                norm = np.linalg.norm(vec)
                if norm >= 1e-9:
                    break
        return vec / norm if norm >= 1e-9 else np.array([1.0, 0.0, 0.0])

    # Stable 3D heading arrow: construct the shaft and two arrowhead wings
    # entirely from world-coordinate line segments. No screen-space marker,
    # quiver glyph, or polygon patch is used, so the head stays attached and
    # aligned as the robot turns and the 3D camera projects it.
    def arrow_3d_points(points, idx, length=.82, head_length=.24, head_width=.15):
        direction = path_direction(points, idx)
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(direction, up)) > .92:
            up = np.array([0.0, 1.0, 0.0])
        side = np.cross(direction, up)
        side /= max(np.linalg.norm(side), 1e-12)
        tip = points[idx] + direction * length
        base = tip - direction * head_length
        left = base + side * head_width
        right = base - side * head_width
        return points[idx], tip, left, right

    tail0, tip0, left0, right0 = arrow_3d_points(scene, 0)
    arrow_shaft, = ax3.plot([tail0[0],tip0[0]],[tail0[1],tip0[1]],
                            [tail0[2],tip0[2]],color=amber,lw=2.8,
                            solid_capstyle="round",zorder=12)
    arrow_wing_a, = ax3.plot([tip0[0],left0[0]],[tip0[1],left0[1]],
                             [tip0[2],left0[2]],color=amber,lw=2.8,
                             solid_capstyle="round",zorder=13)
    arrow_wing_b, = ax3.plot([tip0[0],right0[0]],[tip0[1],right0[1]],
                             [tip0[2],right0[2]],color=amber,lw=2.8,
                             solid_capstyle="round",zorder=13)

    # Top-down schematic map: furniture shapes are illustrative, path comes from estimator.
    axmap.set_title("TOP-DOWN  /  HEADING ARROW", loc="left", color=white, fontsize=9, pad=8)
    axmap.set_xlim(-4,4); axmap.set_ylim(-4,4); axmap.set_aspect("equal")
    axmap.set_xticks([]); axmap.set_yticks([])
    for s in axmap.spines.values(): s.set_color("#414851")
    axmap.add_patch(Rectangle((-3.8,-3.8),7.6,7.6,fill=False,edgecolor="#aaa49a",linewidth=1.1))
    # Empty top-down plan: boundary only, no illustrative furniture.
    mapxy = scene[:, :2]
    mapline, = axmap.plot([],[],color=amber,lw=2.0)
    mapdot, = axmap.plot([mapxy[0,0]],[mapxy[0,1]],marker="o",ms=9,mec="#fff0c2",mew=1.0,mfc=amber,zorder=8)
    heading = FancyArrowPatch((0,0),(0,0),arrowstyle="-|>",mutation_scale=16,color=amber,lw=2.8)
    axmap.add_patch(heading)
    axmap.text(.02,.02,"SCHEMATIC ONLY  •  NOT SENSOR RECONSTRUCTION",
               transform=axmap.transAxes,color=grey,fontsize=6,va="bottom")

    # Comparison panel: estimated XY and optional reference XY, with explicit label.
    axcompare.set_title("ESTIMATED PATH  /  RAW XY (DRIFT VISIBLE)", loc="left", color=white, fontsize=9, pad=8)
    axcompare.set_xlabel("X (m)"); axcompare.set_ylabel("Y (m)")
    axcompare.grid(True,color=grid,linewidth=.55)
    axcompare.tick_params(labelsize=7,colors=grey)
    for s in axcompare.spines.values(): s.set_color("#414851")
    axcompare.plot(xyz[:,0],xyz[:,1],color="#77716a",lw=.8,alpha=.75,label="Estimated (complete)")
    estline, = axcompare.plot([],[],color=amber,lw=1.8,label="Estimated (playback)")
    gtline = None
    if gt is not None:
        gtline, = axcompare.plot(gt[:,1],gt[:,2],color="#ded8cd",lw=1.0,ls="--",label="Reference (synthetic GT)")
    axcompare.legend(loc="best",fontsize=6,facecolor=panel,edgecolor="#414851",labelcolor=white)
    axcompare.set_aspect("equal", adjustable="datalim")

    # Info panel with compact engineering readouts.
    axinfo.set_facecolor(panel)
    axinfo.add_patch(Rectangle((0,0),1,1,transform=axinfo.transAxes,facecolor=panel,edgecolor="#414851",lw=.8))
    axinfo.text(.08,.95,"SYSTEM STATUS",transform=axinfo.transAxes,color=white,fontsize=10,weight="semibold",va="top")
    axinfo.text(.08,.90,"ESTIMATOR OUTPUT  /  TUM",transform=axinfo.transAxes,color=grey,fontsize=7,va="top")
    readout = axinfo.text(.08,.82,"",transform=axinfo.transAxes,color=white,fontsize=9,
                          family="DejaVu Sans Mono",va="top",linespacing=1.8)
    axinfo.text(.08,.30,"DATA SOURCES",transform=axinfo.transAxes,color=white,fontsize=9,weight="semibold")
    axinfo.text(.08,.25,f"IMU trajectory\n{n:,} poses  •  {duration:.2f} s\n\n"
                + (f"Reference trajectory\n{len(gt):,} poses" if gt is not None else "Reference trajectory\nNot loaded"),
                transform=axinfo.transAxes,color=grey,fontsize=8,va="top",linespacing=1.6)
    axinfo.text(.08,.06,"Environment geometry is illustrative.\nOnly the white path is estimator output.",
                transform=axinfo.transAxes,color=grey,fontsize=7,va="bottom",linespacing=1.5)

    def update(i):
        nonlocal robot_dot
        travelled.set_data(scene[:i+1,0],scene[:i+1,1])
        travelled.set_3d_properties(scene[:i+1,2])
        robot_dot.remove()
        robot_dot = ax3.scatter([scene[i,0]],[scene[i,1]],[scene[i,2]],s=155,c=amber,edgecolors="#fff0c2",linewidths=1.25,depthshade=False)
        tail, tip, left, right = arrow_3d_points(scene, i)
        arrow_shaft.set_data_3d([tail[0],tip[0]],[tail[1],tip[1]],[tail[2],tip[2]])
        arrow_wing_a.set_data_3d([tip[0],left[0]],[tip[1],left[1]],[tip[2],left[2]])
        arrow_wing_b.set_data_3d([tip[0],right[0]],[tip[1],right[1]],[tip[2],right[2]])
        mapline.set_data(mapxy[:i+1,0],mapxy[:i+1,1])
        mapdot.set_data([mapxy[i,0]],[mapxy[i,1]])
        estline.set_data(xyz[:i+1,0],xyz[:i+1,1])
        # Top-down arrow follows the same estimated-path tangent as the 3D arrow.
        map_dir = path_direction(mapxy, i)
        fx, fy = map_dir[0], map_dir[1]
        heading.set_positions((mapxy[i,0],mapxy[i,1]),
                              (mapxy[i,0]+fx*.55,mapxy[i,1]+fy*.55))
        elapsed = float(t[i])
        readout.set_text(
            f"PLAYBACK   {i+1:>5,} / {n:,}\n"
            f"TIME       {elapsed:>7.2f} s\n"
            f"DURATION   {duration:>7.2f} s\n\n"
            f"POSITION\n"
            f"X          {xyz[i,0]:>8.3f} m\n"
            f"Y          {xyz[i,1]:>8.3f} m\n"
            f"Z          {xyz[i,2]:>8.3f} m\n\n"
            f"ORIENTATION\n"
            f"qx         {est[i,4]:>8.3f}\n"
            f"qy         {est[i,5]:>8.3f}\n"
            f"qz         {est[i,6]:>8.3f}\n"
            f"qw         {est[i,7]:>8.3f}"
        )
        return travelled, robot_dot, arrow_shaft, arrow_wing_a, arrow_wing_b, mapline, mapdot, heading, estline, readout

    anim = FuncAnimation(fig, update, frames=n, interval=interval, repeat=repeat,
                         blit=False, cache_frame_data=False)
    fig._pathfinder_animation = anim
    update(0)
    return fig, anim


def main():
    p = argparse.ArgumentParser(description="PATHFINDER robotics-style trajectory dashboard")
    p.add_argument("trajectory", nargs="?", default="estimated_trajectory.txt")
    p.add_argument("--ground-truth", help="optional EuRoC state_groundtruth_estimate0/data.csv")
    p.add_argument("--interval", type=int, default=50, help="playback frame interval in ms (default 50 ms for 15 s playback)")
    p.add_argument("--stride", type=int, default=10, help="render every Nth pose (default 10 gives 20 Hz display from 200 Hz data)")
    p.add_argument("--no-repeat", action="store_true")
    args = p.parse_args()
    if args.interval < 1 or args.stride < 1:
        p.error("--interval and --stride must be positive")
    try:
        est = load_tum(Path(args.trajectory))
        gt = load_euroc_groundtruth(Path(args.ground_truth)) if args.ground_truth else None
        if args.stride > 1:
            # Preserve the final sample so the visual playback lands exactly at 15.00 s.
            sampled = est[::args.stride]
            if sampled[-1,0] != est[-1,0]:
                sampled = np.vstack([sampled, est[-1]])
            est = sampled
        fig, anim = build_dashboard(est, gt, args.interval, not args.no_repeat)
        print(f"Loaded {len(est):,} display poses.")
        print(f"Playback duration: {(est[-1,0]-est[0,0]):.2f} seconds at {1000/args.interval:.1f} frames/s.")
        print("The simulated office is illustrative; trajectory is read from estimator output.")
        import matplotlib.pyplot as plt
        plt.show()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
