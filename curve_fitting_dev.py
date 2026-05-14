"""曲线糊。
图进式出,差不多。
招:骨,叉,弯,两试,低先。
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
# 图骨
def load_binary(path, thresh=128):
    a = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise FileNotFoundError(path)
    hh, ww = a.shape
    b = cv2.bitwise_not(a)
    _, c = cv2.threshold(b, thresh, 255, cv2.THRESH_BINARY)
    return c, (hh, ww)
def to_skeleton(binary):
    return skeletonize(binary > 0).astype(np.uint8)
# 邻
NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1),           (0, 1),
             (1, -1),  (1, 0),  (1, 1)]
def neighbor_count(skel):
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    c = cv2.filter2D(skel, -1, k)
    c[skel == 0] = 0
    return c
def trace_segments(skel, min_len=15):
    """爬。"""
    hh, ww = skel.shape
    n = neighbor_count(skel)
    bm = n >= 3
    em = n == 1
    # 去叉
    z = skel.copy()
    z[bm] = 0
    mk = np.zeros_like(z, dtype=bool)
    def walk(sy, sx):
        p = [(sy, sx)]
        mk[sy, sx] = True
        # 两头
        for _ in range(2):
            while True:
                cy, cx = p[-1]
                st = None
                # 先直
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1),
                               (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    ny, nx = cy + dy, cx + dx
                    if (0 <= ny < hh and 0 <= nx < ww
                            and z[ny, nx] and not mk[ny, nx]):
                        st = (ny, nx)
                        break
                if st is None:
                    break
                mk[st] = True
                p.append(st)
            p.reverse()
        return p
    sg = []
    # 端
    ys, xs = np.where(em)
    for sy, sx in zip(ys, xs):
        if mk[sy, sx] or z[sy, sx] == 0:
            continue
        p = walk(sy, sx)
        if len(p) >= min_len:
            sg.append(np.array([[x, y] for y, x in p], dtype=np.float64))
    # 余
    ys, xs = np.where(z & ~mk)
    for sy, sx in zip(ys, xs):
        if mk[sy, sx]:
            continue
        p = walk(sy, sx)
        if len(p) >= min_len:
            sg.append(np.array([[x, y] for y, x in p], dtype=np.float64))
    return sg
# 切弯
def split_at_corners(pts, angle_thresh_deg=55, smooth=4, min_sub=8):
    """折切。"""
    ll = len(pts)
    if ll < 2 * smooth + 4:
        return [pts]
    # 糙向
    ts = []
    for i in range(smooth, ll - smooth):
        d = pts[i + smooth] - pts[i - smooth]
        rn = np.linalg.norm(d)
        ts.append(d / rn if rn > 0 else np.array([1.0, 0.0]))
    ts = np.array(ts)
    if len(ts) < 3:
        return [pts]
    cs = np.clip(np.sum(ts[:-1] * ts[1:], axis=1), -1.0, 1.0)
    ag = np.degrees(np.arccos(cs))
    ct = []
    for i in range(1, len(ag) - 1):
        if (ag[i] > angle_thresh_deg
                and ag[i] >= ag[i - 1] and ag[i] >= ag[i + 1]):
            ct.append(i + smooth)  # 位回
    if not ct:
        return [pts]
    sb, la = [], 0
    for c in ct:
        s = pts[la:c + 1]
        if len(s) >= min_sub:
            sb.append(s)
        la = c
    tl = pts[la:]
    if len(tl) >= min_sub:
        sb.append(tl)
    return sb if sb else [pts]
def split_at_turning(pts, min_sub=8):
    """拐断。"""
    ll = len(pts)
    if ll < 5:
        return [pts]
    ax = 0 if np.ptp(pts[:, 0]) >= np.ptp(pts[:, 1]) else 1
    vv = pts[:, ax]
    dz = np.diff(vv)
    if np.all(dz == 0):
        return [pts]
    # 抖
    ww = max(3, ll // 50 | 1)  # 奇
    if ww >= 3 and ll > ww:
        kr = np.ones(ww) / ww
        dd = np.convolve(dz, kr, mode="same")
    else:
        dd = dz
    sn = np.sign(dd)
    # 零借
    la = 0
    for i in range(len(sn)):
        if sn[i] == 0:
            sn[i] = la
        else:
            la = sn[i]
    ct = []
    for i in range(1, len(sn)):
        if sn[i] != 0 and sn[i - 1] != 0 and sn[i] != sn[i - 1]:
            ct.append(i)
    if not ct:
        return [pts]
    sb, la = [], 0
    for c in ct:
        s = pts[la:c + 1]
        if len(s) >= min_sub:
            sb.append(s)
        la = c
    tl = pts[la:]
    if len(tl) >= min_sub:
        sb.append(tl)
    return sb if sb else [pts]
# 坐换
def pixel_to_math(pts_px, img_shape, bounds):
    x0, x1, y0, y1 = bounds
    hh, ww = img_shape
    xp, yp = pts_px[:, 0], pts_px[:, 1]
    xm = xp / ww * (x1 - x0) + x0
    ym = (hh - yp) / hh * (y1 - y0) + y0
    return np.column_stack([xm, ym])
def _fit_one_orient(u, v, max_degree, rel_tol, noise_floor):
    """单向低先。"""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    us = float(np.ptp(u))
    vs = max(float(np.ptp(v)), 1e-9)
    ta = rel_tol * vs
    ll = len(u)
    if ll < 3 or us < 1e-9:
        return None
    hs = []
    for d in range(1, min(max_degree, ll - 1) + 1):
        try:
            p = np.polynomial.Polynomial.fit(u, v, d, domain=[u.min(), u.max()])
        except (np.linalg.LinAlgError, ValueError):
            continue
        # 系
        st = p.convert(domain=[-1, 1], window=[-1, 1])
        ca = st.coef
        cd = ca[::-1]
        rs = v - np.polyval(cd, u)
        me = float(np.max(np.abs(rs)))
        ms = float(np.mean(rs ** 2))
        hs.append(dict(coeffs=cd, degree=d, max_err=me, mse=ms))
        if me < ta:
            break
    if not hs:
        return None
    # 低过
    ok = [h for h in hs if h["max_err"] < ta]
    if ok:
        ch = min(ok, key=lambda h: h["degree"])
    else:
        be = min(h["max_err"] for h in hs)
        tg = be + max(noise_floor, rel_tol * vs)
        ch = next((h for h in hs if h["max_err"] <= tg), hs[-1])
    return dict(
        coeffs=ch["coeffs"],
        degree=ch["degree"],
        mse=ch["mse"],
        max_err=ch["max_err"],
        u_min=float(u.min()),
        u_max=float(u.max()),
        u_span=us,
        v_span=vs,
    )
def fit_auto_degree(pts_m, max_degree=10, rel_tol=0.005, noise_floor=0.025):
    """两向挑。"""
    x, y = pts_m[:, 0], pts_m[:, 1]
    if len(x) < 3:
        return None
    op = []
    if np.ptp(x) > 1e-6:
        op.append(("y=f(x)", x, y))
    if np.ptp(y) > 1e-6:
        op.append(("x=f(y)", y, x))
    best = None
    for ori, u, v in op:
        ft = _fit_one_orient(u, v, max_degree, rel_tol, noise_floor)
        if ft is None:
            continue
        ft["orient"] = ori
        # 长自
        wd = 0 if ft["u_span"] >= ft["v_span"] else 1
        fk = (wd, ft["max_err"], ft["degree"])
        if best is None or fk < best[0]:
            best = (fk, ft)
    return best[1] if best else None
def evaluate_curve(fit, n=300):
    u = np.linspace(fit["u_min"], fit["u_max"], n)
    v = np.polyval(fit["coeffs"], u)
    if fit["orient"] == "y=f(x)":
        return np.column_stack([u, v])
    return np.column_stack([v, u])
# 管跑
def _arc_length(pts):
    """折长。"""
    if len(pts) < 2:
        return 0.0
    d = np.diff(pts, axis=0)
    return float(np.sum(np.sqrt(np.sum(d * d, axis=1))))
def _bbox_diag(pts):
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
def build_fits(outline_path,
               bounds=(-10.0, 10.0, -7.0, 7.0),
               min_seg_len=12,
               min_sub_len=6,
               corner_angle=55,
               max_degree=10,
               rel_tol=0.005,
               noise_floor=None):
    bi, sh = load_binary(outline_path)
    sk = to_skeleton(bi)
    raw = trace_segments(sk, min_len=min_seg_len)
    # 噪px
    if noise_floor is None:
        x0, x1, _, _ = bounds
        noise_floor = (x1 - x0) / sh[1] * 1.5
    rf = []
    for sg in raw:
        for s1 in split_at_corners(sg, angle_thresh_deg=corner_angle,
                                   min_sub=min_sub_len):
            for s2 in split_at_turning(s1, min_sub=min_sub_len):
                if len(s2) >= min_sub_len:
                    rf.append(s2)
    rf.sort(key=len, reverse=True)
    fs = []
    for sg in rf:
        pm = pixel_to_math(sg, sh, bounds)
        ft = fit_auto_degree(pm, max_degree=max_degree,
                             rel_tol=rel_tol, noise_floor=noise_floor)
        if ft is None:
            continue
        # 糊权
        ar = _arc_length(pm)
        dg = _bbox_diag(pm)
        im = ar + 0.5 * dg
        fs.append({
            "seg_px": sg,
            "pts_m": pm,
            "fit": ft,
            "arc": ar,
            "diag": dg,
            "importance": im,
        })
    # 大先
    fs.sort(key=lambda f: f["importance"], reverse=True)
    return {
        "img_shape": sh,
        "skeleton": sk,
        "raw_segments": raw,
        "refined_segments": rf,
        "fits": fs,
        "bounds": bounds,
    }
def plot_overview(state, out_path, show_points=False):
    bd = state["bounds"]
    x0, x1, y0, y1 = bd
    fs = state["fits"]
    fig, axs = plt.subplots(2, 2, figsize=(16, 11))
    # 骨
    ax = axs[0, 0]
    ax.imshow(state["skeleton"], cmap="gray_r")
    ax.set_title(f"Skeleton ({int(state['skeleton'].sum())} px)")
    ax.axis("off")
    # 段
    ax = axs[0, 1]
    hh, ww = state["img_shape"]
    ax.set_xlim(0, ww)
    ax.set_ylim(hh, 0)
    ax.set_aspect("equal")
    ax.set_title(f"Refined segments ({len(state['refined_segments'])})")
    cm = plt.cm.tab20
    for i, sg in enumerate(state["refined_segments"]):
        ax.plot(sg[:, 0], sg[:, 1],
                color=cm(i % 20), lw=0.9)
    ax.set_facecolor("white")
    # 点拟
    ax = axs[1, 0]
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_facecolor("#fafafa")
    ax.set_title(f"Fits + segment points ({len(fs)} curves)")
    for e in fs:
        pm = e["pts_m"]
        cv = evaluate_curve(e["fit"])
        ax.scatter(pm[:, 0], pm[:, 1], s=0.6,
                   color="0.7", alpha=0.4)
        ax.plot(cv[:, 0], cv[:, 1], lw=1.4, alpha=0.9,
                color="#c0392b")
    # 纯线
    ax = axs[1, 1]
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.set_title(f"Fits only ({len(fs)} curves) — judges shape")
    for e in fs:
        cv = evaluate_curve(e["fit"])
        ax.plot(cv[:, 0], cv[:, 1], color="black",
                lw=1.0, alpha=0.95)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
def plot_fits_only(state, out_path, top_n=None):
    bd = state["bounds"]
    x0, x1, y0, y1 = bd
    es = state["fits"][:top_n] if top_n else state["fits"]
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.set_title(f"Polynomial fits only — top {len(es)} of "
                 f"{len(state['fits'])} curves")
    for e in es:
        cv = evaluate_curve(e["fit"])
        ax.plot(cv[:, 0], cv[:, 1], color="black",
                lw=1.0, alpha=0.95)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
def degree_histogram(state):
    from collections import Counter
    return Counter(e["fit"]["degree"] for e in state["fits"])
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "outline_dexined.jpg"
    print(f"Loading {path} ...")
    state = build_fits(path,
                       bounds=(-10.0, 10.0, -7.0, 7.0),
                       min_seg_len=12,
                       min_sub_len=6,
                       corner_angle=55,
                       max_degree=12,
                       rel_tol=0.005)
    print(f"  raw segments    : {len(state['raw_segments'])}")
    print(f"  refined segments: {len(state['refined_segments'])}")
    print(f"  fitted          : {len(state['fits'])}")
    h = degree_histogram(state)
    print(f"  degree histogram: {dict(sorted(h.items()))}")
    if state["fits"]:
        im = [f["importance"] for f in state["fits"]]
        print(f"  importance: top={im[0]:.2f} median={im[len(im)//2]:.2f} "
              f"min={im[-1]:.2f}")
    plot_overview(state, "debug_overview.png")
    plot_fits_only(state, "curve_fits_only.png")
    # 前验
    for k in (10, 20, 40):
        plot_fits_only(state, f"curve_fits_top{k}.png", top_n=k)
    print("Saved debug_overview.png and curve_fits_only.png + top-N variants")
if __name__ == "__main__":
    main()
