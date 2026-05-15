"""跑 originals/ 里的图，扔线和拟合结果。
每张图丢到 showcase/<名>/：
  - original.<ext>      原图壳
  - outline.png         Canny 反色线
  - overview.png        四格乱看图
  - fits_only.png       只留拟合线
"""
import os
import shutil
import sys
import cv2
hh = os.path.dirname(os.path.abspath(__file__))
sv = os.path.dirname(hh)
sys.path.insert(0, sv)
from app import eo  # noqa: E402  -- Canny 反描壳
from curve_fitting_dev import build_fits, plot_overview, plot_fits_only  # noqa: E402
oo = os.path.join(hh, "originals")
md = 1600  # 大图先压，不然骨架发疯
def maybe_downscale(a, b):
    """a 到 b，太肥就按 md 缩。"""
    x = cv2.imread(a, cv2.IMREAD_COLOR)
    if x is None:
        raise ValueError(f"could not decode {a}")
    y, z = x.shape[:2]
    q = max(y, z)
    if q > md:
        r = md / q
        x = cv2.resize(x, (int(round(z * r)), int(round(y * r))),
                        interpolation=cv2.INTER_AREA)
    e = os.path.splitext(b)[1].lower()
    u = []
    if e in (".jpg", ".jpeg"):
        u = [cv2.IMWRITE_JPEG_QUALITY, 92]
    if not cv2.imwrite(b, x, u):
        raise IOError(f"failed to write {b}")
def process(sp):
    n, e = os.path.splitext(os.path.basename(sp))
    od = os.path.join(hh, n)
    os.makedirs(od, exist_ok=True)
    og = os.path.join(od, f"original{e.lower()}")
    ol = os.path.join(od, "outline.png")
    ov = os.path.join(od, "overview.png")
    fo = os.path.join(od, "fits_only.png")
    print(f"\n=== {n} ===")
    print(f"  copy/downscale → {os.path.relpath(og, hh)}")
    maybe_downscale(sp, og)
    print(f"  outline (Canny) → {os.path.relpath(ol, hh)}")
    eo(og, ol)
    print(f"  build_fits …")
    st = build_fits(ol,
                       bounds=(-10.0, 10.0, -7.0, 7.0),
                       min_seg_len=12,
                       min_sub_len=6,
                       corner_angle=55,
                       max_degree=12,
                       rel_tol=0.005)
    print(f"    raw segs={len(st['raw_segments'])} "
          f"refined={len(st['refined_segments'])} "
          f"fits={len(st['fits'])}")
    plot_overview(st, ov)
    plot_fits_only(st, fo)
    print(f"  → {os.path.relpath(ov, hh)}")
    print(f"  → {os.path.relpath(fo, hh)}")
def main():
    ss = sorted(
        os.path.join(oo, i)
        for i in os.listdir(oo)
        if not i.startswith(".")
        and os.path.splitext(i)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    )
    if not ss:
        print("no source images in originals/", file=sys.stderr)
        sys.exit(1)
    print(f"processing {len(ss)} image(s)")
    for p in ss:
        try:
            process(p)
        except Exception as ex:
            print(f"  !! {p}: {type(ex).__name__}: {ex}", file=sys.stderr)
if __name__ == "__main__":
    main()
