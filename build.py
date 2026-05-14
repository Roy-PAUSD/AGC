"""预烤离线static.html(从outline_dexined.jpg).交互页(index.html+app.py)不用此输出—它拟合用户上传.static.html是默认轮廓独立离线演示.改源轮廓或调管线后跑:.venv/bin/python server/build.py"""
import json
import os
import sys
import warnings
import numpy as n

try:
    _rw = n.exceptions.RankWarning      # np≥2.0
except AttributeError:
    _rw = n.RankWarning                 # np<2.0
warnings.filterwarnings("ignore", category=_rw)
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
from curve_fitting_dev import build_fits  # noqa: E402
OP = os.path.join(H, "outline_dexined.jpg")
SP = os.path.join(H, "static.html")
STP = os.path.join(H, "static.template.html")
BD = (-10.0, 10.0, -7.0, 7.0)
DTN = 40
MD = 12
DC = [
    "#c74440", "#2d70b3", "#388c46", "#6042a6", "#000000",
    "#fa7e19", "#c55c1e", "#3d8545", "#b0407c", "#c78433",
]




def ctl(cs, ot, dm=6):
    """对应index.html中JS的coeffsToLatex,服务端版."""
    lh = "y" if ot == "y=f(x)" else "x"
    v = "x" if ot == "y=f(x)" else "y"
    d = len(cs) - 1
    ts = []
    for i, c in enumerate(cs):
        if abs(c) < 1e-12:
            continue
        p = d - i
        sg = "" if (c >= 0 and not ts) else ("+" if c >= 0 else "-")
        vl = f"{abs(c):.{dm}f}"
        if p == 0:
            ts.append(f"{sg}{vl}")
        elif p == 1:
            ts.append(f"{sg}{vl}{v}")
        else:
            ts.append(f"{sg}{vl}{v}^{{{p}}}")
    return f"{lh}={''.join(ts) if ts else '0'}"








def bcs(fd, bd):
    """返字典匹配calculator.getState()/setState() schema."""
    xn, xx, yn, yx = bd
    es = []
    for f in fd:
        eq = ctl(f["coeffs"], f["orient"])
        u = "x" if f["orient"] == "y=f(x)" else "y"
        eq += (f"\\left\\{{{f['u_min']:.6f}\\le {u}\\le "
               f"{f['u_max']:.6f}\\right\\}}")
        es.append({
            "type": "expression",
            "id": f["id"],
            "latex": eq,
            "color": f["color"],
            "lineWidth": 2.0,
        })
    return {
        "version": 10,
        "graph": {
            "viewport": {"xmin": xn, "xmax": xx,
                         "ymin": yn, "ymax": yx},
            "squareAxes": True,
        },
        "expressions": {"list": es},
    }



def sf(st):
    fs = st["fits"]
    o = []
    for i, e in enumerate(fs):
        f = e["fit"]
        o.append({
            "id": f"c{i}",
            "coeffs": [float(c) for c in f["coeffs"]],
            "orient": f["orient"],
            "degree": int(f["degree"]),
            "u_min": float(f["u_min"]),
            "u_max": float(f["u_max"]),
            "max_err": float(f["max_err"]),
            "arc": float(e["arc"]),
            "importance": float(e["importance"]),
            "color": DC[i % len(DC)],
        })
    return o



def main():
    print(f"Loading {OP} ...")
    st = build_fits(OP, bounds=BD,
                       max_degree=MD, rel_tol=0.005)
    fd = sf(st)
    print(f"  {len(fd)} fits ready")
    nt = len(fd)
    dtn = min(DTN, nt)
    sd = bcs(fd[:dtn], BD)
    with open(STP) as f:
        st_tpl = f.read()
    sh = st_tpl.replace("__STATE_JSON__", json.dumps(sd))
    with open(SP, "w") as f:
        f.write(sh)
    print(f"Wrote {SP}")
if __name__ == "__main__":
    main()
