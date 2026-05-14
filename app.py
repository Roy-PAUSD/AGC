"""微HTTP服:供前端+拟合上传图.
运行:.venv/bin/python server/app.py 或 PORT=9000 ...
端点:GET/->index.html;GET/static.html->离线页;POST/api/fit->原图字节,Canny边缘+反转+build_fits,返JSON.
POST体为原图(非multipart)."""
import base64
import json
import os
import sys
import tempfile
import warnings
from http.server import BaseHTTPRequestHandler as B, ThreadingHTTPServer as T
import cv2
import numpy as n
try:
    _rw = n.exceptions.RankWarning
except AttributeError:
    _rw = n.RankWarning
warnings.filterwarnings("ignore", category=_rw)
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
from curve_fitting_dev import build_fits  # noqa: E402
BD = (-10.0, 10.0, -7.0, 7.0)
MD = 12
DC = [
    "#c74440", "#2d70b3", "#388c46", "#6042a6", "#000000",
    "#fa7e19", "#c55c1e", "#3d8545", "#b0407c", "#c78433",
]
IP = os.path.join(H, "index.html")
SP = os.path.join(H, "static.html")
MUB = 20 * 1024 * 1024
def eo(rp, op):
    """原图→白底黑线轮廓供build_fits用.灰度→高斯模糊→Canny→反转."""
    im = cv2.imread(rp, cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError("could not decode image")
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    bl = cv2.GaussianBlur(g, (5, 5), 0)
    ed = cv2.Canny(bl, threshold1=30, threshold2=100)
    ol = cv2.bitwise_not(ed)
    if not cv2.imwrite(op, ol):
        raise IOError(f"failed to write outline to {op}")
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
def fp(ipx):
    st = build_fits(ipx, bounds=BD,
                       max_degree=MD, rel_tol=0.005)
    with open(ipx, "rb") as f:
        ob64 = base64.b64encode(f.read()).decode("ascii")
    return {
        "fits": sf(st),
        "bounds": list(BD),
        "max_degree": MD,
        "outline_png_b64": ob64,
    }
class Hd(B):
    def _sb(self, s, b, ct):
        self.send_response(s)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)
    def _sj(self, s, pl):
        b = json.dumps(pl).encode("utf-8")
        self._sb(s, b, "application/json; charset=utf-8")
    def _sfi(self, p, ct):
        try:
            with open(p, "rb") as f:
                b = f.read()
        except FileNotFoundError:
            self.send_error(404, f"{os.path.basename(p)} not found")
            return
        self._sb(200, b, ct)
    def do_GET(self):
        p = self.path.split("?", 1)[0]
        if p in ("/", "/index.html"):
            self._sfi(IP, "text/html; charset=utf-8")
        elif p == "/static.html":
            self._sfi(SP, "text/html; charset=utf-8")
        else:
            self.send_error(404)
    def do_POST(self):
        if self.path != "/api/fit":
            self.send_error(404)
            return
        try:
            ln = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400, "missing Content-Length")
            return
        if ln <= 0:
            self.send_error(400, "empty body")
            return
        if ln > MUB:
            self.send_error(413, "image too large")
            return
        d = self.rfile.read(ln)
        ct = self.headers.get("Content-Type", "")
        ex = ".png"
        if "jpeg" in ct or "jpg" in ct:
            ex = ".jpg"
        elif "bmp" in ct:
            ex = ".bmp"
        elif "webp" in ct:
            ex = ".webp"
        rfd, rp = tempfile.mkstemp(suffix=ex)
        ofd, op = tempfile.mkstemp(suffix=".png")
        os.close(ofd)
        try:
            with os.fdopen(rfd, "wb") as f:
                f.write(d)
            try:
                eo(rp, op)
                pl = fp(op)
            except (ValueError, FileNotFoundError) as e:
                self._sj(400, {"error": str(e)})
                return
            except Exception as e:
                self._sj(500, {"error": f"{type(e).__name__}: {e}"})
                return
            self._sj(200, pl)
        finally:
            for px in (rp, op):
                try:
                    os.remove(px)
                except OSError:
                    pass
    def log_message(self, fmt, *args):
        sys.stderr.write(f"[{self.log_date_time_string()}] {fmt % args}\n")
def main():
    pt = int(os.environ.get("PORT", "8000"))
    ht = os.environ.get("HOST", "127.0.0.1")
    print(f"serving on http://{ht}:{pt}")
    T((ht, pt), Hd).serve_forever()
if __name__ == "__main__":
    main()
