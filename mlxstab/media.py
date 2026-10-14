"""ffmpeg / ffprobe I/O. Interlaced sources are deinterlaced first with bwdif (send_field: 59.94i -> 59.94p)."""
import json, subprocess
import numpy as np


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,avg_frame_rate,field_order,nb_frames,duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    n, d = (s.get("avg_frame_rate") or s["r_frame_rate"]).split("/")
    fps = float(n) / float(d) if float(d) else 25.0
    fo = s.get("field_order", "unknown")
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": fps, "field_order": fo,
            "interlaced": fo in ("tt", "bb", "tb", "bt"), "duration": float(s.get("duration") or 0)}


def decode(path, info=None, start=None, duration=None, deinterlace=True, vf=None):
    """Yield RGB uint8 frames [H,W,3]. Returns (generator, effective_fps, (W,H))."""
    info = info or probe(path)
    filt = []
    if info["interlaced"] and deinterlace: filt.append("bwdif=mode=send_field:parity=auto:deint=all")
    if vf: filt.append(vf)
    fps = info["fps"] * (2 if info["interlaced"] and deinterlace else 1)
    if vf and vf.startswith("fps="):
        n_, _, d_ = vf[4:].partition("/"); fps = float(n_) / float(d_ or 1)
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start is not None: cmd += ["-ss", str(start)]
    cmd += ["-i", str(path)]
    if duration is not None: cmd += ["-t", str(duration)]
    cmd += ["-an", "-sn"] + (["-vf", ",".join(filt)] if filt else []) + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    W, H = info["width"], info["height"]

    def gen():
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        n = W * H * 3
        try:
            while True:
                b = p.stdout.read(n)
                if len(b) < n: break
                yield np.frombuffer(b, np.uint8).reshape(H, W, 3)
        finally:
            p.stdout.close(); p.terminate(); p.wait()
    return gen(), fps, (W, H)


class Encoder:
    """Pipe raw frames into ffmpeg. gray=True writes a single-channel (mask) stream, lossless."""
    def __init__(self, path, W, H, fps, gray=False, crf=14, preset="slow", out_fps=None):
        pf = "gray" if gray else "rgb24"
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", pf, "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-"]
        if out_fps: cmd += ["-r", f"{out_fps}"]
        if gray: cmd += ["-c:v", "libx264", "-qp", "0", "-pix_fmt", "gray"]
        else: cmd += ["-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p"]
        self.p = subprocess.Popen(cmd + [str(path)], stdin=subprocess.PIPE)
    def write(self, frame): self.p.stdin.write(np.ascontiguousarray(frame, np.uint8).tobytes())
    def close(self): self.p.stdin.close(); self.p.wait()
