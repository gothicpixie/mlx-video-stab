"""ffmpeg / ffprobe I/O. Interlaced sources are deinterlaced first with bwdif (send_field: 59.94i -> 59.94p)."""
import json, subprocess
import numpy as np


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,avg_frame_rate,field_order,nb_frames,duration,start_time,time_base,codec_name,pix_fmt", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    n, d = (s.get("avg_frame_rate") or s["r_frame_rate"]).split("/")
    fps = float(n) / float(d) if float(d) else 25.0
    fo = s.get("field_order", "unknown")
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": fps, "field_order": fo,
            "interlaced": fo in ("tt", "bb", "tb", "bt"), "duration": float(s.get("duration") or 0),
            "start_time": float(s.get("start_time") or 0), "time_base": s.get("time_base"),
            "codec_name": s.get("codec_name"), "pix_fmt": s.get("pix_fmt")}


def frame_pts(path):
    """Frame presentation timestamps from the encoded stream (ffprobe; no re-encoding).

    Returns ``{'count', 'n_pts', 'pts_s', 'first_pts_s', 'last_pts_s', 'monotonic'}``. ``count`` is the number of
    frames; ``n_pts`` is how many had a usable ``pts_time``. Missing/``N/A`` PTS are not invented. Used to
    validate the encoded output and mask timelines without re-encoding or re-inferring anything.
    """
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pts_time",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    frames = json.loads(out).get("frames", [])
    pts = []
    for f in frames:
        v = f.get("pts_time")
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = None
        pts.append(v if v is not None and np.isfinite(v) else None)
    valid = [v for v in pts if v is not None]
    return {"count": len(frames), "n_pts": len(valid), "pts_s": pts,
            "first_pts_s": pts[0] if pts else None, "last_pts_s": pts[-1] if pts else None,
            "monotonic": len(valid) == len(pts) and all(a < b for a, b in zip(valid, valid[1:]))}


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
    """Pipe raw frames into ffmpeg. gray=True writes a single-channel (mask) stream, lossless.

    ``out_fps`` converts to a constant output frame rate by resampling the raw input explicitly
    (``-vf fps=... -r ... -fps_mode cfr``). Relying on output ``-r`` alone duplicated/dropped frames
    unpredictably (60 fps -> 30 fps produced 32 frames for 1 s); the filter path is exact. Frame counts are
    validated by the caller (see ``tests/test_media_cadence.py``).

    Failure handling: a stdin write failure (e.g. ffmpeg exited early -> ``BrokenPipeError``) is stored on the
    writer thread, which then keeps consuming and discarding queued frames until the close sentinel. That keeps
    the bounded queue from wedging a producer that is still writing, and lets ``write``/``drain``/``close``
    surface the first failure instead of hanging or dropping it. ``close`` always sends the sentinel, joins the
    writer, closes stdin and waits for the child, then raises the first writer/close/nonzero-exit failure.
    """
    QUEUE_SIZE = 6

    def __init__(self, path, W, H, fps, gray=False, crf=14, preset="slow", out_fps=None):
        pf = "gray" if gray else "rgb24"
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", pf, "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-"]
        if out_fps:
            cmd += ["-vf", f"fps={out_fps}", "-r", f"{out_fps}", "-fps_mode", "cfr"]
        if gray: cmd += ["-c:v", "libx264", "-qp", "0", "-pix_fmt", "gray"]
        else: cmd += ["-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p"]
        import queue, threading
        self.p = subprocess.Popen(cmd + [str(path)], stdin=subprocess.PIPE)
        self.q = queue.Queue(self.QUEUE_SIZE)                     # writer thread: encoding overlaps with the next frame's GPU work
        self._writer_error = None                                # first stdin failure, kept for write/drain/close
        self._closed = False
        self._failure = None                                     # first failure seen by close(), for idempotent re-raise
        def run():
            while True:
                b = self.q.get()
                try:
                    if b is None: return                         # close sentinel: the only way the writer stops
                    if self._writer_error is None:
                        try:
                            self.p.stdin.write(b)
                        except BaseException as exc:
                            self._writer_error = exc             # keep consuming/discarding so writes cannot deadlock
                finally:
                    self.q.task_done()                           # unblock drain() as each frame is handled
        self.t = threading.Thread(target=run, daemon=True); self.t.start()
    def write(self, frame):
        if self._closed: raise RuntimeError("Encoder is closed")
        if self._writer_error is not None: raise self._writer_error
        self.q.put(np.ascontiguousarray(frame, np.uint8).tobytes())
    def drain(self):
        """Barrier: block until every queued frame has been handed to ffmpeg's stdin (queue empty, writer idle).

        Lets a chunked shared-lock yield release the lock only after the writer thread has consumed the
        completed chunk, instead of leaving up to ``Queue(QUEUE_SIZE)`` frames queued. Raises the writer's
        exception (e.g. a broken pipe) rather than hanging when the writer has already failed. Note this drains
        only this queue and the stdin buffer: ffmpeg/x264 internal frame buffering still spans the yield.
        """
        if self._closed: raise RuntimeError("Encoder is closed")
        self.q.join()
        if self._writer_error is not None: raise self._writer_error
        try:
            self.p.stdin.flush()
        except BaseException as exc:
            if self._writer_error is None: self._writer_error = exc
            raise
    def close(self):
        """Finish the stream: send the sentinel, join the writer, close stdin and wait for ffmpeg.

        Returns ``None`` on success. Raises the first stored writer failure, close failure or nonzero ffmpeg
        exit status (``subprocess.CalledProcessError``). The writer survives a stdin failure, so the sentinel
        always reaches it. Idempotent: a second call returns, or re-raises the same failure, without touching
        the stream again.
        """
        if self._closed:
            if self._failure is not None: raise self._failure
            return
        self._closed = True
        close_error = None
        try:
            self.q.put(None)                                     # the writer stays alive after a stdin failure
        except BaseException as exc:
            if close_error is None: close_error = exc
        try:
            self.t.join()
        except BaseException as exc:
            if close_error is None: close_error = exc
        try:
            self.p.stdin.close()
        except BaseException as exc:
            if close_error is None: close_error = exc
        returncode = None
        try:
            returncode = self.p.wait()
        except BaseException as exc:
            if close_error is None: close_error = exc
        failure = self._writer_error
        if failure is None: failure = close_error
        if failure is None and isinstance(returncode, int) and returncode != 0:
            failure = subprocess.CalledProcessError(returncode, getattr(self.p, "args", None))
        self._failure = failure
        if failure is not None: raise failure


def close_all(encoders):
    """Close every encoder, attempting all of them even if one fails, then raise the first failure.

    Multi-encoder callers own several ``Encoder`` objects whose ``close`` can now fail (a writer error or a
    nonzero ffmpeg exit). Closing them one-by-one with a bare loop would skip every sibling after the first
    failure; this helper closes all of them and then re-raises the first failure, so a caller on the success
    path reports a single stable error without leaking a process or writer thread. ``None`` entries are
    skipped so a call site can pass an optional encoder (e.g. ``(video, mask)``) directly. Cleanup paths that
    must preserve a primary exception should call this inside ``try``/``except`` and suppress its failure.
    """
    first = None
    for encoder in encoders:
        if encoder is None:
            continue
        try:
            encoder.close()
        except BaseException as exc:  # noqa: BLE001 - finish closing siblings, surface the first failure
            if first is None:
                first = exc
    if first is not None:
        raise first


def prefetch(gen, n=4):
    """Run a frame generator in a thread so decoding overlaps with GPU/CPU work."""
    import queue, threading
    q = queue.Queue(n); end = object()
    def run():
        try:
            for x in gen: q.put(x)
        finally: q.put(end)
    threading.Thread(target=run, daemon=True).start()
    while True:
        x = q.get()
        if x is end: return
        yield x
