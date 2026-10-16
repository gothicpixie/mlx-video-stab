"""ffmpeg / ffprobe I/O. Interlaced sources are deinterlaced first with bwdif (send_field: 59.94i -> 59.94p)."""
import json, os, secrets, subprocess
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
    """Yield RGB uint8 frames [H,W,3]. Returns (generator, effective_fps, (W,H)).

    Failure handling (mirrors :class:`Encoder`): the ffmpeg child is always reaped and its stdout closed. On a
    natural end of stream the generator waits for ffmpeg and raises ``subprocess.CalledProcessError`` for a
    nonzero exit. A trailing short read (fewer than one raw RGB frame) is reported instead of dropping bytes.
    If the generator is closed early, or the consumer/read raises, a still-running child is terminated and
    waited for; the primary exception (including ``GeneratorExit``) is preserved even if that cleanup fails.
    """
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
        primary = None
        natural_eof = False
        try:
            while True:
                b = p.stdout.read(n)
                if not b:
                    natural_eof = True
                    break
                if len(b) < n:
                    raise RuntimeError(
                        f"ffmpeg raw RGB stream ended with a partial frame: "
                        f"{len(b)} of {n} bytes for {str(path)!r}")
                yield np.frombuffer(b, np.uint8).reshape(H, W, 3)
        except BaseException as exc:                      # consumer throw/close, read error or partial frame
            primary = exc
            raise
        finally:
            failure = None
            try:
                p.stdout.close()
            except BaseException as exc:
                failure = exc
            if natural_eof:
                returncode = None
                try:
                    returncode = p.wait()
                except BaseException as exc:
                    if failure is None: failure = exc
                if returncode is not None and returncode != 0 and failure is None:
                    failure = subprocess.CalledProcessError(returncode, getattr(p, "args", None))
            else:
                running = True                            # only signal a child that has not already exited
                try:
                    running = p.poll() is None
                except BaseException as exc:
                    if failure is None: failure = exc
                if running:
                    try:
                        p.terminate()
                    except BaseException as exc:
                        if failure is None: failure = exc
                try:
                    p.wait()
                except BaseException as exc:
                    if failure is None: failure = exc
            if failure is not None and primary is None:
                raise failure
    return gen(), fps, (W, H)


def _unlink_quiet(path):
    """Best-effort unlink used on every Encoder failure path; a missing file is already clean."""
    if path is None:
        return
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _existing_output_mode(target):
    """Permission bits of an existing destination (following symlinks), or ``None`` if it does not exist.

    ``None`` also covers a target that cannot be stat'd (e.g. a permission error); the published file then keeps
    the mode the kernel derived from the process umask when the temp was created.
    """
    try:
        return os.stat(target).st_mode & 0o777
    except OSError:
        return None


def _create_output_tmp(target):
    """Create a unique hidden same-directory ``.mp4`` sibling; return an open ``(fd, path)``.

    The file is created with mode ``0o666`` so the kernel applies the process umask directly. That honors
    ``umask`` for a brand-new output without reading it: ``os.umask`` is process-global, so the usual
    read/restore pair mutates shared state and can race a concurrent thread creating a file. ``O_CREAT|O_EXCL``
    claims the random name atomically, so a collision is retried rather than reusing or following a path, and the
    ``.mp4`` suffix keeps ffmpeg's muxer inference intact. When the destination already exists its mode is copied
    onto the temp (``fchmod``), because publishing uses ``os.replace``, which swaps inodes and would otherwise
    make the temp's umask-derived mode win.
    """
    directory = os.path.dirname(os.path.abspath(target)) or os.curdir
    mode = _existing_output_mode(target)
    for _ in range(100):
        candidate = os.path.join(directory, f".mlxstab-{secrets.token_hex(8)}.mp4")
        try:
            fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        except FileExistsError:
            continue
        try:
            if mode is not None:
                os.fchmod(fd, mode)                              # preserve an existing destination's permissions
            return fd, candidate
        except BaseException:
            try: os.close(fd)
            except BaseException: pass
            _unlink_quiet(candidate)
            raise
    raise FileExistsError(f"could not create a unique Encoder temp beside {target!r}")


class Encoder:
    """Pipe raw frames into ffmpeg. gray=True writes a single-channel (mask) stream, lossless.

    ``out_fps`` converts to a constant output frame rate by resampling the raw input explicitly
    (``-vf fps=... -r ... -fps_mode cfr``). Relying on output ``-r`` alone duplicated/dropped frames
    unpredictably (60 fps -> 30 fps produced 32 frames for 1 s); the filter path is exact. Frame counts are
    validated by the caller (see ``tests/test_media_cadence.py``).

    Atomic output: ffmpeg writes a unique hidden sibling temporary (same directory, ``.mp4`` suffix so the
    muxer is still inferred). Only after the writer, stdin flush/close and the child all succeed with a zero exit
    status is the temp ``os.replace``d onto the requested path; the replacement is therefore atomic and
    same-filesystem. A new output gets the mode the process umask yields for a fresh ``0o666`` file; an existing
    destination's mode is copied onto the temp first so publishing preserves it. Any construction,
    encode/close/nonzero-exit or replace failure removes the partial and leaves the requested path unchanged.

    Symlinks: an existing symlink is untouched on every failure. On success ``os.replace`` swaps in the temp
    *inode*, so the symlink entry itself -- not its referent -- is replaced by a regular file. Because the
    destination mode is read with ``os.stat`` (following the link), that new file carries the referent's mode
    while the referent keeps its original bytes.

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
        self._target = str(path)                                 # requested output; published only on success
        self._tmp_path = None                                    # unique hidden sibling, .mp4 so ffmpeg picks the muxer
        self.p = None
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
        fd = None
        try:
            fd, tmp_name = _create_output_tmp(self._target)      # unique hidden sibling; umask-honoring mode
            os.close(fd); fd = None
            self._tmp_path = tmp_name
            self.p = subprocess.Popen(cmd + [tmp_name], stdin=subprocess.PIPE)
            self.q = queue.Queue(self.QUEUE_SIZE)                # writer thread: encoding overlaps with the next frame's GPU work
            self.t = threading.Thread(target=run, daemon=True); self.t.start()
        except BaseException:                                    # construction failure: reap the child, drop the partial
            if fd is not None:
                try: os.close(fd)
                except BaseException: pass
            if self.p is not None:
                try:
                    if self.p.stdin is not None:
                        try: self.p.stdin.close()
                        except BaseException: pass
                    self.p.kill()
                except BaseException: pass
                try: self.p.wait()
                except BaseException: pass
            self._discard_tmp()
            raise

    def _discard_tmp(self):
        """Remove the unpublished temporary; safe to call repeatedly."""
        path = getattr(self, "_tmp_path", None)
        self._tmp_path = None
        _unlink_quiet(path)

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
        always reaches it. Only a clean close publishes the temporary onto the requested path via ``os.replace``.
        Idempotent: a second call returns, or re-raises the same failure, without touching the stream again.
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
        if failure is None and returncode != 0:                  # success requires a real zero exit status
            failure = (subprocess.CalledProcessError(returncode, getattr(self.p, "args", None))
                       if isinstance(returncode, int)
                       else RuntimeError(f"ffmpeg exited without a status: {returncode!r}"))
        if failure is None:                                      # writer/flush/close/wait all clean: publish atomically
            try:
                os.replace(self._tmp_path, self._target)
                self._tmp_path = None
            except BaseException as exc:
                failure = exc
        if failure is not None:
            self._discard_tmp()                                  # never leave a partial hidden temp behind
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
    """Run a frame generator in a background thread so decoding overlaps with GPU/CPU work.

    Frames travel through a single bounded queue of ``n`` slots, in order: the producer reads ahead by at most
    ``n`` frames and a second, unbounded buffer is never created. A producer failure is stashed on the worker and
    re-raised by the consumer only after every frame queued before the failure has been yielded.

    Cancellation: when the consumer closes this generator early (or an exception propagates out of it), the
    producer is signalled so a blocked ``put`` can exit instead of wedging on a full queue. The upstream ``gen``
    is closed by that same producer thread -- never cross-thread -- even when cancellation lands right after a
    frame was returned, so ``media.decode`` can terminate and reap its ffmpeg child. A primary producer or
    consumer exception is never masked by a failure while closing ``gen``.

    A non-positive ``n`` is refused with ``ValueError``: Python's ``queue.Queue(0)`` (or any negative maxsize)
    is unbounded, which would silently defeat the bounded read-ahead guarantee. The signature and every valid
    ``n`` keep their normal behavior.
    """
    if n <= 0:
        raise ValueError(f"prefetch capacity must be positive, got {n!r}")
    return _prefetch(gen, n)


def _prefetch(gen, n):
    import queue, threading
    q = queue.Queue(n)
    cancel = threading.Event()
    end = object()
    result = {"error": None}                                  # producer's primary failure, read after join()

    def put(item):
        # Bounded, cancellation-aware put: a full queue whose consumer has left must not block the worker forever.
        while not cancel.is_set():
            try:
                q.put(item, timeout=0.05)                     # wakes on get() (not_full) or the cancel timeout
                return True
            except queue.Full:
                pass
        return False

    def run():
        primary = None
        src = None
        try:
            src = iter(gen)
            while not cancel.is_set():
                try:
                    x = next(src)
                except StopIteration:
                    break
                except BaseException as exc:                  # a real producer failure: keep it for the consumer
                    primary = exc
                    break
                if not put(x):
                    break
        except BaseException as exc:                          # iter() or an unexpected worker failure
            if primary is None: primary = exc
        finally:
            close = getattr(src, "close", None) if src is not None else None
            if close is not None:
                try:
                    close()                                   # producer thread owns src: decode can reap
                except BaseException as exc:
                    if primary is None: primary = exc         # never mask the primary producer failure
            result["error"] = primary
            put(end)                                          # ordered trailer; no-op once the consumer has left

    t = threading.Thread(target=run, daemon=True); t.start()
    try:
        while True:
            x = q.get()
            if x is end:
                break
            yield x
    finally:
        cancel.set()                                          # unblock a worker parked on a full queue
        t.join()                                              # the worker owns and closes gen; never leave it unjoined
    if result["error"] is not None:
        raise result["error"]                                 # only after every queued frame has been yielded
