import argparse
import time
import threading
import sys
import os
import platform
import signal
import faulthandler

import cv2
import numpy as np

# Alarm: generate a short sine-wave beep and play it asynchronously.
# Uses simpleaudio if available; otherwise falls back to a console message.
try:
    import simpleaudio as sa
    SIMPLEAUDIO_OK = True
except Exception:
    SIMPLEAUDIO_OK = False

def play_beep(duration_s: float = 0.6, freq_hz: int = 880, sample_rate: int = 44100, volume: float = 0.4):
    if not SIMPLEAUDIO_OK:
        print("[ALARM] Person detected (simpleaudio not installed).")
        return
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), False, dtype=np.float32)
    wave = (np.sin(2 * np.pi * freq_hz * t) * volume).astype(np.float32)
    # Fade in/out to avoid clicks
    fade = min(0.03, duration_s / 4)
    n_fade = int(sample_rate * fade)
    if n_fade > 0:
        fade_win = np.linspace(0, 1, n_fade, dtype=np.float32)
        wave[:n_fade] *= fade_win
        wave[-n_fade:] *= fade_win[::-1]
    # Convert to 16-bit PCM
    audio = (wave * 32767).astype(np.int16)
    try:
        # Fire-and-forget to avoid CoreAudio/simpleaudio wait_done issues on macOS
        sa.play_buffer(audio, 1, 2, sample_rate)
    except Exception:
        print("[ALARM] Person detected (audio playback error)")

def play_system_sound(volume: float = 0.6):
    """Play a short macOS system sound via afplay to avoid Python audio stack issues."""
    try:
        import subprocess
        sound = "/System/Library/Sounds/Glass.aiff"
        subprocess.Popen(["afplay", "-v", str(volume), sound], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        print("[ALARM] Beep (system)")

def play_alarm(args):
    backend = args.alarm_backend
    if backend == "auto":
        # Prefer system sound on macOS to avoid simpleaudio/CoreAudio instability
        if platform.system() == "Darwin":
            backend = "system"
        else:
            backend = "simpleaudio" if SIMPLEAUDIO_OK else "system"
    try:
        if getattr(args, 'debug', False):
            print(f"[DEBUG] Alarm backend selected: {backend} (simpleaudio_ok={SIMPLEAUDIO_OK})")
        if backend == "simpleaudio" and SIMPLEAUDIO_OK:
            try:
                play_beep()
            except Exception as e:
                if getattr(args, 'debug', False):
                    print(f"[DEBUG] simpleaudio failed ({e}); falling back to system sound")
                play_system_sound()
        else:
            play_system_sound()
    except Exception:
        print("[ALARM] Person detected")

def parse_args():
    ap = argparse.ArgumentParser(description="RTSP YOLO person alarm")
    ap.add_argument("--rtsp", required=True, help="RTSP URL, e.g. rtsp://user:pass@host:554/stream")
    ap.add_argument("--model", default="yolov10n.pt", help="Ultralytics model path or name")
    ap.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    ap.add_argument("--cooldown", type=float, default=30.0, help="Min seconds between alarms")
    ap.add_argument("--noshow", action="store_true", help="Run headless (no OpenCV window)")
    ap.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"),
                    help="Resize frames before inference (width height)")
    ap.add_argument("--device", default=None, help="Inference device, e.g. '0' (GPU 0), 'cpu'")
    ap.add_argument("--debug", action="store_true", help="Enable verbose diagnostic logs")
    ap.add_argument("--engine", choices=["auto", "onnx"], default="auto",
                    help="Inference engine: 'auto' (PyTorch) or 'onnx' (onnxruntime via Ultralytics)")
    ap.add_argument("--rtsp-transport", choices=["auto", "tcp", "udp"], default="auto",
                    help="Preferred RTSP transport for both GStreamer and FFmpeg backends")
    ap.add_argument("--gst-latency", type=int, default=200, help="GStreamer rtspsrc latency in ms")
    ap.add_argument("--reconnect-delay", type=float, default=1.0, help="Seconds to wait before reconnecting")
    # FFmpeg backend tuning via environment variables OpenCV recognizes
    ap.add_argument("--ffmpeg-read-attempts", type=int, default=None,
                    help="Set OPENCV_FFMPEG_READ_ATTEMPTS (e.g. 16384)")
    ap.add_argument("--ffmpeg-capture-options", default=None,
                    help="Set OPENCV_FFMPEG_CAPTURE_OPTIONS (e.g. 'rtsp_transport;tcp|stimeout;20000000|rw_timeout;20000000')")
    # Torch stability & performance tuning
    ap.add_argument("--torch-threads", type=int, default=None, help="torch.set_num_threads")
    ap.add_argument("--torch-interop-threads", type=int, default=None, help="torch.set_num_interop_threads")
    ap.add_argument("--no-mkldnn", action="store_true", help="Disable MKLDNN (oneDNN) backend in Torch")
    ap.add_argument("--torch-safe", action="store_true", help="Safe mode: disable MKLDNN and force single-threaded Torch/BLAS")
    # Throughput vs latency control
    ap.add_argument("--every-n", type=int, default=3, help="Run inference every N frames (skip N-1 frames)")
    # Smoothing / debouncing
    ap.add_argument("--presence-on", type=int, default=2, help="Consecutive inference frames with a person required to switch presence ON")
    ap.add_argument("--presence-off", type=int, default=4, help="Consecutive inference frames without a person required to switch presence OFF")
    ap.add_argument("--box-persist", type=int, default=6, help="Keep drawing last person boxes for N frames after they vanish")
    ap.add_argument("--no-alarm", action="store_true", help="Disable audio alarm (diagnose crashes or run silently)")
    ap.add_argument("--alarm-backend", choices=["auto", "simpleaudio", "system"], default="auto",
                    help="Alarm backend: 'simpleaudio' (Python) or 'system' (afplay on macOS); auto picks best for OS")
    return ap.parse_args()

def main():
    args = parse_args()

    # Enable low-level Python crash diagnostics (helps when native libs segfault)
    try:
        faulthandler.enable(all_threads=True)
        try:
            faulthandler.register(signal.SIGSEGV, all_threads=True, chain=True)
        except Exception:
            pass
    except Exception:
        pass

    def debug(msg: str):
        if args.debug:
            print(f"[DEBUG] {msg}")

    # Configure Torch-related environment BEFORE importing Ultralytics/Torch
    if args.torch_safe:
        # Minimize threading to avoid unstable SIMD paths in some builds
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[var] = os.environ.get(var, "1")
        # Also request ffmpeg to be conservative on threads
        os.environ.setdefault("FFMPEG_THREAD", "1")
    if args.torch_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(max(1, args.torch_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
        os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
        os.environ.setdefault("NUMEXPR_NUM_THREADS", os.environ["OMP_NUM_THREADS"])

    # Lazy import to make start-up errors clearer
    try:
        from ultralytics import YOLO
        import ultralytics as _ultra
    except Exception as e:
        print("ERROR: Could not import Ultralytics. Install with `pip install ultralytics`.")
        raise

    # Environment quick summary
    if args.debug:
        print(f"[DEBUG] Python: {platform.python_version()} ({platform.python_implementation()}) on {platform.system()} {platform.release()}")
        print(f"[DEBUG] OpenCV: {cv2.__version__}")
        try:
            print(f"[DEBUG] Ultralytics: {getattr(_ultra, '__version__', 'unknown')}")
        except Exception:
            pass
        # Increase OpenCV internal logging
        try:
            import cv2 as _cv
            if hasattr(_cv, 'utils') and hasattr(_cv.utils, 'logging'):
                _cv.utils.logging.setLogLevel(_cv.utils.logging.LOG_LEVEL_DEBUG)
                print("[DEBUG] OpenCV log level set to DEBUG")
        except Exception:
            pass
        # Summarize key Video I/O build flags from OpenCV
        try:
            info = cv2.getBuildInformation()
            lines = []
            for line in info.splitlines():
                if any(k in line for k in ("Video I/O:", "GStreamer", "FFmpeg", "AVFoundation", "QTKit", "Media I/O")):
                    lines.append(line.strip())
            if lines:
                print("[DEBUG] OpenCV Video I/O build summary:")
                for ln in lines:
                    print(f"[DEBUG]   {ln}")
        except Exception:
            pass
        # List available VideoCapture backends
        try:
            if hasattr(cv2, 'videoio_registry'):
                ids = cv2.videoio_registry.getBackends()
                names = []
                for bid in ids:
                    try:
                        names.append(cv2.videoio_registry.getBackendName(bid))
                    except Exception:
                        names.append(str(bid))
                print(f"[DEBUG] Available VideoCapture backends: {', '.join(names)}")
        except Exception:
            pass

    # Torch diagnostics and runtime tuning
    th = None
    try:
        import torch as th
        if args.debug:
            print(f"[DEBUG] Torch: {th.__version__}")
            try:
                print(f"[DEBUG] Torch num threads: {th.get_num_threads()}")
            except Exception:
                pass
            try:
                print(f"[DEBUG] Torch interop threads: {th.get_num_interop_threads()}")
            except Exception:
                pass
            try:
                mkldnn_avail = getattr(th.backends, 'mkldnn', None)
                if mkldnn_avail is not None:
                    print(f"[DEBUG] MKLDNN available={mkldnn_avail.is_available()} enabled={mkldnn_avail.enabled}")
            except Exception:
                pass
            try:
                xnn = getattr(th.backends, 'xnnpack', None)
                if xnn is not None:
                    print(f"[DEBUG] XNNPACK enabled={xnn.enabled}")
            except Exception:
                pass
        # Apply user-requested runtime settings
        if args.no_mkldnn or args.torch_safe:
            try:
                th.backends.mkldnn.enabled = False
                if args.debug:
                    print("[DEBUG] Disabled Torch MKLDNN backend")
            except Exception:
                pass
        if args.torch_safe:
            try:
                th.backends.xnnpack.enabled = False
                if args.debug:
                    print("[DEBUG] Disabled Torch XNNPACK backend")
            except Exception:
                pass
        if args.torch_threads is not None or args.torch_safe:
            try:
                th.set_num_threads(max(1, args.torch_threads or 1))
                if args.debug:
                    print(f"[DEBUG] Torch num_threads set to {th.get_num_threads()}")
            except Exception:
                pass
        if args.torch_interop_threads is not None or args.torch_safe:
            try:
                th.set_num_interop_threads(max(1, args.torch_interop_threads or 1))
                if args.debug:
                    print(f"[DEBUG] Torch interop threads set to {th.get_num_interop_threads()}")
            except Exception:
                pass
    except Exception:
        th = None

    # Model load with optional ONNX engine
    if args.engine == 'onnx':
        print(f"[INFO] Loading model: {args.model} (engine: onnx)")
        onnx_path = args.model
        if not onnx_path.lower().endswith('.onnx'):
            onnx_path = os.path.splitext(args.model)[0] + '.onnx'
        if not os.path.exists(onnx_path):
            print(f"[INFO] Exporting to ONNX: {onnx_path}")
            try:
                _tmp_model = YOLO(args.model)
                _tmp_model.export(format='onnx', dynamic=True, opset=12)
            except Exception as e:
                print(f"[ERROR] ONNX export failed: {e}")
                sys.exit(1)
            if not os.path.exists(onnx_path):
                # Ultralytics may emit to a model-specific path; attempt to find it
                base = os.path.splitext(os.path.basename(args.model))[0]
                cand = os.path.join(os.getcwd(), f"{base}.onnx")
                if os.path.exists(cand):
                    onnx_path = cand
        model = YOLO(onnx_path)
    else:
        print(f"[INFO] Loading model: {args.model}")
        model = YOLO(args.model)
        if args.device is not None:
            try:
                model.to(args.device)
                print(f"[INFO] Using device: {args.device}")
            except Exception as e:
                print(f"[WARN] Could not set device '{args.device}': {e}. Falling back to default.")

    print(f"[INFO] Opening RTSP stream: {args.rtsp}")
    # Apply FFmpeg backend tuning if requested (affects fallback and FFmpeg use)
    if args.ffmpeg_read_attempts is not None:
        os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = str(args.ffmpeg_read_attempts)
        debug(f"Set OPENCV_FFMPEG_READ_ATTEMPTS={os.environ['OPENCV_FFMPEG_READ_ATTEMPTS']}")
    if args.ffmpeg_capture_options:
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = args.ffmpeg_capture_options
        debug(f"Set OPENCV_FFMPEG_CAPTURE_OPTIONS={os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS']}")

    # Prefer GStreamer appsink (no GTK UI needed). For rtsps, use TCP protocols.
    gst_protocol = 'tcp' if args.rtsp_transport in ('auto', 'tcp') else 'udp'
    gst = (
        f'rtspsrc location="{args.rtsp}" protocols={gst_protocol} latency={args.gst_latency} '
        f'! decodebin ! videoconvert ! appsink drop=true sync=false'
    )
    debug(f"Trying GStreamer pipeline: {gst}")
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)

    backend_name = None
    if hasattr(cap, 'getBackendName'):
        try:
            backend_name = cap.getBackendName()
        except Exception:
            backend_name = None

    if not cap.isOpened():
        print("[WARN] GStreamer open failed, falling back to OpenCV/FFmpeg…")
        cap.release()
        # For FFmpeg backend, optionally guide transport and options
        # Use OPENCV_FFMPEG_CAPTURE_OPTIONS if provided; otherwise, append rtsp_transport via options string on rtsps/rtsp
        if args.ffmpeg_capture_options is None and args.rtsp_transport in ('tcp', 'udp'):
            # Compose a minimal options set prioritizing transport
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = f"rtsp_transport;{args.rtsp_transport}"
            debug(f"Defaulted OPENCV_FFMPEG_CAPTURE_OPTIONS={os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS']}")
        cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)
        backend_name = None
        if hasattr(cap, 'getBackendName'):
            try:
                backend_name = cap.getBackendName()
            except Exception:
                backend_name = None

    # Report capture backend and properties
    if cap.isOpened():
        if backend_name:
            print(f"[INFO] VideoCapture backend: {backend_name}")
        try:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
            fourcc_str = ''.join([chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)]) if fourcc else 'n/a'
            print(f"[INFO] Stream properties: {w}x{h} @ {fps:.2f} FPS, FOURCC={fourcc_str}")
        except Exception as e:
            debug(f"Couldn't read capture properties: {e}")
        # Try to minimize internal buffering to reduce latency
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            debug(f"Set CAP_PROP_BUFFERSIZE to 1")
        except Exception:
            pass


    if not cap.isOpened():
        print("[ERROR] Failed to open RTSP stream. Check URL/credentials/network.")
        sys.exit(1)

    last_alarm_time = 0.0
    person_class_ids = set()

    # Try to learn which class index corresponds to 'person'
    # Ultralytics models typically map 0 -> 'person' for COCO, but we'll check.
    names = None
    try:
        names = model.model.names if hasattr(model, "model") else model.names
    except Exception:
        names = None

    if isinstance(names, dict):
        person_class_ids = {i for i, n in names.items() if str(n).lower() == "person"}
    elif isinstance(names, list):
        person_class_ids = {i for i, n in enumerate(names) if str(n).lower() == "person"}
    else:
        # Fallback assumption for COCO models
        person_class_ids = {0}

    print(f"[INFO] Person class id(s): {sorted(person_class_ids)}")
    print(f"[INFO] Confidence threshold: {args.conf}")
    print(f"[INFO] Alarm cooldown: {args.cooldown:.1f}s")
    if not SIMPLEAUDIO_OK:
        print("[WARN] simpleaudio not available. Install with `pip install simpleaudio` for audible alarms.")

    # Main loop
    first_frame_logged = False
    first_infer_logged = False
    # Validate and report stride
    stride_n = max(1, int(args.every_n))
    if stride_n != args.every_n:
        print(f"[WARN] Adjusted --every-n to {stride_n} (must be >=1)")
    if stride_n > 1:
        print(f"[INFO] Processing every {stride_n} frames (skipping {stride_n-1} frames between inferences)")
    # Presence hysteresis state
    presence = False
    on_streak = 0
    off_streak = 0
    # Box persistence state (store original-scale boxes)
    last_xyxy = None
    last_confs = None
    last_clss = None
    persist_left = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            # Attempt a lightweight reconnect
            print("[WARN] Empty frame. Reconnecting...")
            cap.release()
            time.sleep(max(0.0, args.reconnect_delay))
            # Try GStreamer again first, then FFmpeg
            debug("Reopening with GStreamer…")
            # Rebuild GStreamer pipeline in case options changed
            gst_protocol = 'tcp' if args.rtsp_transport in ('auto', 'tcp') else 'udp'
            gst = (
                f'rtspsrc location="{args.rtsp}" protocols={gst_protocol} latency={args.gst_latency} '
                f'! decodebin ! videoconvert ! appsink drop=true sync=false'
            )
            debug(f"Trying GStreamer pipeline: {gst}")
            cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                print("[WARN] Reopen via GStreamer failed. Trying FFmpeg…")
                cap.release()
                # Ensure FFmpeg options are set consistently on reconnect
                if args.ffmpeg_capture_options is None and args.rtsp_transport in ('tcp', 'udp'):
                    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = f"rtsp_transport;{args.rtsp_transport}"
                    debug(f"Defaulted OPENCV_FFMPEG_CAPTURE_OPTIONS={os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS']}")
                cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)
            else:
                debug("Reconnected via GStreamer")
            continue

        original_h, original_w = frame.shape[:2]

        if not first_frame_logged:
            print(f"[INFO] First frame: shape={frame.shape}, dtype={frame.dtype}")
            first_frame_logged = True

        frame_idx += 1
        do_infer = (frame_idx % stride_n == 1)

        # Optional resize (can improve throughput on weak hardware)
        infer_frame = frame
        if do_infer:
            if args.resize:
                infer_frame = cv2.resize(frame, tuple(args.resize), interpolation=cv2.INTER_AREA)

        # Run inference only on scheduled frames
        results = None
        if do_infer:
            infer_input = np.ascontiguousarray(infer_frame)
            t0 = time.perf_counter()
            # Be explicit about dtype and device; keep half=False on CPU. For ONNX engine, device is ignored.
            predict_kwargs = dict(source=infer_input, conf=args.conf, verbose=False, half=False)
            if args.engine != 'onnx':
                predict_kwargs['device'] = (args.device or 'cpu')
            results = model.predict(**predict_kwargs)[0]
            infer_dt = time.perf_counter() - t0
            if args.debug and not first_infer_logged:
                try:
                    num = len(results.boxes) if results and results.boxes is not None else 0
                except Exception:
                    num = 0
                print(f"[DEBUG] First inference took {infer_dt*1000:.1f} ms, detections={num}")
                first_infer_logged = True

        # Determine person presence (based on inference frames only)
        saw_person = False
        cur_xyxy = None
        cur_confs = None
        cur_clss = None
        if results is not None and results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes
            cur_xyxy = boxes.xyxy.cpu().numpy()
            cur_confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros((cur_xyxy.shape[0],))
            cur_clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros((cur_xyxy.shape[0],), dtype=int)

            # If we resized for inference, scale boxes back to original frame for drawing/persisting
            if args.resize:
                sx = original_w / float(infer_frame.shape[1])
                sy = original_h / float(infer_frame.shape[0])
                cur_xyxy[:, [0, 2]] *= sx
                cur_xyxy[:, [1, 3]] *= sy

            # Check if any box is a person
            if cur_clss.size > 0:
                saw_person = bool(np.any(np.isin(cur_clss, list(person_class_ids))))

        # Update hysteresis on inference frames
        rising_edge = False
        if do_infer:
            if saw_person:
                on_streak += 1
                off_streak = 0
            else:
                off_streak += 1
                on_streak = 0

            if not presence and on_streak >= max(1, args.presence_on):
                presence = True
                rising_edge = True
            elif presence and off_streak >= max(1, args.presence_off):
                presence = False

        # Update box persistence
        if cur_xyxy is not None and saw_person:
            last_xyxy = cur_xyxy
            last_confs = cur_confs
            last_clss = cur_clss
            persist_left = max(persist_left, int(args.box_persist))
        else:
            if persist_left > 0:
                persist_left -= 1

        # Choose boxes to draw: current if available, else persisted
        draw_xyxy = cur_xyxy if cur_xyxy is not None else (last_xyxy if persist_left > 0 else None)
        draw_confs = cur_confs if cur_confs is not None else (last_confs if persist_left > 0 else None)
        draw_clss = cur_clss if cur_clss is not None else (last_clss if persist_left > 0 else None)

        # Draw detections (current or persisted)
        if draw_xyxy is not None and draw_xyxy.size > 0:
            for (x1, y1, x2, y2), c, cls in zip(draw_xyxy, draw_confs, draw_clss):
                label = names.get(cls, str(cls)) if isinstance(names, dict) else (names[cls] if isinstance(names, list) and cls < len(names) else str(cls))
                color = (0, 255, 0) if cls in person_class_ids else (255, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                txt = f"{label} {c:.2f}"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (int(x1), int(y1) - th - 6), (int(x1) + tw + 4, int(y1)), color, -1)
                cv2.putText(frame, txt, (int(x1) + 2, int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Alarm only on rising edge of presence and subject to cooldown
        trigger_alarm = rising_edge

        # Alarm rate-limited
        now = time.time()
        if trigger_alarm and (now - last_alarm_time) >= args.cooldown:
            last_alarm_time = now
            if not args.no_alarm:
                threading.Thread(target=play_alarm, args=(args,), daemon=True).start()
            else:
                if args.debug:
                    print("[DEBUG] Alarm suppressed by --no-alarm (person detected)")


        # Show window (unless headless)
        if not args.noshow:
            try:
                # If we skipped inference, just display the raw frame quickly.
                cv2.imshow("RTSP YOLO Person Alarm (press q to quit)", frame)
                # ~30 FPS display pacing
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(f"[WARN] Display failed: {e}. Consider running with --noshow.")
                args.noshow = True

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
