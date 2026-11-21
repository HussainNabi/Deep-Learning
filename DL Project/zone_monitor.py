"""Person detection + tracking + restricted zone entry events (YOLO + ByteTrack).
Usage: python zone_monitor.py video.mp4 [--redraw] [--model yolo11n.pt] [--conf 0.3]
"""
import argparse, csv, json, os, time
import cv2
import numpy as np
from ultralytics import YOLO

EXIT_CONFIRM = 15  # consecutive frames outside before re-arm



# ---------- zone editor ----------
def _ccw(a, b, c):
    return (c[1]-a[1])*(b[0]-a[0]) - (b[1]-a[1])*(c[0]-a[0])

def _seg_intersect(a, b, c, d):
    return (_ccw(a, c, d) > 0) != (_ccw(b, c, d) > 0) and (_ccw(a, b, c) > 0) != (_ccw(a, b, d) > 0)

def validate(pts):
    if len(set(map(tuple, pts))) < 3:
        return "fewer than 3 distinct points"
    p = np.array(pts, dtype=np.float32)
    if cv2.contourArea(p) == 0:
        return "zero area"
    n = len(pts)
    for i in range(n):
        for j in range(i+1, n):
            if j == i+1 or (i == 0 and j == n-1):
                continue
            if _seg_intersect(pts[i], pts[(i+1) % n], pts[j], pts[(j+1) % n]):
                return "crossing lines"
    return None

def edit_zone(frame):
    pts, msg = [], ""
    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
    cv2.namedWindow("zone"); cv2.setMouseCallback("zone", on_mouse)
    while True:
        img = frame.copy()
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts)], True, (0, 0, 255), 2)
        for i, p in enumerate(pts):
            cv2.circle(img, p, 4, (0, 0, 255), -1)
            cv2.putText(img, f"P{i+1}", (p[0]+5, p[1]-5), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
        cv2.putText(img, "click=add  u=undo  Enter=save  q=quit  " + msg, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
        cv2.imshow("zone", img)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("u") and pts:
            pts.pop()
        elif k == 13:
            err = validate(pts)
            if err:
                msg = "INVALID: " + err
            else:
                break
        elif k == ord("q"):
            raise SystemExit
    cv2.destroyWindow("zone")
    return pts

def load_or_make_zone(video, frame, redraw):
    path = os.path.splitext(video)[0] + "_zone.json"
    h, w = frame.shape[:2]
    if os.path.exists(path) and not redraw:
        z = json.load(open(path))
        if z["frame_w"] == w and z["frame_h"] == h and z["video"] == os.path.basename(video):
            return z
    pts = edit_zone(frame)
    z = {"zone_name": "Zone A", "video": os.path.basename(video),
         "frame_w": w, "frame_h": h, "points": pts}
    json.dump(z, open(path, "w"), indent=2)
    return z


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--redraw", action="store_true")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    ok, first = cap.read()
    if not ok:
        raise SystemExit("cannot read video")
    zone = load_or_make_zone(a.video, first, a.redraw)
    poly = np.array(zone["points"], dtype=np.int32)

    run_id = "R" + time.strftime("%H%M%S")
    out = os.path.join(a.out, run_id); os.makedirs(out, exist_ok=True)
    f = open(os.path.join(out, "events.csv"), "w", newline="")
    wr = csv.writer(f)
    wr.writerow(["Run ID", "Event ID", "Video filename", "Elapsed video time", "Track ID",
                 "Zone label", "Event type", "Screenshot path"])

    # video info for report
    info = {"file": os.path.basename(a.video), "fps": fps,
            "w": first.shape[1], "h": first.shape[0],
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "model": a.model, "conf": a.conf}
    json.dump(info, open(os.path.join(out, "video_info.json"), "w"), indent=2)

    model = YOLO(a.model)
    state = {}  # tid -> {"inside": bool, "out_count": int}
    n_events, frame_idx, t0 = 0, 0, time.time()
    warn_until = -1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.track(frame, classes=[0], conf=a.conf, tracker="bytetrack.yaml",
                          persist=True, verbose=False)[0]
        events_now = []
        if res.boxes is not None and res.boxes.id is not None:
            for box, tid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.int().cpu().tolist()):
                x1, y1, x2, y2 = box
                pt = (float((x1+x2)/2), float(y2))  # bottom-center
                inside = cv2.pointPolygonTest(poly, pt, False) >= 0  # boundary counts as inside
                s = state.get(tid)
                etype = None
                if s is None:
                    state[tid] = {"inside": inside, "out_count": 0}
                    if inside:
                        etype = "initial_presence"
                else:
                    if inside:
                        if not s["inside"]:
                            etype = "entry"
                            s["inside"] = True
                        s["out_count"] = 0
                    else:
                        s["out_count"] += 1
                        if s["inside"] and s["out_count"] >= EXIT_CONFIRM:
                            s["inside"] = False  # confirmed exit -> re-armed
                if etype:
                    n_events += 1
                    eid = f"E{n_events:03d}"
                    shot = os.path.join(out, f"{eid}.jpg")
                    snap = frame.copy()
                    cv2.rectangle(snap, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.imwrite(shot, snap)
                    t = frame_idx / fps
                    wr.writerow([run_id, eid, info["file"], f"{int(t//60):02d}:{t%60:04.1f}",
                                 tid, zone["zone_name"], etype, shot])
                    f.flush()
                    warn_until = frame_idx + int(fps)
                col = (0, 0, 255) if state[tid]["inside"] else (0, 200, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
                cv2.putText(frame, f"ID {tid}", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, .6, col, 2)
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, col, -1)

        any_inside = any(s["inside"] for s in state.values())
        cv2.polylines(frame, [poly], True, (0, 0, 255), 2)
        if any_inside or frame_idx <= warn_until:
            cv2.putText(frame, "WARNING: RESTRICTED AREA", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 0, 255), 3)
        cv2.imshow("monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_idx += 1

    el = time.time() - t0
    json.dump({"frames": frame_idx, "seconds": el, "avg_fps": frame_idx/el if el else 0,
               "events": n_events}, open(os.path.join(out, "run_stats.json"), "w"), indent=2)
    f.close(); cap.release(); cv2.destroyAllWindows()
    print(f"{run_id}: {n_events} events, {frame_idx/el:.1f} FPS -> {out}")

if __name__ == "__main__":
    main()
