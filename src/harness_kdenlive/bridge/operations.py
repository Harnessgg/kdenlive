import json
import os
import shutil
import subprocess
import platform
import tempfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from lxml import etree

from harness_kdenlive import __version__
from harness_kdenlive.api.timeline import TimelineAPI
from harness_kdenlive.core.diff_engine import DiffEngine
from harness_kdenlive.core.transaction import TransactionManager
from harness_kdenlive.core.validator import ProjectValidator
from harness_kdenlive.core.xml_engine import KdenliveProject


class BridgeOperationError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


ACTION_METHODS = [
    "system.health",
    "system.version",
    "system.actions",
    "system.doctor",
    "system.soak",
    "project.create",
    "project.clone",
    "project.plan_edit",
    "project.undo",
    "project.redo",
    "project.autosave",
    "project.pack",
    "project.recalculate_timeline_bounds",
    "project.inspect",
    "project.validate",
    "project.diff",
    "project.snapshot",
    "asset.import",
    "asset.create_text",
    "asset.update_text",
    "asset.metadata",
    "asset.replace",
    "bin.list",
    "bin.create_folder",
    "bin.move_asset",
    "effect.list",
    "effect.apply",
    "effect.update",
    "effect.remove",
    "effect.keyframes",
    "transition.list",
    "transition.apply",
    "transition.remove",
    "transition.wipe",
    "timeline.add_clip",
    "timeline.move_clip",
    "timeline.trim_clip",
    "timeline.remove_clip",
    "timeline.split_clip",
    "timeline.ripple_delete",
    "timeline.insert_gap",
    "timeline.remove_all_gaps",
    "timeline.stitch_clips",
    "timeline.list_clips",
    "timeline.select_zone",
    "timeline.detect_gaps",
    "clip.resolve",
    "timeline.time_remap",
    "timeline.transform",
    "timeline.nudge_clip",
    "timeline.slip_clip",
    "timeline.slide_clip",
    "timeline.ripple_insert",
    "timeline.group_clips",
    "timeline.ungroup_clips",
    "sequence.list",
    "sequence.copy",
    "sequence.set_active",
    "audio.add_music",
    "audio.duck",
    "audio.fade",
    "audio.normalize",
    "audio.remove_silence",
    "audio.pan",
    "color.grade",
    "track.add",
    "track.remove",
    "track.reorder",
    "track.resolve",
    "track.mute",
    "track.unmute",
    "track.lock",
    "track.unlock",
    "track.show",
    "track.hide",
    "render.clip",
    "render.project",
    "render.status",
    "render.latest",
    "render.retry",
    "render.cancel",
    "render.list_jobs",
    "render.wait",
    "export.edl",
    "export.xml",
    "export.otio",
    "batch.execute",
]

RENDER_JOBS: Dict[str, Dict[str, Any]] = {}
AUTOSAVE_STATE: Dict[str, Dict[str, Any]] = {}
MLT_PRODUCERS_CACHE: Optional[set[str]] = None


def _load(path: str) -> KdenliveProject:
    try:
        return KdenliveProject(path)
    except FileNotFoundError as exc:
        raise BridgeOperationError("NOT_FOUND", str(exc)) from exc


def _save(project: KdenliveProject, output: Optional[str]) -> str:
    target = Path(output) if output else None
    return str(project.save(target))


def _resolve_bin(name: str) -> Path:
    env_key = f"HARNESS_KDENLIVE_{name.upper()}_PATH"
    from_env = os.getenv(env_key)
    if from_env:
        p = Path(from_env)
        if p.exists():
            return p
    default_root = Path(r"C:\Program Files\kdenlive\bin")
    default = default_root / f"{name}.exe"
    if default.exists():
        return default
    raise BridgeOperationError("NOT_FOUND", f"Kdenlive binary not found: {name}. Set {env_key}.")


def _project_fps(project: KdenliveProject) -> float:
    profile = project.root.find(".//profile")
    if profile is None:
        return 30.0
    num = int(profile.get("frame_rate_num", "30"))
    den = int(profile.get("frame_rate_den", "1"))
    if den == 0:
        return 30.0
    return num / den


def _validate_project_for_edit(project: KdenliveProject) -> None:
    issues = ProjectValidator(project).validate_all(check_files=False)
    errors = [e for e in issues if e.severity == "error"]
    if errors:
        raise BridgeOperationError("VALIDATION_FAILED", errors[0].message)


def _next_producer_id(project: KdenliveProject, prefix: str = "producer") -> str:
    existing = {p.id for p in project.get_producers()}
    idx = 1
    while f"{prefix}{idx}" in existing:
        idx += 1
    return f"{prefix}{idx}"


def _next_playlist_id(project: KdenliveProject) -> str:
    existing = {p.get("id", "") for p in project.get_playlists()}
    idx = 0
    while f"playlist{idx}" in existing:
        idx += 1
    return f"playlist{idx}"


def _next_kdenlive_clip_id(project: KdenliveProject) -> str:
    max_id = 0
    for prop in project.root.findall('.//property[@name="kdenlive:id"]'):
        if prop.text and prop.text.isdigit():
            max_id = max(max_id, int(prop.text))
    return str(max_id + 1)


def _producer_duration_frames(project: KdenliveProject, producer_id: str) -> int:
    matches = project.get_producers(id_filter=producer_id)
    if not matches:
        raise BridgeOperationError("INVALID_INPUT", f"Producer '{producer_id}' not found")
    producer = matches[0]
    in_point = int(producer.in_point or "0")
    if producer.out_point is not None:
        return max(1, int(producer.out_point) - in_point + 1)
    return 1


def _get_project_tractor(project: KdenliveProject) -> etree._Element:
    for tractor in project.root.findall(".//tractor"):
        marker = tractor.find('.//property[@name="kdenlive:projectTractor"]')
        if marker is not None and marker.text == "1":
            return tractor
    raise BridgeOperationError("INVALID_INPUT", "Project tractor not found")


def _probe_media_duration_seconds(path: Path) -> Optional[float]:
    try:
        ffprobe = _resolve_bin("ffprobe")
    except BridgeOperationError:
        return None
    probe = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        return None
    try:
        return float(probe.stdout.strip())
    except ValueError:
        return None


def _append_main_bin_entry(project: KdenliveProject, producer_id: str, out_frame: int) -> None:
    main_bin = project.get_main_bin()
    if main_bin is None:
        raise BridgeOperationError("INVALID_INPUT", "Project is missing main_bin playlist")
    entry = etree.SubElement(
        main_bin,
        "entry",
        producer=producer_id,
        **{"in": "0", "out": str(max(0, out_frame))},
    )
    entry.text = None


def _render_and_probe_duration(cmd: List[str], output: Path, cwd: Optional[Path] = None) -> float:
    process = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd is not None else None,
    )
    if process.returncode != 0:
        message = (process.stderr or process.stdout or "render failed").strip()
        raise BridgeOperationError("ERROR", message)
    if not output.exists():
        raise BridgeOperationError("ERROR", f"Render failed, output missing: {output}")
    rendered_duration = _probe_media_duration_seconds(output)
    if rendered_duration is None:
        raise BridgeOperationError("ERROR", f"Render created invalid output: {output}")
    return rendered_duration


def _mutation_payload(
    data: Dict[str, Any],
    changed: bool = True,
    idempotent: bool = False,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    merged = dict(data)
    merged["changed"] = changed
    merged["idempotent"] = idempotent
    merged["warnings"] = warnings or []
    return merged


def _track_rows(project: KdenliveProject) -> List[Dict[str, Any]]:
    tractor = _get_project_tractor(project)
    rows: List[Dict[str, Any]] = []
    for index, track in enumerate(tractor.findall("track")):
        track_id = track.get("producer", "")
        playlist = project.root.find(f'.//playlist[@id="{track_id}"]')
        name = track_id
        muted = False
        locked = False
        hidden = False
        if playlist is not None:
            name_prop = playlist.find('./property[@name="kdenlive:track_name"]')
            name = (name_prop.text or track_id) if name_prop is not None else track_id
            muted = ((playlist.findtext('./property[@name="harness:muted"]') or "0") == "1")
            locked = ((playlist.findtext('./property[@name="harness:locked"]') or "0") == "1")
            hidden = ((playlist.findtext('./property[@name="harness:hidden"]') or "0") == "1")
        rows.append(
            {
                "index": index,
                "trackId": track_id,
                "name": name,
                "type": "audio" if track.get("hide") == "video" else "video",
                "muted": muted,
                "locked": locked,
                "hidden": hidden,
            }
        )
    return rows


def _apply_render_preset(params: Dict[str, Any]) -> Dict[str, Any]:
    preset = str(params.get("preset_name", "")).lower().strip()
    merged = dict(params)
    if preset in {"h264", ""}:
        merged.setdefault("vcodec", "libx264")
        merged.setdefault("acodec", "aac")
    elif preset == "hevc":
        merged.setdefault("vcodec", "libx265")
        merged.setdefault("acodec", "aac")
        merged.setdefault("preset", "medium")
    elif preset == "prores":
        merged.setdefault("vcodec", "prores_ks")
        merged.setdefault("acodec", "pcm_s16le")
        merged.setdefault("crf", "10")
    else:
        raise BridgeOperationError("INVALID_INPUT", f"Unknown preset_name: {preset}")
    return merged


def _resolve_clip_element(project: KdenliveProject, clip_ref: str) -> Tuple[etree._Element, str]:
    timeline = TimelineAPI(project)
    clip = timeline._resolve_clip(clip_ref)
    if clip is None or clip.element is None:
        raise BridgeOperationError("INVALID_INPUT", f"Clip not found: {clip_ref}")
    return clip.element, clip.track_id


def _ensure_effect_id(entry: etree._Element, effect_id: Optional[str] = None) -> str:
    return effect_id or f"effect_{uuid4().hex[:12]}"


def _get_or_create_property(parent: etree._Element, name: str) -> etree._Element:
    prop = parent.find(f'./property[@name="{name}"]')
    if prop is None:
        prop = etree.SubElement(parent, "property", name=name)
        prop.text = ""
    return prop


def _load_bin_folders(project: KdenliveProject) -> Dict[str, Dict[str, Any]]:
    prop = project.root.find('.//property[@name="harness:bin-folders"]')
    if prop is None or not prop.text:
        return {"root": {"id": -1, "name": "root", "parentId": None}}
    try:
        data = json.loads(prop.text)
    except Exception:
        return {"root": {"id": -1, "name": "root", "parentId": None}}
    if "root" not in data:
        data["root"] = {"id": -1, "name": "root", "parentId": None}
    return data


def _save_bin_folders(project: KdenliveProject, folders: Dict[str, Dict[str, Any]]) -> None:
    prop = _get_or_create_property(project.root, "harness:bin-folders")
    prop.text = json.dumps(folders, separators=(",", ":"))


def _next_folder_id(folders: Dict[str, Dict[str, Any]]) -> int:
    existing = [int(v.get("id", -1)) for v in folders.values()]
    candidate = 1
    while candidate in existing:
        candidate += 1
    return candidate


def _producer_element(project: KdenliveProject, producer_id: str) -> etree._Element:
    producer = project.root.find(f'.//producer[@id="{producer_id}"]')
    if producer is None:
        raise BridgeOperationError("INVALID_INPUT", f"Producer '{producer_id}' not found")
    return producer


def _producer_resource_metadata(project: KdenliveProject, producer_id: str) -> Dict[str, Any]:
    producer = _producer_element(project, producer_id)
    resource_prop = producer.find('./property[@name="resource"]')
    resource = resource_prop.text if resource_prop is not None else None
    in_point = int(producer.get("in", "0"))
    out_raw = producer.get("out")
    out_point = int(out_raw) if out_raw is not None else in_point
    duration_frames = max(1, out_point - in_point + 1)
    meta: Dict[str, Any] = {
        "producerId": producer_id,
        "resource": resource,
        "in": in_point,
        "out": out_point,
        "durationFrames": duration_frames,
    }
    if not resource or resource.startswith("#") or resource == "black":
        return meta
    path = Path(resource)
    if not path.is_absolute():
        path = project.project_path.parent / path
    if not path.exists():
        meta["missing"] = True
        return meta
    meta["path"] = str(path)
    meta["sizeBytes"] = path.stat().st_size
    duration = _probe_media_duration_seconds(path)
    if duration is not None:
        meta["durationSeconds"] = duration
    try:
        ffprobe = _resolve_bin("ffprobe")
        probe = subprocess.run(
            [
                str(ffprobe),
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,width,height,r_frame_rate,sample_rate,channels",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if probe.returncode == 0 and probe.stdout:
            raw = json.loads(probe.stdout)
            meta["streams"] = raw.get("streams", [])
    except Exception:
        pass
    return meta


def _clip_context(project: KdenliveProject, clip_ref: str) -> Tuple[etree._Element, etree._Element, int]:
    timeline = TimelineAPI(project)
    clip = timeline._resolve_clip(clip_ref)
    if clip is None or clip.element is None:
        raise BridgeOperationError("INVALID_INPUT", f"Clip not found: {clip_ref}")
    playlist = project.root.find(f'.//playlist[@id="{clip.track_id}"]')
    if playlist is None:
        raise BridgeOperationError("INVALID_INPUT", f"Track not found: {clip.track_id}")
    idx = list(playlist).index(clip.element)
    return clip.element, playlist, idx


def _timeline_max_end(project: KdenliveProject) -> int:
    clips = project.get_clips_on_timeline()
    return max([c.timeline_end for c in clips], default=0)


def _clip_rows(project: KdenliveProject, track_id: Optional[str] = None) -> List[Dict[str, Any]]:
    clips = project.get_clips_on_timeline(track_id=track_id)
    rows: List[Dict[str, Any]] = []
    for clip in clips:
        rows.append(
            {
                "clipRef": clip.instance_id,
                "producerId": clip.producer_id,
                "trackId": clip.track_id,
                "in": int(clip.in_point or "0"),
                "out": int(clip.out_point or clip.in_point or "0"),
                "timelineStart": int(clip.timeline_start),
                "timelineEnd": int(clip.timeline_end),
                "durationFrames": int(clip.duration),
            }
        )
    rows.sort(key=lambda x: (x["timelineStart"], x["trackId"], x["clipRef"]))
    return rows


def _detect_gaps_rows(project: KdenliveProject, track_id: Optional[str] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    playlist_ids: List[str]
    if track_id:
        playlist_ids = [track_id]
    else:
        playlist_ids = [p.get("id", "") for p in project.get_playlists() if p.get("id", "").startswith("playlist")]
    for pid in playlist_ids:
        clips = [c for c in _clip_rows(project, track_id=pid)]
        clips.sort(key=lambda c: c["timelineStart"])
        cursor = 0
        for clip in clips:
            start = int(clip["timelineStart"])
            if start > cursor:
                rows.append({"trackId": pid, "start": cursor, "end": start - 1, "durationFrames": start - cursor})
            cursor = max(cursor, int(clip["timelineEnd"]) + 1)
    return rows


def _producer_media_paths(project: KdenliveProject) -> List[Path]:
    paths: List[Path] = []
    seen = set()
    for producer in project.get_producers():
        elem = producer.element
        if elem is None:
            continue
        resource_prop = elem.find('./property[@name="resource"]')
        if resource_prop is None or not resource_prop.text:
            continue
        resource = resource_prop.text
        if resource == "black" or resource.startswith("#"):
            continue
        path = Path(resource)
        if not path.is_absolute():
            path = (project.project_path.parent / path).resolve()
        if path.exists():
            key = str(path).lower()
            if key not in seen:
                seen.add(key)
                paths.append(path)
    return paths


def _available_mlt_producers() -> set[str]:
    global MLT_PRODUCERS_CACHE
    if MLT_PRODUCERS_CACHE is not None:
        return MLT_PRODUCERS_CACHE
    try:
        melt = _resolve_bin("melt")
        proc = subprocess.run(
            [str(melt), "-query", "producers"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0:
            MLT_PRODUCERS_CACHE = {"qtext"}
            return MLT_PRODUCERS_CACHE
        producers = set()
        for line in proc.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("-"):
                body = stripped.lstrip("-").strip()
                if not body:
                    continue
                token = body.split()[0]
                if token:
                    producers.add(token.lower())
        MLT_PRODUCERS_CACHE = producers
        return producers
    except Exception:
        MLT_PRODUCERS_CACHE = {"qtext"}
        return MLT_PRODUCERS_CACHE


def _srt_timestamp_from_frames(frame: int, fps: float) -> str:
    millis = int(round((max(0, frame) / max(0.001, fps)) * 1000))
    hours = millis // 3_600_000
    millis -= hours * 3_600_000
    minutes = millis // 60_000
    millis -= minutes * 60_000
    seconds = millis // 1000
    millis -= seconds * 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _write_single_cue_srt(project: KdenliveProject, producer_id: str, text: str, duration_frames: int) -> Path:
    folder = project.project_path.parent / ".harness_text"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{producer_id}.srt"
    fps = _project_fps(project)
    end_ts = _srt_timestamp_from_frames(max(1, duration_frames) - 1, fps)
    content = f"1\n00:00:00,000 --> {end_ts}\n{text.strip()}\n"
    path.write_text(content, encoding="utf-8")
    return path


def _collect_text_overlay_cues(project: KdenliveProject) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    text_by_producer: Dict[str, str] = {}
    for producer in project.get_producers():
        elem = producer.element
        if elem is None:
            continue
        mode = (elem.findtext('./property[@name="harness:text_mode"]') or "").strip().lower()
        service = (elem.findtext('./property[@name="mlt_service"]') or "").strip().lower()
        if mode not in {"subtitle", "qtext"} and service not in {"subtitle", "qtext"}:
            continue
        raw = elem.findtext('./property[@name="harness:text_raw"]')
        if not raw:
            raw = elem.findtext('./property[@name="resource"]') or ""
        if raw:
            text_by_producer[producer.id] = raw
    cues: List[Dict[str, Any]] = []
    for clip in _clip_rows(project):
        text = text_by_producer.get(str(clip["producerId"]))
        if not text:
            continue
        cues.append(
            {
                "producerId": str(clip["producerId"]),
                "text": text.strip(),
                "start": int(clip["timelineStart"]),
                "end": int(clip["timelineEnd"]),
            }
        )
    cues.sort(key=lambda c: c["start"])
    return text_by_producer, cues


def _remove_text_overlays(project: KdenliveProject, producer_ids: set[str]) -> None:
    if not producer_ids:
        return
    for playlist in project.get_playlists():
        for node in list(playlist):
            if node.tag != "entry":
                continue
            if str(node.get("producer", "")) in producer_ids:
                playlist.remove(node)
    for producer in list(project.root.findall(".//producer")):
        if str(producer.get("id", "")) in producer_ids:
            project.root.remove(producer)
    main_bin = project.get_main_bin()
    if main_bin is not None:
        for node in list(main_bin):
            if node.tag != "entry":
                continue
            if str(node.get("producer", "")) in producer_ids:
                main_bin.remove(node)
    # Remove empty timeline playlists left behind by text overlays to avoid
    # MLT truncating render length on some builds.
    empty_playlist_ids: List[str] = []
    for playlist in project.get_playlists():
        pid = str(playlist.get("id", ""))
        if not pid.startswith("playlist"):
            continue
        has_entry = any(node.tag == "entry" for node in list(playlist))
        if has_entry:
            continue
        # Keep base audio/video tracks when possible.
        if pid in {"playlist0", "playlist1"}:
            continue
        empty_playlist_ids.append(pid)
    if empty_playlist_ids:
        for tractor in project.root.findall(".//tractor"):
            for track in list(tractor.findall("track")):
                if str(track.get("producer", "")) in empty_playlist_ids:
                    tractor.remove(track)
        for pid in empty_playlist_ids:
            playlist = project.root.find(f'.//playlist[@id="{pid}"]')
            if playlist is not None:
                project.root.remove(playlist)


def _write_cues_srt(path: Path, cues: List[Dict[str, Any]], fps: float, frame_offset: int = 0) -> int:
    lines: List[str] = []
    idx = 1
    for cue in cues:
        start = int(cue["start"]) - frame_offset
        end = int(cue["end"]) - frame_offset
        if end < 0:
            continue
        start = max(0, start)
        end = max(start, end)
        text = str(cue["text"]).strip()
        if not text:
            continue
        lines.append(str(idx))
        lines.append(f"{_srt_timestamp_from_frames(start, fps)} --> {_srt_timestamp_from_frames(end, fps)}")
        lines.append(text)
        lines.append("")
        idx += 1
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return idx - 1


def _recalculate_timeline_bounds(project: KdenliveProject) -> int:
    out_frame = _timeline_max_end(project)
    for tractor in project.root.findall(".//tractor"):
        tractor.set("out", str(out_frame))
        if tractor.get("in") is None:
            tractor.set("in", "0")
    return out_frame


def _set_project_bounds(project: KdenliveProject, in_frame: int, out_frame: int) -> None:
    for tractor in project.root.findall(".//tractor"):
        tractor.set("in", str(max(0, in_frame)))
        tractor.set("out", str(max(in_frame, out_frame)))


def _extract_semver(raw: str) -> Optional[str]:
    for token in raw.replace(",", " ").split():
        stripped = token.strip()
        if stripped and stripped[0].isdigit() and "." in stripped:
            return stripped
    return None


def _binary_version(path: Path, args: List[str]) -> Optional[str]:
    try:
        process = subprocess.run(
            [str(path), *args],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    output = (process.stdout or process.stderr or "").strip()
    return _extract_semver(output) or (output.splitlines()[0] if output else None)


def _latest_kdenlive_version() -> Optional[str]:
    request = urllib.request.Request(
        "https://api.github.com/repos/KDE/kdenlive/releases/latest",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "harness-kdenlive/doctor"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            raw = response.read().decode("utf-8")
        payload = json.loads(raw)
    except Exception:
        return None
    tag = str(payload.get("tag_name", "")).strip()
    return tag.removeprefix("v") if tag else None


def _report_breakage_via_curl(report_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    curl_path = shutil.which("curl")
    if not curl_path:
        return {"reported": False, "error": "curl not found on PATH"}
    process = subprocess.run(
        [
            curl_path,
            "-sS",
            "-X",
            "POST",
            report_url,
            "-H",
            "Content-Type: application/json",
            "--data-binary",
            json.dumps(payload),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if process.returncode != 0:
        return {
            "reported": False,
            "error": (process.stderr or process.stdout or "curl failed").strip(),
        }
    return {
        "reported": True,
        "status": process.returncode,
        "response": (process.stdout or "").strip()[:500],
    }


def _run_doctor(params: Dict[str, Any]) -> Dict[str, Any]:
    report_on_failure = bool(params.get("report_on_failure", True))
    include_render = bool(params.get("include_render", True))
    report_url = str(
        params.get("report_url")
        or os.getenv("HARNESS_KDENLIVE_DOCTOR_REPORT_URL")
        or "https://harness.gg/kdenlive"
    )

    versions: Dict[str, Optional[str]] = {"kdenlive": None, "melt": None, "ffprobe": None}
    binaries: Dict[str, Optional[str]] = {"kdenlive": None, "melt": None, "ffprobe": None}
    for name in ["kdenlive", "melt", "ffprobe"]:
        try:
            binary = _resolve_bin(name)
            binaries[name] = str(binary)
            args = ["--version"] if name != "ffprobe" else ["-version"]
            versions[name] = _binary_version(binary, args)
        except BridgeOperationError:
            binaries[name] = None
            versions[name] = None

    latest = _latest_kdenlive_version()
    installed = versions["kdenlive"]
    is_latest_installed = bool(latest and installed and installed.startswith(latest))

    checks: List[Dict[str, Any]] = []
    broken: List[str] = []
    temp_root = Path(tempfile.mkdtemp(prefix="harness_kdenlive_doctor_"))
    project = temp_root / "doctor_project.kdenlive"
    clone = temp_root / "doctor_clone.kdenlive"
    render_out = temp_root / "doctor_render.mp4"

    def run_check(action: str, check_params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = execute(action, check_params)
            checks.append({"action": action, "ok": True})
            return result
        except Exception as exc:
            checks.append({"action": action, "ok": False, "error": str(exc)})
            broken.append(action)
            return {}

    run_check(
        "project.create",
        {
            "output": str(project),
            "title": "Doctor Project",
            "overwrite": True,
            "width": 1280,
            "height": 720,
            "fps": 30,
        },
    )
    run_check("project.inspect", {"project": str(project)})
    run_check("project.validate", {"project": str(project), "check_files": False})
    imported = run_check(
        "asset.import",
        {
            "project": str(project),
            "media": str(project),
            "producer_id": "doctor_media_1",
            "output": str(project),
        },
    )
    text = run_check(
        "asset.create_text",
        {
            "project": str(project),
            "text": "doctor",
            "duration_frames": 30,
            "track_id": "playlist0",
            "position": 0,
            "producer_id": "doctor_text_1",
            "output": str(project),
        },
    )
    run_check(
        "timeline.stitch_clips",
        {
            "project": str(project),
            "track_id": "playlist0",
            "clip_ids": ["doctor_media_1", "doctor_text_1"],
            "position": 35,
            "duration_frames": 10,
            "gap": 2,
            "output": str(project),
        },
    )
    clip_ref = text.get("clipRef")
    if clip_ref:
        split = run_check(
            "timeline.split_clip",
            {"project": str(project), "clip_ref": clip_ref, "position": 15, "output": str(project)},
        )
        new_ref = split.get("newClipRef")
        if new_ref:
            run_check(
                "timeline.ripple_delete",
                {"project": str(project), "clip_ref": new_ref, "output": str(project)},
            )
    run_check(
        "timeline.insert_gap",
        {
            "project": str(project),
            "track_id": "playlist0",
            "position": 5,
            "length": 3,
            "output": str(project),
        },
    )
    run_check(
        "timeline.remove_all_gaps",
        {"project": str(project), "track_id": "playlist0", "output": str(project)},
    )
    run_check(
        "track.add",
        {
            "project": str(project),
            "track_type": "video",
            "track_id": "playlist9",
            "output": str(project),
        },
    )
    run_check(
        "track.reorder",
        {"project": str(project), "track_id": "playlist9", "index": 0, "output": str(project)},
    )
    run_check(
        "track.remove",
        {"project": str(project), "track_id": "playlist9", "force": True, "output": str(project)},
    )
    run_check(
        "project.clone",
        {"source": str(project), "target": str(clone), "overwrite": True},
    )
    if include_render:
        run_check(
            "render.project",
            {
                "project": str(project),
                "output": str(render_out),
                "start_seconds": 0,
                "duration_seconds": 1,
            },
        )

    healthy = len(broken) == 0
    should_report = report_on_failure and len(broken) > 0 and is_latest_installed
    report: Dict[str, Any] = {"reported": False}
    if should_report:
        payload = {
            "type": "kdenlive_doctor_breakage",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "harnessVersion": __version__,
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
            "kdenlive": {
                "installedVersion": installed,
                "latestVersion": latest,
                "isLatestInstalled": is_latest_installed,
                "binaries": binaries,
            },
            "brokenActions": broken,
            "checks": checks,
        }
        report = _report_breakage_via_curl(report_url, payload)

    return {
        "healthy": healthy,
        "latestCheck": {
            "installedVersion": installed,
            "latestVersion": latest,
            "isLatestInstalled": is_latest_installed,
        },
        "versions": versions,
        "binaries": binaries,
        "checks": checks,
        "brokenActions": broken,
        "report": report,
    }


def _create_project_file(params: Dict[str, Any]) -> Dict[str, Any]:
    output = Path(params["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not bool(params.get("overwrite", False)):
        raise BridgeOperationError("INVALID_INPUT", f"File already exists: {output}")

    title = str(params.get("title", output.stem))
    width = int(params.get("width", 1920))
    height = int(params.get("height", 1080))
    fps = float(params.get("fps", 30))
    if width <= 0 or height <= 0 or fps <= 0:
        raise BridgeOperationError("INVALID_INPUT", "width, height, and fps must be > 0")

    root = etree.Element(
        "mlt",
        LC_NUMERIC="C",
        version="7.14.0",
        root="",
        title=title,
        producer="main_bin",
    )
    fps_num = int(round(fps * 1000))
    fps_den = 1000
    etree.SubElement(
        root,
        "profile",
        description=f"{width}x{height} {fps:.3f} fps",
        width=str(width),
        height=str(height),
        progressive="1",
        sample_aspect_num="1",
        sample_aspect_den="1",
        display_aspect_num="16",
        display_aspect_den="9",
        frame_rate_num=str(fps_num),
        frame_rate_den=str(fps_den),
        colorspace="709",
    )
    created = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    for name, value in [
        ("kdenlive:docproperties.version", "23.08.4"),
        ("kdenlive:docproperties.kdenliveversion", "23.08.4"),
        ("kdenlive:docproperties.generation", "5"),
        ("kdenlive:docproperties.documentid", created),
        ("kdenlive:docproperties.uuid", f"{{{uuid4()}}}"),
    ]:
        prop = etree.SubElement(root, "property", name=name)
        prop.text = value

    black = etree.SubElement(root, "producer", id="black", **{"in": "0", "out": "10000"})
    for name, value in [
        ("length", "10001"),
        ("eof", "pause"),
        ("resource", "black"),
        ("mlt_service", "color"),
    ]:
        prop = etree.SubElement(black, "property", name=name)
        prop.text = value

    main_bin = etree.SubElement(root, "playlist", id="main_bin")
    prop = etree.SubElement(main_bin, "property", name="xml_retain")
    prop.text = "1"
    video = etree.SubElement(root, "playlist", id="playlist0")
    prop = etree.SubElement(video, "property", name="kdenlive:track_name")
    prop.text = "Video 1"
    audio = etree.SubElement(root, "playlist", id="playlist1")
    prop = etree.SubElement(audio, "property", name="kdenlive:track_name")
    prop.text = "Audio 1"

    tractor = etree.SubElement(root, "tractor", id="tractor0", **{"in": "0", "out": "0"})
    marker = etree.SubElement(tractor, "property", name="kdenlive:projectTractor")
    marker.text = "1"
    etree.SubElement(tractor, "track", producer="playlist0", hide="audio")
    etree.SubElement(tractor, "track", producer="playlist1", hide="video")

    sequence = etree.SubElement(root, "tractor", id="timeline_sequence_1", **{"in": "0", "out": "0"})
    seq_uuid = etree.SubElement(sequence, "property", name="kdenlive:sequenceproperties.documentuuid")
    seq_uuid.text = f"{{{uuid4()}}}"
    etree.SubElement(sequence, "track", producer="tractor0")

    preview = etree.SubElement(root, "tractor", id="timeline_preview", **{"in": "0", "out": "0"})
    etree.SubElement(preview, "track", producer="timeline_sequence_1")

    tree = etree.ElementTree(root)
    tree.write(str(output), encoding="utf-8", xml_declaration=True, pretty_print=True)
    return {"path": str(output), "created": True}


def _render_clip(params: Dict[str, Any]) -> Dict[str, Any]:
    source = Path(params["source"])
    output = Path(params["output"])
    if not source.exists():
        raise BridgeOperationError("NOT_FOUND", f"Source not found: {source}")
    start_seconds = float(params.get("start_seconds", 0))
    duration_seconds = float(params["duration_seconds"])
    if duration_seconds <= 0:
        raise BridgeOperationError("INVALID_INPUT", "duration_seconds must be > 0")
    in_frame = int(round(start_seconds * 30))
    out_frame = in_frame + int(round(duration_seconds * 30)) - 1
    melt = _resolve_bin("melt")
    ffprobe = _resolve_bin("ffprobe")

    cmd = [
        str(melt),
        str(source),
        f"in={in_frame}",
        f"out={out_frame}",
        "-consumer",
        f"avformat:{output}",
        f"vcodec={params.get('vcodec', 'libx264')}",
        f"acodec={params.get('acodec', 'aac')}",
        f"ab={params.get('audio_bitrate', '192k')}",
        f"crf={params.get('crf', '18')}",
        f"preset={params.get('preset', 'fast')}",
    ]
    rendered_duration = _render_and_probe_duration(cmd, output)

    return {
        "source": str(source),
        "output": str(output),
        "durationSeconds": rendered_duration,
        "targetDurationSeconds": duration_seconds,
    }


def execute(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if method == "system.health":
            return {"status": "ok", "version": __version__}
        if method == "system.version":
            return {"version": __version__}
        if method == "system.actions":
            return {"actions": ACTION_METHODS}
        if method == "system.doctor":
            return _run_doctor(params)
        if method == "system.soak":
            iterations = int(params.get("iterations", 100))
            duration_seconds = float(params.get("duration_seconds", 5))
            if iterations <= 0 or duration_seconds <= 0:
                raise BridgeOperationError("INVALID_INPUT", "iterations and duration_seconds must be > 0")
            action = str(params.get("action", "system.health"))
            action_params = params.get("action_params", {})
            failures = 0
            latencies: List[float] = []
            start = datetime.utcnow()
            end_by = time.perf_counter() + duration_seconds
            for _ in range(iterations):
                if time.perf_counter() > end_by:
                    break
                t0 = time.perf_counter()
                try:
                    execute(action, action_params)
                except Exception:
                    failures += 1
                latencies.append((time.perf_counter() - t0) * 1000)
            ran = len(latencies)
            return {
                "action": action,
                "iterationsRequested": iterations,
                "iterationsRun": ran,
                "durationSeconds": duration_seconds,
                "failures": failures,
                "stable": failures == 0,
                "latencyMs": {
                    "min": round(min(latencies), 3) if latencies else 0.0,
                    "max": round(max(latencies), 3) if latencies else 0.0,
                    "avg": round(sum(latencies) / ran, 3) if ran else 0.0,
                },
                "startedAt": start.isoformat() + "Z",
            }
        if method == "project.create":
            return _create_project_file(params)
        if method == "project.clone":
            source = Path(params["source"])
            if not source.exists():
                raise BridgeOperationError("NOT_FOUND", f"Project file not found: {source}")
            target = Path(params["target"])
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and not bool(params.get("overwrite", False)):
                raise BridgeOperationError("INVALID_INPUT", f"File already exists: {target}")
            shutil.copy2(source, target)
            return _mutation_payload({"source": str(source), "target": str(target), "cloned": True})
        if method == "project.plan_edit":
            source = _load(params["project"])
            target = source.clone()
            action = str(params["action"])
            action_params = dict(params.get("params", {}))
            action_params["project"] = str(target.project_path)
            action_params["output"] = None
            action_params["dry_run"] = True
            execute(action, action_params)
            diff = DiffEngine(source, target).to_dict()
            return {
                "project": str(source.project_path),
                "action": action,
                "params": params.get("params", {}),
                "previewDiff": diff,
                "wouldChange": diff["stats"]["total_changes"] > 0,
            }
        if method == "project.undo":
            loaded = _load(params["project"])
            txn = TransactionManager(loaded)
            history = txn.get_history()
            if not history:
                raise BridgeOperationError("INVALID_INPUT", "No snapshots available for undo")
            target_id = str(params.get("snapshot_id") or history[0]["id"])
            redo_dir = loaded.project_path.parent / ".kdenlive_history" / "redo"
            redo_dir.mkdir(parents=True, exist_ok=True)
            current_copy = redo_dir / f"redo_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.kdenlive"
            loaded.save(current_copy)
            txn.rollback_to_snapshot(target_id)
            saved = _save(loaded, str(loaded.project_path))
            return _mutation_payload({"snapshotId": target_id, "savedTo": saved})
        if method == "project.redo":
            loaded = _load(params["project"])
            redo_dir = loaded.project_path.parent / ".kdenlive_history" / "redo"
            redo_files = sorted(redo_dir.glob("redo_*.kdenlive"), reverse=True)
            if not redo_files:
                raise BridgeOperationError("INVALID_INPUT", "No redo entries available")
            redo_file = redo_files[0]
            redo_project = KdenliveProject(redo_file)
            loaded.tree = redo_project.tree
            loaded.root = redo_project.root
            saved = _save(loaded, str(loaded.project_path))
            redo_file.unlink(missing_ok=True)
            return _mutation_payload({"savedTo": saved, "restoredFrom": str(redo_file)})
        if method == "project.autosave":
            project_path = str(Path(params["project"]).resolve())
            enabled = bool(params.get("enabled", True))
            interval_seconds = int(params.get("interval_seconds", 60))
            if interval_seconds <= 0:
                raise BridgeOperationError("INVALID_INPUT", "interval_seconds must be > 0")
            if enabled:
                AUTOSAVE_STATE[project_path] = {
                    "enabled": True,
                    "intervalSeconds": interval_seconds,
                    "updatedAt": datetime.utcnow().isoformat() + "Z",
                }
            else:
                AUTOSAVE_STATE.pop(project_path, None)
            state = AUTOSAVE_STATE.get(project_path, {"enabled": False, "intervalSeconds": interval_seconds})
            return _mutation_payload(
                {"project": project_path, "autosave": state},
                changed=True,
                idempotent=False,
            )
        if method == "project.pack":
            loaded = _load(params["project"])
            output_dir = Path(params["output_dir"])
            media_dir_name = str(params.get("media_dir_name", "media"))
            media_dir = output_dir / media_dir_name
            output_dir.mkdir(parents=True, exist_ok=True)
            media_dir.mkdir(parents=True, exist_ok=True)
            packed_project_path = output_dir / loaded.project_path.name
            copied: List[Dict[str, str]] = []
            for source in _producer_media_paths(loaded):
                target = media_dir / source.name
                if not target.exists():
                    shutil.copy2(source, target)
                copied.append({"source": str(source), "target": str(target)})
            packed_copy = loaded.clone()
            for producer in packed_copy.get_producers():
                if producer.element is None:
                    continue
                prop = producer.element.find('./property[@name="resource"]')
                if prop is None or not prop.text:
                    continue
                current = Path(prop.text)
                if current.name and (media_dir / current.name).exists():
                    prop.text = str(Path(media_dir_name) / current.name)
            packed_copy.save(packed_project_path)
            return _mutation_payload(
                {
                    "project": str(loaded.project_path),
                    "packedProject": str(packed_project_path),
                    "mediaDirectory": str(media_dir),
                    "copiedMedia": copied,
                    "copiedCount": len(copied),
                }
            )
        if method == "project.recalculate_timeline_bounds":
            loaded = _load(params["project"])
            out_frame = _recalculate_timeline_bounds(loaded)
            saved = _save(loaded, params.get("output"))
            return _mutation_payload({"projectOut": out_frame, "savedTo": saved})
        if method == "project.inspect":
            loaded = _load(params["project"])
            clips = loaded.get_clips_on_timeline()
            return {
                "path": str(params["project"]),
                "generation": loaded.generation,
                "version": loaded.version,
                "statistics": {
                    "tracks": len(loaded.get_tracks()),
                    "producers": len(loaded.get_producers()),
                    "clips": len(clips),
                    "durationFrames": max([c.timeline_end for c in clips], default=-1) + 1,
                },
            }
        if method == "project.validate":
            loaded = _load(params["project"])
            validator = ProjectValidator(loaded)
            check_files = bool(params.get("check_files", True))
            issues = validator.validate_all(check_files=check_files)
            errors = [e for e in issues if e.severity == "error"]
            warnings = [e for e in issues if e.severity == "warning"]
            return {
                "path": str(params["project"]),
                "isValid": len(errors) == 0,
                "errorCount": len(errors),
                "warningCount": len(warnings),
                "errors": [e.__dict__ for e in errors],
                "warnings": [e.__dict__ for e in warnings],
            }
        if method == "project.diff":
            source = _load(params["source"])
            target = _load(params["target"])
            return DiffEngine(source, target).to_dict()
        if method == "project.snapshot":
            loaded = _load(params["project"])
            snap = TransactionManager(loaded).create_snapshot(params["description"])
            return {"snapshotId": snap}
        if method == "asset.import":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            dry_run = bool(params.get("dry_run", False))
            media_path = Path(params["media"])
            if not media_path.exists():
                raise BridgeOperationError("NOT_FOUND", f"Media file not found: {media_path}")
            producer_id = params.get("producer_id") or _next_producer_id(loaded)
            if loaded.get_producers(id_filter=producer_id):
                raise BridgeOperationError("INVALID_INPUT", f"Producer '{producer_id}' already exists")
            fps = _project_fps(loaded)
            duration_seconds = _probe_media_duration_seconds(media_path)
            duration_frames = (
                max(1, int(round(duration_seconds * fps))) if duration_seconds is not None else int(params.get("fallback_frames", 250))
            )
            producer = etree.SubElement(
                loaded.root,
                "producer",
                id=str(producer_id),
                **{"in": "0", "out": str(duration_frames - 1)},
            )
            for name, value in [
                ("resource", str(media_path)),
                ("mlt_service", "avformat"),
                ("kdenlive:clipname", media_path.stem),
                ("kdenlive:id", _next_kdenlive_clip_id(loaded)),
                ("kdenlive:duration", str(duration_frames)),
                ("length", str(duration_frames)),
            ]:
                prop = etree.SubElement(producer, "property", name=name)
                prop.text = value
            _append_main_bin_entry(loaded, str(producer_id), duration_frames - 1)
            saved = None if dry_run else _save(loaded, params.get("output"))
            result = _mutation_payload(
                {
                    "producerId": str(producer_id),
                    "durationFrames": duration_frames,
                    "savedTo": saved,
                }
            )
            if dry_run:
                result["dryRun"] = True
            return result
        if method == "asset.create_text":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            text = str(params["text"])
            if not text:
                raise BridgeOperationError("INVALID_INPUT", "text is required")
            duration_frames = int(params.get("duration_frames", 90))
            if duration_frames <= 0:
                raise BridgeOperationError("INVALID_INPUT", "duration_frames must be > 0")
            producer_id = params.get("producer_id") or f"text_{uuid4().hex[:12]}"
            if loaded.get_producers(id_filter=producer_id):
                raise BridgeOperationError("INVALID_INPUT", f"Producer '{producer_id}' already exists")
            producer = etree.SubElement(
                loaded.root,
                "producer",
                id=str(producer_id),
                **{"in": "0", "out": str(duration_frames - 1)},
            )
            producers = _available_mlt_producers()
            can_qtext = "qtext" in producers
            can_subtitle = "subtitle" in producers
            warnings: List[str] = []
            if can_qtext:
                style = {
                    "fgcolour": str(params.get("color", "#ffffff")),
                    "bgcolour": str(params.get("background", "#00000000")),
                    "family": str(params.get("font", "DejaVu Sans")),
                    "size": str(params.get("size", 64)),
                    "geometry": str(params.get("geometry", "0%/0%:100%x100%")),
                    "halign": str(params.get("halign", "center")),
                    "valign": str(params.get("valign", "center")),
                }
                prop_rows = [
                    ("resource", text),
                    ("mlt_service", "qtext"),
                    ("harness:text_mode", "qtext"),
                    ("kdenlive:clipname", str(params.get("name", "Text"))),
                    ("kdenlive:id", _next_kdenlive_clip_id(loaded)),
                    ("kdenlive:duration", str(duration_frames)),
                    ("length", str(duration_frames)),
                    *style.items(),
                ]
            elif can_subtitle:
                subtitle_path = _write_single_cue_srt(loaded, str(producer_id), text, duration_frames)
                prop_rows = [
                    ("resource", str(subtitle_path)),
                    ("mlt_service", "subtitle"),
                    ("harness:text_mode", "subtitle"),
                    ("harness:text_raw", text),
                    ("kdenlive:clipname", str(params.get("name", "Text"))),
                    ("kdenlive:id", _next_kdenlive_clip_id(loaded)),
                    ("kdenlive:duration", str(duration_frames)),
                    ("length", str(duration_frames)),
                ]
                warnings.append("qtext producer unavailable; used subtitle file fallback")
                for unsupported in ("font", "size", "color", "background", "geometry", "halign", "valign"):
                    if params.get(unsupported) is not None:
                        warnings.append(f"style option '{unsupported}' ignored by subtitle fallback")
            else:
                raise BridgeOperationError("ERROR", "No supported text producer available (qtext/subtitle)")

            for name, value in prop_rows:
                prop = etree.SubElement(producer, "property", name=name)
                prop.text = value
            _append_main_bin_entry(loaded, str(producer_id), duration_frames - 1)
            clip_ref = None
            track_id = params.get("track_id")
            if track_id:
                timeline = TimelineAPI(loaded)
                position = int(params.get("position", 0))
                clip_ref = timeline.add_clip(
                    clip_id=str(producer_id),
                    track_id=str(track_id),
                    position=position,
                    in_point="0",
                    out_point=str(duration_frames - 1),
                )
            saved = _save(loaded, params.get("output"))
            return _mutation_payload(
                {
                    "producerId": str(producer_id),
                    "durationFrames": duration_frames,
                    "clipRef": clip_ref,
                    "savedTo": saved,
                },
                warnings=warnings,
            )
        if method == "asset.update_text":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            producer_id = str(params["producer_id"])
            producers = loaded.get_producers(id_filter=producer_id)
            if not producers:
                raise BridgeOperationError("INVALID_INPUT", f"Producer '{producer_id}' not found")
            producer = producers[0].element
            if producer is None:
                raise BridgeOperationError("INVALID_INPUT", f"Producer '{producer_id}' not found")
            service = (producer.findtext('./property[@name="mlt_service"]') or "").strip().lower()
            text_mode = (producer.findtext('./property[@name="harness:text_mode"]') or "").strip().lower()
            changed = False
            warnings: List[str] = []
            if service == "subtitle" or text_mode == "subtitle":
                current_out = int(producer.get("out", "0"))
                frames = int(params.get("duration_frames", current_out + 1))
                if frames <= 0:
                    raise BridgeOperationError("INVALID_INPUT", "duration_frames must be > 0")
                text_prop = _get_or_create_property(producer, "harness:text_raw")
                raw_text = str(params.get("text", text_prop.text or ""))
                subtitle_path = _write_single_cue_srt(loaded, producer_id, raw_text, frames)
                for key, value in {
                    "resource": str(subtitle_path),
                    "harness:text_raw": raw_text,
                    "kdenlive:duration": str(frames),
                    "length": str(frames),
                }.items():
                    prop = producer.find(f'./property[@name="{key}"]')
                    if prop is None:
                        prop = etree.SubElement(producer, "property", name=key)
                        prop.text = value
                        changed = True
                    elif (prop.text or "") != value:
                        prop.text = value
                        changed = True
                if str(producer.get("out", "")) != str(frames - 1):
                    producer.set("out", str(frames - 1))
                    changed = True
                for unsupported in ("font", "size", "color", "background", "geometry", "halign", "valign"):
                    if params.get(unsupported) is not None:
                        warnings.append(f"style option '{unsupported}' ignored by subtitle fallback")
            else:
                updates = {
                    "resource": params.get("text"),
                    "family": params.get("font"),
                    "size": params.get("size"),
                    "fgcolour": params.get("color"),
                    "bgcolour": params.get("background"),
                    "geometry": params.get("geometry"),
                }
                if params.get("duration_frames") is not None:
                    frames = int(params["duration_frames"])
                    if frames <= 0:
                        raise BridgeOperationError("INVALID_INPUT", "duration_frames must be > 0")
                    producer.set("out", str(frames - 1))
                    updates["kdenlive:duration"] = str(frames)
                    updates["length"] = str(frames)
                for key, value in updates.items():
                    if value is None:
                        continue
                    prop = producer.find(f'./property[@name="{key}"]')
                    value_str = str(value)
                    if prop is None:
                        prop = etree.SubElement(producer, "property", name=key)
                        prop.text = value_str
                        changed = True
                    elif (prop.text or "") != value_str:
                        prop.text = value_str
                        changed = True
            saved = _save(loaded, params.get("output"))
            return _mutation_payload(
                {"producerId": producer_id, "savedTo": saved},
                changed=changed,
                idempotent=not changed,
                warnings=warnings,
            )
        if method == "asset.metadata":
            loaded = _load(params["project"])
            producer_id = str(params["producer_id"])
            return _producer_resource_metadata(loaded, producer_id)
        if method == "asset.replace":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            producer_id = str(params["producer_id"])
            new_media = Path(params["media"])
            if not new_media.exists():
                raise BridgeOperationError("NOT_FOUND", f"Media file not found: {new_media}")
            producer = _producer_element(loaded, producer_id)
            resource_prop = _get_or_create_property(producer, "resource")
            changed = False
            if (resource_prop.text or "") != str(new_media):
                resource_prop.text = str(new_media)
                changed = True
            if bool(params.get("update_name", True)):
                clipname = _get_or_create_property(producer, "kdenlive:clipname")
                if (clipname.text or "") != new_media.stem:
                    clipname.text = new_media.stem
                    changed = True
            duration = _probe_media_duration_seconds(new_media)
            if duration is not None:
                frames = max(1, int(round(duration * _project_fps(loaded))))
                new_out = str(frames - 1)
                if producer.get("out") != new_out:
                    producer.set("out", new_out)
                    changed = True
                duration_prop = _get_or_create_property(producer, "kdenlive:duration")
                if (duration_prop.text or "") != str(frames):
                    duration_prop.text = str(frames)
                    changed = True
            saved = _save(loaded, params.get("output"))
            return _mutation_payload(
                {"producerId": producer_id, "resource": str(new_media), "savedTo": saved},
                changed=changed,
                idempotent=not changed,
            )
        if method == "bin.list":
            loaded = _load(params["project"])
            folders = _load_bin_folders(loaded)
            producers = []
            for p in loaded.get_producers():
                folder_prop = loaded.get_property("kdenlive:folderid", p.element) if p.element is not None else None
                folder_id = int(folder_prop) if folder_prop and folder_prop.lstrip("-").isdigit() else -1
                producers.append(
                    {
                        "producerId": p.id,
                        "name": loaded.get_property("kdenlive:clipname", p.element) if p.element is not None else None,
                        "resource": p.resource,
                        "folderId": folder_id,
                    }
                )
            return {"folders": list(folders.values()), "assets": producers}
        if method == "bin.create_folder":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            name = str(params["name"]).strip()
            if not name:
                raise BridgeOperationError("INVALID_INPUT", "name is required")
            parent_id = int(params.get("parent_id", -1))
            folders = _load_bin_folders(loaded)
            existing = next((f for f in folders.values() if f["name"] == name and int(f["parentId"] or -1) == parent_id), None)
            if existing is not None:
                saved = _save(loaded, params.get("output"))
                return _mutation_payload(
                    {"folder": existing, "savedTo": saved},
                    changed=False,
                    idempotent=True,
                )
            folder_id = _next_folder_id(folders)
            key = f"folder_{folder_id}"
            folders[key] = {"id": folder_id, "name": name, "parentId": parent_id}
            _save_bin_folders(loaded, folders)
            saved = _save(loaded, params.get("output"))
            return _mutation_payload({"folder": folders[key], "savedTo": saved})
        if method == "bin.move_asset":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            producer_id = str(params["producer_id"])
            folder_id = int(params["folder_id"])
            folders = _load_bin_folders(loaded)
            if folder_id != -1 and not any(int(f.get("id", -1)) == folder_id for f in folders.values()):
                raise BridgeOperationError("INVALID_INPUT", f"Folder id not found: {folder_id}")
            producer = _producer_element(loaded, producer_id)
            folder_prop = _get_or_create_property(producer, "kdenlive:folderid")
            changed = (folder_prop.text or "-1") != str(folder_id)
            folder_prop.text = str(folder_id)
            saved = _save(loaded, params.get("output"))
            return _mutation_payload(
                {"producerId": producer_id, "folderId": folder_id, "savedTo": saved},
                changed=changed,
                idempotent=not changed,
            )
        if method == "effect.list":
            loaded = _load(params["project"])
            clip_ref = str(params["clip_ref"])
            entry, _ = _resolve_clip_element(loaded, clip_ref)
            effects = []
            for filt in entry.findall("./filter"):
                effects.append(
                    {
                        "id": filt.get("id"),
                        "properties": {
                            p.get("name"): p.text for p in filt.findall("./property") if p.get("name")
                        },
                    }
                )
            return {"clipRef": clip_ref, "effects": effects}
        if method == "effect.apply":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            clip_ref = str(params["clip_ref"])
            service = str(params["service"])
            entry, _ = _resolve_clip_element(loaded, clip_ref)
            effect_id = _ensure_effect_id(entry, params.get("effect_id"))
            existing = entry.find(f'./filter[@id="{effect_id}"]')
            if existing is not None:
                saved = _save(loaded, params.get("output"))
                return _mutation_payload(
                    {"clipRef": clip_ref, "effectId": effect_id, "savedTo": saved},
                    changed=False,
                    idempotent=True,
                )
            filt = etree.SubElement(entry, "filter", id=effect_id)
            mlt = etree.SubElement(filt, "property", name="mlt_service")
            mlt.text = service
            for key, value in dict(params.get("properties", {})).items():
                prop = etree.SubElement(filt, "property", name=str(key))
                prop.text = str(value)
            saved = _save(loaded, params.get("output"))
            return _mutation_payload({"clipRef": clip_ref, "effectId": effect_id, "savedTo": saved})
        if method == "effect.update":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            clip_ref = str(params["clip_ref"])
            effect_id = str(params["effect_id"])
            entry, _ = _resolve_clip_element(loaded, clip_ref)
            filt = entry.find(f'./filter[@id="{effect_id}"]')
            if filt is None:
                raise BridgeOperationError("INVALID_INPUT", f"Effect '{effect_id}' not found")
            changed = False
            for key, value in dict(params.get("properties", {})).items():
                prop = filt.find(f'./property[@name="{key}"]')
                text_val = str(value)
                if prop is None:
                    prop = etree.SubElement(filt, "property", name=str(key))
                    prop.text = text_val
                    changed = True
                elif (prop.text or "") != text_val:
                    prop.text = text_val
                    changed = True
            saved = _save(loaded, params.get("output"))
            return _mutation_payload(
                {"clipRef": clip_ref, "effectId": effect_id, "savedTo": saved},
                changed=changed,
                idempotent=not changed,
            )
        if method == "effect.remove":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            clip_ref = str(params["clip_ref"])
            effect_id = str(params["effect_id"])
            entry, _ = _resolve_clip_element(loaded, clip_ref)
            filt = entry.find(f'./filter[@id="{effect_id}"]')
            if filt is None:
                saved = _save(loaded, params.get("output"))
                return _mutation_payload(
                    {"clipRef": clip_ref, "effectId": effect_id, "savedTo": saved},
                    changed=False,
                    idempotent=True,
                )
            entry.remove(filt)
            saved = _save(loaded, params.get("output"))
            return _mutation_payload({"clipRef": clip_ref, "effectId": effect_id, "savedTo": saved})
        if method == "effect.keyframes":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            clip_ref = str(params["clip_ref"])
            effect_id = str(params["effect_id"])
            parameter = str(params["parameter"])
            entry, _ = _resolve_clip_element(loaded, clip_ref)
            filt = entry.find(f'./filter[@id="{effect_id}"]')
            if filt is None:
                raise BridgeOperationError("INVALID_INPUT", f"Effect '{effect_id}' not found")
            keyframes = params.get("keyframes")
            if not isinstance(keyframes, list):
                raise BridgeOperationError("INVALID_INPUT", "keyframes must be a list")
            keyframes_text = json.dumps(keyframes, separators=(",", ":"))
            prop_name = f"harness:keyframes:{parameter}"
            prop = filt.find(f'./property[@name="{prop_name}"]')
            changed = False
            if prop is None:
                prop = etree.SubElement(filt, "property", name=prop_name)
                prop.text = keyframes_text
                changed = True
            elif (prop.text or "") != keyframes_text:
                prop.text = keyframes_text
                changed = True
            saved = _save(loaded, params.get("output"))
            return _mutation_payload(
                {"clipRef": clip_ref, "effectId": effect_id, "parameter": parameter, "savedTo": saved},
                changed=changed,
                idempotent=not changed,
            )
        if method == "transition.list":
            loaded = _load(params["project"])
            tractor = _get_project_tractor(loaded)
            transitions = []
            for t in tractor.findall("./transition"):
                transitions.append(
                    {
                        "id": t.get("id"),
                        "in": t.get("in"),
                        "out": t.get("out"),
                        "properties": {
                            p.get("name"): p.text for p in t.findall("./property") if p.get("name")
                        },
                    }
                )
            return {"transitions": transitions}
        if method == "transition.apply":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            tractor = _get_project_tractor(loaded)
            transition_id = str(params.get("transition_id") or f"transition_{uuid4().hex[:12]}")
            if tractor.find(f'./transition[@id="{transition_id}"]') is not None:
                saved = _save(loaded, params.get("output"))
                return _mutation_payload(
                    {"transitionId": transition_id, "savedTo": saved},
                    changed=False,
                    idempotent=True,
                )
            in_frame = int(params.get("in_frame", 0))
            out_frame = int(params.get("out_frame", in_frame))
            if in_frame < 0 or out_frame < in_frame:
                raise BridgeOperationError("INVALID_INPUT", "invalid in_frame/out_frame")
            trans = etree.SubElement(
                tractor,
                "transition",
                id=transition_id,
                **{"in": str(in_frame), "out": str(out_frame)},
            )
            service = etree.SubElement(trans, "property", name="mlt_service")
            service.text = str(params.get("service", "mix"))
            for key, value in dict(params.get("properties", {})).items():
                prop = etree.SubElement(trans, "property", name=str(key))
                prop.text = str(value)
            saved = _save(loaded, params.get("output"))
            return _mutation_payload({"transitionId": transition_id, "savedTo": saved})
        if method == "transition.remove":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            tractor = _get_project_tractor(loaded)
            transition_id = str(params["transition_id"])
            trans = tractor.find(f'./transition[@id="{transition_id}"]')
            if trans is None:
                saved = _save(loaded, params.get("output"))
                return _mutation_payload(
                    {"transitionId": transition_id, "savedTo": saved},
                    changed=False,
                    idempotent=True,
                )
            tractor.remove(trans)
            saved = _save(loaded, params.get("output"))
            return _mutation_payload({"transitionId": transition_id, "savedTo": saved})
        if method == "transition.wipe":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            wipe_preset = str(params.get("preset", "circle")).lower()
            resource_map = {
                "circle": "circle",
                "clock": "clock",
                "barn": "barn",
                "iris": "circle",
                "linear": "luma",
            }
            resource = resource_map.get(wipe_preset)
            if resource is None:
                raise BridgeOperationError("INVALID_INPUT", f"Unknown wipe preset: {wipe_preset}")
            transition_params = {
                "project": params["project"],
                "transition_id": params.get("transition_id"),
                "service": "luma",
                "in_frame": int(params.get("in_frame", 0)),
                "out_frame": int(params.get("out_frame", 0)),
                "properties": {
                    "resource": resource,
                    "softness": str(params.get("softness", 0.05)),
                    "invert": str(int(bool(params.get("invert", False)))),
                },
                "output": params.get("output"),
            }
            return execute("transition.apply", transition_params)
        if method == "timeline.add_clip":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            dry_run = bool(params.get("dry_run", False))
            timeline = TimelineAPI(loaded)
            clip_ref = timeline.add_clip(
                clip_id=params["clip_id"],
                track_id=params["track_id"],
                position=int(params["position"]),
                in_point=str(params.get("in_point", "0")),
                out_point=params.get("out_point"),
            )
            saved = None if dry_run else _save(loaded, params.get("output"))
            result = _mutation_payload({"clipRef": clip_ref, "savedTo": saved})
            if dry_run:
                result["dryRun"] = True
            return result
        if method == "timeline.move_clip":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            dry_run = bool(params.get("dry_run", False))
            timeline = TimelineAPI(loaded)
            current = timeline._resolve_clip(params["clip_ref"])
            if current and current.track_id == params["track_id"] and current.timeline_start == int(params["position"]):
                saved = None if dry_run else _save(loaded, params.get("output"))
                return _mutation_payload(
                    {"clipRef": params["clip_ref"], "savedTo": saved},
                    changed=False,
                    idempotent=True,
                )
            ok = timeline.move_clip(
                clip_ref=params["clip_ref"],
                new_track=params["track_id"],
                new_position=int(params["position"]),
            )
            if not ok:
                raise BridgeOperationError("INVALID_INPUT", f"Clip not found: {params['clip_ref']}")
            saved = None if dry_run else _save(loaded, params.get("output"))
            result = _mutation_payload({"clipRef": params["clip_ref"], "savedTo": saved})
            if dry_run:
                result["dryRun"] = True
            return result
        if method == "timeline.trim_clip":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            dry_run = bool(params.get("dry_run", False))
            timeline = TimelineAPI(loaded)
            existing = timeline._resolve_clip(params["clip_ref"])
            if existing is None:
                raise BridgeOperationError("INVALID_INPUT", f"Clip not found: {params['clip_ref']}")
            same_in = params.get("in_point") is None or str(params.get("in_point")) == str(existing.in_point)
            same_out = params.get("out_point") is None or str(params.get("out_point")) == str(existing.out_point)
            if same_in and same_out:
                saved = None if dry_run else _save(loaded, params.get("output"))
                return _mutation_payload(
                    {"clipRef": params["clip_ref"], "savedTo": saved},
                    changed=False,
                    idempotent=True,
                )
            ok = timeline.trim_clip(
                clip_ref=params["clip_ref"],
                new_in=params.get("in_point"),
                new_out=params.get("out_point"),
            )
            if not ok:
                raise BridgeOperationError("INVALID_INPUT", f"Clip not found: {params['clip_ref']}")
            saved = None if dry_run else _save(loaded, params.get("output"))
            result = _mutation_payload({"clipRef": params["clip_ref"], "savedTo": saved})
            if dry_run:
                result["dryRun"] = True
            return result
        if method == "timeline.remove_clip":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            dry_run = bool(params.get("dry_run", False))
            timeline = TimelineAPI(loaded)
            ok = timeline.remove_clip(
                clip_ref=params["clip_ref"],
                close_gap=bool(params.get("close_gap", False)),
            )
            if not ok:
                raise BridgeOperationError("INVALID_INPUT", f"Clip not found: {params['clip_ref']}")
            saved = None if dry_run else _save(loaded, params.get("output"))
            result = _mutation_payload({"clipRef": params["clip_ref"], "savedTo": saved})
            if dry_run:
                result["dryRun"] = True
            return result
        if method == "timeline.split_clip":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            timeline = TimelineAPI(loaded)
            new_ref = timeline.split_clip(
                clip_ref=params["clip_ref"],
                position=int(params["position"]),
            )
            dry_run = bool(params.get("dry_run", False))
            saved = None if dry_run else _save(loaded, params.get("output"))
            result = _mutation_payload({"clipRef": params["clip_ref"], "newClipRef": new_ref, "savedTo": saved})
            if dry_run:
                result["dryRun"] = True
            return result
        if method == "timeline.ripple_delete":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            timeline = TimelineAPI(loaded)
            ok = timeline.ripple_delete(clip_ref=params["clip_ref"])
            if not ok:
                raise BridgeOperationError("INVALID_INPUT", f"Clip not found: {params['clip_ref']}")
            dry_run = bool(params.get("dry_run", False))
            saved = None if dry_run else _save(loaded, params.get("output"))
            result = _mutation_payload({"clipRef": params["clip_ref"], "savedTo": saved})
            if dry_run:
                result["dryRun"] = True
            return result
        if method == "timeline.insert_gap":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            timeline = TimelineAPI(loaded)
            timeline.insert_gap(
                track_id=params["track_id"],
                position=int(params["position"]),
                length=int(params["length"]),
            )
            dry_run = bool(params.get("dry_run", False))
            saved = None if dry_run else _save(loaded, params.get("output"))
            result = _mutation_payload({"trackId": params["track_id"], "savedTo": saved})
            if dry_run:
                result["dryRun"] = True
            return result
        if method == "timeline.remove_all_gaps":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            timeline = TimelineAPI(loaded)
            removed = timeline.remove_all_gaps(track_id=params["track_id"])
            dry_run = bool(params.get("dry_run", False))
            saved = None if dry_run else _save(loaded, params.get("output"))
            return _mutation_payload(
                {"trackId": params["track_id"], "removedFrames": removed, "savedTo": saved},
                changed=removed > 0,
                idempotent=removed == 0,
            )
        if method == "timeline.stitch_clips":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            timeline = TimelineAPI(loaded)
            track_id = str(params["track_id"])
            clip_ids = [str(c) for c in params.get("clip_ids", [])]
            if not clip_ids:
                raise BridgeOperationError("INVALID_INPUT", "clip_ids must contain at least one producer id")
            position = params.get("position")
            cursor = int(position) if position is not None else max(
                [c.timeline_end for c in timeline.get_clips(track_id)], default=-1
            ) + 1
            gap = int(params.get("gap", 0))
            if gap < 0:
                raise BridgeOperationError("INVALID_INPUT", "gap must be >= 0")
            uniform_duration = params.get("duration_frames")
            clip_refs: List[str] = []
            for clip_id in clip_ids:
                duration = int(uniform_duration) if uniform_duration is not None else _producer_duration_frames(loaded, clip_id)
                if duration <= 0:
                    raise BridgeOperationError("INVALID_INPUT", "duration_frames must be > 0")
                clip_ref = timeline.add_clip(
                    clip_id=clip_id,
                    track_id=track_id,
                    position=cursor,
                    in_point="0",
                    out_point=str(duration - 1),
                )
                clip_refs.append(clip_ref)
                cursor += duration + gap
            dry_run = bool(params.get("dry_run", False))
            saved = None if dry_run else _save(loaded, params.get("output"))
            result = _mutation_payload({"trackId": track_id, "clipRefs": clip_refs, "savedTo": saved})
            if dry_run:
                result["dryRun"] = True
            return result
        if method == "timeline.list_clips":
            loaded = _load(params["project"])
            track_id = params.get("track_id")
            producer_id = params.get("producer_id")
            clips = _clip_rows(loaded, track_id=str(track_id) if track_id else None)
            if producer_id:
                clips = [c for c in clips if c["producerId"] == str(producer_id)]
            return {"count": len(clips), "clips": clips}
        if method == "timeline.select_zone":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            zone_in = int(params.get("zone_in", 0))
            zone_out = int(params["zone_out"])
            if zone_in < 0 or zone_out < zone_in:
                raise BridgeOperationError("INVALID_INPUT", "zone_out must be >= zone_in and zone_in >= 0")
            p_in = _get_or_create_property(loaded.root, "harness:zone-in")
            p_out = _get_or_create_property(loaded.root, "harness:zone-out")
            changed = False
            if (p_in.text or "") != str(zone_in):
                p_in.text = str(zone_in)
                changed = True
            if (p_out.text or "") != str(zone_out):
                p_out.text = str(zone_out)
                changed = True
            saved = _save(loaded, params.get("output"))
            return _mutation_payload(
                {"zoneIn": zone_in, "zoneOut": zone_out, "savedTo": saved},
                changed=changed,
                idempotent=not changed,
            )
        if method == "timeline.detect_gaps":
            loaded = _load(params["project"])
            track_id = params.get("track_id")
            rows = _detect_gaps_rows(loaded, track_id=str(track_id) if track_id else None)
            total = sum(int(r["durationFrames"]) for r in rows)
            return {"count": len(rows), "totalFrames": total, "gaps": rows}
        if method == "clip.resolve":
            loaded = _load(params["project"])
            selector = str(params["selector"])
            track_id = params.get("track_id")
            at_frame = params.get("at_frame")
            clips = _clip_rows(loaded, track_id=str(track_id) if track_id else None)
            if not clips:
                raise BridgeOperationError("INVALID_INPUT", "No clips found on timeline")

            by_ref = [c for c in clips if c["clipRef"] == selector]
            if len(by_ref) == 1:
                return {"selector": selector, "matchedBy": "clipRef", "clip": by_ref[0], "candidates": 1}

            candidates: List[Dict[str, Any]] = [c for c in clips if c["producerId"] == selector]
            if not candidates and selector.isdigit():
                idx = int(selector)
                if idx < 0 or idx >= len(clips):
                    raise BridgeOperationError(
                        "INVALID_INPUT",
                        f"Clip index {idx} out of range (0-{len(clips)-1})",
                    )
                return {"selector": selector, "matchedBy": "index", "clip": clips[idx], "candidates": 1}

            if at_frame is not None:
                frame = int(at_frame)
                frame_hits = [c for c in candidates if c["timelineStart"] <= frame <= c["timelineEnd"]]
                if len(frame_hits) == 1:
                    return {
                        "selector": selector,
                        "matchedBy": "producer+frame",
                        "clip": frame_hits[0],
                        "candidates": len(candidates),
                    }
                if len(frame_hits) > 1:
                    frame_hits.sort(key=lambda c: abs(c["timelineStart"] - frame))
                    return {
                        "selector": selector,
                        "matchedBy": "producer+frame_nearest",
                        "clip": frame_hits[0],
                        "candidates": len(candidates),
                    }

            if len(candidates) == 1:
                return {"selector": selector, "matchedBy": "producerId", "clip": candidates[0], "candidates": 1}
            if len(candidates) > 1:
                raise BridgeOperationError(
                    "INVALID_INPUT",
                    f"Clip selector '{selector}' is ambiguous ({len(candidates)} matches); provide at_frame or clipRef",
                )

            raise BridgeOperationError("INVALID_INPUT", f"Clip '{selector}' not found")
        if method == "timeline.time_remap":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            clip_ref = str(params["clip_ref"])
            speed = float(params.get("speed", 1.0))
            if speed <= 0:
                raise BridgeOperationError("INVALID_INPUT", "speed must be > 0")
            entry, _ = _resolve_clip_element(loaded, clip_ref)
            start = int(entry.get("in", "0"))
            end = int(entry.get("out", str(start)))
            original_duration = max(1, end - start + 1)
            new_duration = max(1, int(round(original_duration / speed)))
            new_out = start + new_duration - 1
            changed = new_out != end
            entry.set("out", str(new_out))
            prop = entry.find('./property[@name="harness:time-remap"]')
            if prop is None:
                prop = etree.SubElement(entry, "property", name="harness:time-remap")
            prop.text = json.dumps({"speed": speed})
            _recalculate_timeline_bounds(loaded)
            saved = _save(loaded, params.get("output"))
            return _mutation_payload(
                {
                    "clipRef": clip_ref,
                    "speed": speed,
                    "oldDurationFrames": original_duration,
                    "newDurationFrames": new_duration,
                    "savedTo": saved,
                },
                changed=changed,
                idempotent=not changed,
            )
        if method == "timeline.transform":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            clip_ref = str(params["clip_ref"])
            entry, _ = _resolve_clip_element(loaded, clip_ref)
            effect_id = str(params.get("effect_id", "transform"))
            filt = entry.find(f'./filter[@id="{effect_id}"]')
            if filt is None:
                filt = etree.SubElement(entry, "filter", id=effect_id)
                service = etree.SubElement(filt, "property", name="mlt_service")
                service.text = "affine"
            changed = False
            for key, value in {
                "geometry": params.get("geometry"),
                "rotate": params.get("rotate"),
                "scale": params.get("scale"),
                "opacity": params.get("opacity"),
                "harness:keyframes": params.get("keyframes"),
            }.items():
                if value is None:
                    continue
                text_val = json.dumps(value) if key == "harness:keyframes" else str(value)
                prop = filt.find(f'./property[@name="{key}"]')
                if prop is None:
                    prop = etree.SubElement(filt, "property", name=key)
                    prop.text = text_val
                    changed = True
                elif (prop.text or "") != text_val:
                    prop.text = text_val
                    changed = True
            saved = _save(loaded, params.get("output"))
            return _mutation_payload(
                {"clipRef": clip_ref, "effectId": effect_id, "savedTo": saved},
                changed=changed,
                idempotent=not changed,
            )
        if method == "timeline.nudge_clip":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            clip_ref = str(params["clip_ref"])
            delta = int(params.get("delta_frames", 0))
            if delta == 0:
                saved = _save(loaded, params.get("output"))
                return _mutation_payload(
                    {"clipRef": clip_ref, "deltaFrames": 0, "savedTo": saved},
                    changed=False,
                    idempotent=True,
                )
            timeline = TimelineAPI(loaded)
            clip = timeline._resolve_clip(clip_ref)
            if clip is None:
                raise BridgeOperationError("INVALID_INPUT", f"Clip not found: {clip_ref}")
            new_position = max(0, clip.timeline_start + delta)
            timeline.move_clip(clip_ref=clip_ref, new_track=clip.track_id, new_position=new_position)
            _recalculate_timeline_bounds(loaded)
            saved = _save(loaded, params.get("output"))
            return _mutation_payload({"clipRef": clip_ref, "newPosition": new_position, "savedTo": saved})
        if method == "timeline.slip_clip":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            clip_ref = str(params["clip_ref"])
            delta = int(params.get("delta_frames", 0))
            entry, _, _ = _clip_context(loaded, clip_ref)
            in_point = int(entry.get("in", "0"))
            out_point = int(entry.get("out", str(in_point)))
            duration = max(1, out_point - in_point + 1)
            new_in = max(0, in_point + delta)
            new_out = new_in + duration - 1
            changed = new_in != in_point
            entry.set("in", str(new_in))
            entry.set("out", str(new_out))
            saved = _save(loaded, params.get("output"))
            return _mutation_payload(
                {"clipRef": clip_ref, "newIn": new_in, "newOut": new_out, "savedTo": saved},
                changed=changed,
                idempotent=not changed,
            )
        if method == "timeline.slide_clip":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            clip_ref = str(params["clip_ref"])
            delta = int(params.get("delta_frames", 0))
            timeline = TimelineAPI(loaded)
            clip = timeline._resolve_clip(clip_ref)
            if clip is None:
                raise BridgeOperationError("INVALID_INPUT", f"Clip not found: {clip_ref}")
            new_position = max(0, clip.timeline_start + delta)
            timeline.move_clip(clip_ref=clip_ref, new_track=clip.track_id, new_position=new_position)
            _recalculate_timeline_bounds(loaded)
            saved = _save(loaded, params.get("output"))
            return _mutation_payload({"clipRef": clip_ref, "newPosition": new_position, "savedTo": saved})
        if method == "timeline.ripple_insert":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            track_id = str(params["track_id"])
            position = int(params["position"])
            length = int(params.get("length", 1))
            if length <= 0:
                raise BridgeOperationError("INVALID_INPUT", "length must be > 0")
            timeline = TimelineAPI(loaded)
            timeline.insert_gap(track_id=track_id, position=position, length=length)
            inserted_clip_ref = None
            clip_id = params.get("clip_id")
            if clip_id:
                out_point = params.get("out_point")
                in_point = str(params.get("in_point", "0"))
                inserted_clip_ref = timeline.add_clip(
                    clip_id=str(clip_id),
                    track_id=track_id,
                    position=position,
                    in_point=in_point,
                    out_point=str(out_point) if out_point is not None else None,
                )
            _recalculate_timeline_bounds(loaded)
            saved = _save(loaded, params.get("output"))
            return _mutation_payload(
                {"trackId": track_id, "position": position, "length": length, "clipRef": inserted_clip_ref, "savedTo": saved}
            )
        if method == "timeline.group_clips":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            clip_refs = [str(x) for x in params.get("clip_refs", [])]
            if not clip_refs:
                raise BridgeOperationError("INVALID_INPUT", "clip_refs is required")
            group_id = str(params.get("group_id") or f"group_{uuid4().hex[:8]}")
            changed = False
            for clip_ref in clip_refs:
                entry, _, _ = _clip_context(loaded, clip_ref)
                prop = _get_or_create_property(entry, "harness:group-id")
                if (prop.text or "") != group_id:
                    prop.text = group_id
                    changed = True
            saved = _save(loaded, params.get("output"))
            return _mutation_payload(
                {"groupId": group_id, "clipRefs": clip_refs, "savedTo": saved},
                changed=changed,
                idempotent=not changed,
            )
        if method == "timeline.ungroup_clips":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            clip_refs = [str(x) for x in params.get("clip_refs", [])]
            if not clip_refs:
                raise BridgeOperationError("INVALID_INPUT", "clip_refs is required")
            changed = False
            for clip_ref in clip_refs:
                entry, _, _ = _clip_context(loaded, clip_ref)
                prop = entry.find('./property[@name="harness:group-id"]')
                if prop is not None:
                    entry.remove(prop)
                    changed = True
            saved = _save(loaded, params.get("output"))
            return _mutation_payload(
                {"clipRefs": clip_refs, "savedTo": saved},
                changed=changed,
                idempotent=not changed,
            )
        if method == "sequence.list":
            loaded = _load(params["project"])
            sequences = []
            for tractor in loaded.root.findall(".//tractor"):
                is_sequence = tractor.get("id", "").startswith("timeline_sequence_") or tractor.find(
                    './/property[@name="kdenlive:sequenceproperties.documentuuid"]'
                ) is not None
                if is_sequence:
                    sequences.append({"id": tractor.get("id"), "in": tractor.get("in"), "out": tractor.get("out")})
            active = loaded.get_property("harness:active-sequence")
            return {"sequences": sequences, "activeSequence": active}
        if method == "sequence.copy":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            source_id = str(params["source_id"])
            source = loaded.root.find(f'.//tractor[@id="{source_id}"]')
            if source is None:
                raise BridgeOperationError("INVALID_INPUT", f"Sequence not found: {source_id}")
            new_id = str(params.get("new_id") or f"{source_id}_copy_{uuid4().hex[:6]}")
            if loaded.root.find(f'.//tractor[@id="{new_id}"]') is not None:
                raise BridgeOperationError("INVALID_INPUT", f"Sequence already exists: {new_id}")
            cloned = etree.fromstring(etree.tostring(source))
            cloned.set("id", new_id)
            loaded.root.append(cloned)
            saved = _save(loaded, params.get("output"))
            return _mutation_payload({"sourceId": source_id, "newId": new_id, "savedTo": saved})
        if method == "sequence.set_active":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            sequence_id = str(params["sequence_id"])
            if loaded.root.find(f'.//tractor[@id="{sequence_id}"]') is None:
                raise BridgeOperationError("INVALID_INPUT", f"Sequence not found: {sequence_id}")
            prop = _get_or_create_property(loaded.root, "harness:active-sequence")
            changed = (prop.text or "") != sequence_id
            prop.text = sequence_id
            saved = _save(loaded, params.get("output"))
            return _mutation_payload(
                {"sequenceId": sequence_id, "savedTo": saved},
                changed=changed,
                idempotent=not changed,
            )
        if method == "audio.add_music":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            media = Path(params["media"])
            if not media.exists():
                raise BridgeOperationError("NOT_FOUND", f"Media file not found: {media}")
            track_id = str(params.get("track_id", "playlist1"))
            producer_id = str(params.get("producer_id") or f"music_{uuid4().hex[:8]}")
            if not loaded.get_producers(id_filter=producer_id):
                execute(
                    "asset.import",
                    {
                        "project": str(loaded.project_path),
                        "media": str(media),
                        "producer_id": producer_id,
                        "output": None,
                    },
                )
                loaded = _load(str(loaded.project_path))
            position = int(params.get("position", 0))
            duration_frames = params.get("duration_frames")
            out_point = str(int(duration_frames) - 1) if duration_frames is not None else None
            clip_ref = TimelineAPI(loaded).add_clip(
                clip_id=producer_id,
                track_id=track_id,
                position=position,
                in_point="0",
                out_point=out_point,
            )
            _recalculate_timeline_bounds(loaded)
            saved = _save(loaded, params.get("output"))
            return _mutation_payload(
                {"clipRef": clip_ref, "producerId": producer_id, "trackId": track_id, "savedTo": saved}
            )
        if method == "audio.duck":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            track_id = str(params["track_id"])
            gain = float(params.get("duck_gain", 0.3))
            if gain <= 0:
                raise BridgeOperationError("INVALID_INPUT", "duck_gain must be > 0")
            clips = TimelineAPI(loaded).get_clips(track_id=track_id)
            changed = False
            for clip in clips:
                entry = clip.element
                if entry is None:
                    continue
                filt = entry.find('./filter[@id="audio_duck"]')
                if filt is None:
                    filt = etree.SubElement(entry, "filter", id="audio_duck")
                    svc = etree.SubElement(filt, "property", name="mlt_service")
                    svc.text = "volume"
                    changed = True
                prop = filt.find('./property[@name="gain"]')
                gain_text = str(gain)
                if prop is None:
                    prop = etree.SubElement(filt, "property", name="gain")
                    prop.text = gain_text
                    changed = True
                elif (prop.text or "") != gain_text:
                    prop.text = gain_text
                    changed = True
            saved = _save(loaded, params.get("output"))
            return _mutation_payload(
                {"trackId": track_id, "duckGain": gain, "clipsAffected": len(clips), "savedTo": saved},
                changed=changed,
                idempotent=not changed,
            )
        if method == "audio.fade":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            clip_ref = str(params["clip_ref"])
            fade_type = str(params.get("fade_type", "in")).lower()
            frames = int(params.get("frames", 24))
            if fade_type not in {"in", "out"}:
                raise BridgeOperationError("INVALID_INPUT", "fade_type must be 'in' or 'out'")
            if frames <= 0:
                raise BridgeOperationError("INVALID_INPUT", "frames must be > 0")
            entry, _ = _resolve_clip_element(loaded, clip_ref)
            filt = entry.find(f'./filter[@id="audio_fade_{fade_type}"]')
            if filt is None:
                filt = etree.SubElement(entry, "filter", id=f"audio_fade_{fade_type}")
                svc = etree.SubElement(filt, "property", name="mlt_service")
                svc.text = "volume"
            shape = {"type": fade_type, "frames": frames}
            prop = filt.find('./property[@name="harness:fade"]')
            changed = False
            shape_text = json.dumps(shape, separators=(",", ":"))
            if prop is None:
                prop = etree.SubElement(filt, "property", name="harness:fade")
                prop.text = shape_text
                changed = True
            elif (prop.text or "") != shape_text:
                prop.text = shape_text
                changed = True
            saved = _save(loaded, params.get("output"))
            return _mutation_payload(
                {"clipRef": clip_ref, "fadeType": fade_type, "frames": frames, "savedTo": saved},
                changed=changed,
                idempotent=not changed,
            )
        if method == "audio.normalize":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            clip_ref = str(params["clip_ref"])
            target_db = float(params.get("target_db", -14.0))
            entry, _ = _resolve_clip_element(loaded, clip_ref)
            filt = entry.find('./filter[@id="audio_normalize"]')
            if filt is None:
                filt = etree.SubElement(entry, "filter", id="audio_normalize")
                svc = etree.SubElement(filt, "property", name="mlt_service")
                svc.text = "volume"
            prop = _get_or_create_property(filt, "target_db")
            changed = (prop.text or "") != str(target_db)
            prop.text = str(target_db)
            saved = _save(loaded, params.get("output"))
            return _mutation_payload(
                {"clipRef": clip_ref, "targetDb": target_db, "savedTo": saved},
                changed=changed,
                idempotent=not changed,
            )
        if method == "audio.remove_silence":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            clip_ref = str(params["clip_ref"])
            threshold_db = float(params.get("threshold_db", -35.0))
            min_duration = int(params.get("min_duration_frames", 6))
            if min_duration <= 0:
                raise BridgeOperationError("INVALID_INPUT", "min_duration_frames must be > 0")
            entry, _ = _resolve_clip_element(loaded, clip_ref)
            filt = entry.find('./filter[@id="audio_remove_silence"]')
            if filt is None:
                filt = etree.SubElement(entry, "filter", id="audio_remove_silence")
                svc = etree.SubElement(filt, "property", name="mlt_service")
                svc.text = "gate"
            p1 = _get_or_create_property(filt, "threshold_db")
            p2 = _get_or_create_property(filt, "min_duration_frames")
            changed = False
            if (p1.text or "") != str(threshold_db):
                p1.text = str(threshold_db)
                changed = True
            if (p2.text or "") != str(min_duration):
                p2.text = str(min_duration)
                changed = True
            saved = _save(loaded, params.get("output"))
            return _mutation_payload(
                {"clipRef": clip_ref, "thresholdDb": threshold_db, "minDurationFrames": min_duration, "savedTo": saved},
                changed=changed,
                idempotent=not changed,
            )
        if method == "audio.pan":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            clip_ref = str(params["clip_ref"])
            pan = float(params.get("pan", 0.0))
            if pan < -1.0 or pan > 1.0:
                raise BridgeOperationError("INVALID_INPUT", "pan must be between -1.0 and 1.0")
            entry, _ = _resolve_clip_element(loaded, clip_ref)
            filt = entry.find('./filter[@id="audio_pan"]')
            if filt is None:
                filt = etree.SubElement(entry, "filter", id="audio_pan")
                svc = etree.SubElement(filt, "property", name="mlt_service")
                svc.text = "panner"
            p = _get_or_create_property(filt, "start")
            changed = (p.text or "") != str(pan)
            p.text = str(pan)
            saved = _save(loaded, params.get("output"))
            return _mutation_payload(
                {"clipRef": clip_ref, "pan": pan, "savedTo": saved},
                changed=changed,
                idempotent=not changed,
            )
        if method == "color.grade":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            clip_ref = str(params["clip_ref"])
            entry, _ = _resolve_clip_element(loaded, clip_ref)
            effect_id = str(params.get("effect_id", "color_grade"))
            filt = entry.find(f'./filter[@id="{effect_id}"]')
            if filt is None:
                filt = etree.SubElement(entry, "filter", id=effect_id)
                service = etree.SubElement(filt, "property", name="mlt_service")
                service.text = "movit.lift_gamma_gain"
            changed = False
            values = {
                "lift": params.get("lift"),
                "gamma": params.get("gamma"),
                "gain": params.get("gain"),
                "saturation": params.get("saturation"),
                "temperature": params.get("temperature"),
                "lut_path": params.get("lut_path"),
            }
            for key, value in values.items():
                if value is None:
                    continue
                text_val = str(value)
                prop = filt.find(f'./property[@name="{key}"]')
                if prop is None:
                    prop = etree.SubElement(filt, "property", name=key)
                    prop.text = text_val
                    changed = True
                elif (prop.text or "") != text_val:
                    prop.text = text_val
                    changed = True
            saved = _save(loaded, params.get("output"))
            return _mutation_payload(
                {"clipRef": clip_ref, "effectId": effect_id, "savedTo": saved},
                changed=changed,
                idempotent=not changed,
            )
        if method == "track.add":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            tractor = _get_project_tractor(loaded)
            track_type = str(params.get("track_type", "video")).lower()
            if track_type not in {"video", "audio"}:
                raise BridgeOperationError("INVALID_INPUT", "track_type must be 'video' or 'audio'")
            playlist_id = str(params.get("track_id") or _next_playlist_id(loaded))
            existing_playlist = loaded.root.find(f'.//playlist[@id="{playlist_id}"]')
            if existing_playlist is not None:
                saved = _save(loaded, params.get("output"))
                return _mutation_payload(
                    {"trackId": playlist_id, "savedTo": saved},
                    changed=False,
                    idempotent=True,
                )
            playlist = etree.SubElement(loaded.root, "playlist", id=playlist_id)
            name_prop = etree.SubElement(playlist, "property", name="kdenlive:track_name")
            name_prop.text = str(params.get("name", playlist_id))
            hide = "video" if track_type == "audio" else "audio"
            track = etree.Element("track", producer=playlist_id, hide=hide)
            index_raw = params.get("index")
            tracks = tractor.findall("track")
            if index_raw is None:
                tractor.append(track)
                index = len(tracks)
            else:
                index = int(index_raw)
                if index < 0 or index > len(tracks):
                    raise BridgeOperationError("INVALID_INPUT", f"index must be between 0 and {len(tracks)}")
                tractor.insert(index, track)
            saved = _save(loaded, params.get("output"))
            return _mutation_payload({"trackId": playlist_id, "index": index, "savedTo": saved})
        if method == "track.remove":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            track_id = str(params["track_id"])
            tractor = _get_project_tractor(loaded)
            playlist = loaded.root.find(f'.//playlist[@id="{track_id}"]')
            if playlist is None:
                saved = _save(loaded, params.get("output"))
                return _mutation_payload(
                    {"trackId": track_id, "savedTo": saved},
                    changed=False,
                    idempotent=True,
                )
            has_entries = any(node.tag == "entry" for node in list(playlist))
            if has_entries and not bool(params.get("force", False)):
                raise BridgeOperationError(
                    "INVALID_INPUT", f"Track '{track_id}' is not empty; set force=true to remove"
                )
            for track in tractor.findall("track"):
                if track.get("producer") == track_id:
                    tractor.remove(track)
                    break
            loaded.root.remove(playlist)
            saved = _save(loaded, params.get("output"))
            return _mutation_payload({"trackId": track_id, "removed": True, "savedTo": saved})
        if method == "track.reorder":
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            track_id = str(params["track_id"])
            new_index = int(params["index"])
            tractor = _get_project_tractor(loaded)
            tracks = tractor.findall("track")
            current = next((t for t in tracks if t.get("producer") == track_id), None)
            if current is None:
                raise BridgeOperationError("INVALID_INPUT", f"Track '{track_id}' not found")
            if new_index < 0 or new_index >= len(tracks):
                raise BridgeOperationError("INVALID_INPUT", f"index must be between 0 and {len(tracks)-1}")
            current_index = tracks.index(current)
            if current_index == new_index:
                saved = _save(loaded, params.get("output"))
                return _mutation_payload(
                    {"trackId": track_id, "index": new_index, "savedTo": saved},
                    changed=False,
                    idempotent=True,
                )
            tractor.remove(current)
            tractor.insert(new_index, current)
            saved = _save(loaded, params.get("output"))
            return _mutation_payload({"trackId": track_id, "index": new_index, "savedTo": saved})
        if method == "track.resolve":
            loaded = _load(params["project"])
            selector = str(params["selector"])
            rows = _track_rows(loaded)
            if not rows:
                raise BridgeOperationError("INVALID_INPUT", "Project has no timeline tracks")

            by_id = [row for row in rows if row["trackId"] == selector]
            if len(by_id) == 1:
                return {"selector": selector, "matchedBy": "id", "track": by_id[0], "tracks": rows}

            by_name = [row for row in rows if row["name"] == selector]
            if len(by_name) == 1:
                return {"selector": selector, "matchedBy": "name", "track": by_name[0], "tracks": rows}
            if len(by_name) > 1:
                raise BridgeOperationError(
                    "INVALID_INPUT",
                    f"Track selector '{selector}' is ambiguous by name; use trackId",
                )

            selector_lower = selector.lower()
            by_name_ci = [row for row in rows if str(row["name"]).lower() == selector_lower]
            if len(by_name_ci) == 1:
                return {"selector": selector, "matchedBy": "name_ci", "track": by_name_ci[0], "tracks": rows}
            if len(by_name_ci) > 1:
                raise BridgeOperationError(
                    "INVALID_INPUT",
                    f"Track selector '{selector}' is ambiguous (case-insensitive); use trackId",
                )

            available = [row["trackId"] for row in rows]
            raise BridgeOperationError(
                "INVALID_INPUT",
                f"Track '{selector}' not found. Available trackIds: {', '.join(available)}",
            )
        if method in {"track.mute", "track.unmute", "track.lock", "track.unlock", "track.show", "track.hide"}:
            loaded = _load(params["project"])
            _validate_project_for_edit(loaded)
            track_id = str(params["track_id"])
            playlist = loaded.root.find(f'.//playlist[@id="{track_id}"]')
            if playlist is None:
                raise BridgeOperationError("INVALID_INPUT", f"Track '{track_id}' not found")
            state_map = {
                "track.mute": ("harness:muted", "1"),
                "track.unmute": ("harness:muted", "0"),
                "track.lock": ("harness:locked", "1"),
                "track.unlock": ("harness:locked", "0"),
                "track.show": ("harness:hidden", "0"),
                "track.hide": ("harness:hidden", "1"),
            }
            prop_name, value = state_map[method]
            prop = _get_or_create_property(playlist, prop_name)
            changed = (prop.text or "") != value
            prop.text = value
            saved = _save(loaded, params.get("output"))
            return _mutation_payload(
                {"trackId": track_id, "property": prop_name, "value": value, "savedTo": saved},
                changed=changed,
                idempotent=not changed,
            )
        if method == "render.clip":
            rendered_params = _apply_render_preset(params)
            job_id = f"job_{uuid4().hex[:12]}"
            start = datetime.utcnow().isoformat() + "Z"
            RENDER_JOBS[job_id] = {
                "status": "running",
                "startedAt": start,
                "type": "clip",
                "request": {"method": "render.clip", "params": dict(params)},
            }
            try:
                data = _render_clip(rendered_params)
                RENDER_JOBS[job_id].update({"status": "completed", "endedAt": datetime.utcnow().isoformat() + "Z"})
                data["jobId"] = job_id
                data["status"] = "completed"
                return data
            except Exception as exc:
                RENDER_JOBS[job_id].update(
                    {"status": "failed", "error": str(exc), "endedAt": datetime.utcnow().isoformat() + "Z"}
                )
                raise
        if method == "render.project":
            source = Path(params["project"])
            if not source.exists():
                raise BridgeOperationError("NOT_FOUND", f"Project file not found: {source}")
            loaded = _load(str(source))
            _recalculate_timeline_bounds(loaded)
            loaded.save(source)
            output = Path(params["output"])
            melt = _resolve_bin("melt")
            rendered_params = _apply_render_preset(params)
            cmd_source = source
            cmd = [str(melt)]
            start_seconds = rendered_params.get("start_seconds")
            duration_seconds = rendered_params.get("duration_seconds")
            zone_in = rendered_params.get("zone_in")
            zone_out = rendered_params.get("zone_out")
            render_in: Optional[int] = None
            render_out: Optional[int] = None
            if zone_in is not None or zone_out is not None:
                z_in = int(zone_in or 0)
                z_out = int(zone_out if zone_out is not None else z_in)
                if z_in < 0 or z_out < z_in:
                    raise BridgeOperationError("INVALID_INPUT", "invalid zone_in/zone_out")
                render_in, render_out = z_in, z_out
            else:
                start = float(start_seconds) if start_seconds is not None else 0.0
                if start < 0:
                    raise BridgeOperationError("INVALID_INPUT", "start_seconds must be >= 0")
                in_frame = int(round(start * 30))
                render_in = in_frame
                if duration_seconds is not None:
                    duration = float(duration_seconds)
                    if duration <= 0:
                        raise BridgeOperationError("INVALID_INPUT", "duration_seconds must be > 0")
                    out_frame = in_frame + int(round(duration * 30)) - 1
                    render_out = out_frame
            tmp_project_path: Optional[Path] = None
            if render_in is not None and render_out is not None:
                tmp_project_path = Path(tempfile.mkdtemp(prefix="harness_kdenlive_render_")) / "render_bounds.kdenlive"
                bounded = _load(str(source))
                _set_project_bounds(bounded, render_in, render_out)
                bounded.save(tmp_project_path)
                cmd_source = tmp_project_path

            # Fallback path for harness-generated text overlays on MLT builds
            # where qtext/subtitle producers are missing or unstable.
            text_map, cues = _collect_text_overlay_cues(_load(str(cmd_source)))
            cue_count = 0
            fallback_dir: Optional[Path] = None
            render_target = output
            if cues and text_map:
                fallback_dir = Path(tempfile.mkdtemp(prefix="harness_kdenlive_subtitles_"))
                base_project = _load(str(cmd_source))
                _remove_text_overlays(base_project, set(text_map.keys()))
                _recalculate_timeline_bounds(base_project)
                base_project_path = fallback_dir / "base_no_text.kdenlive"
                base_project.save(base_project_path)
                cmd_source = base_project_path
                render_target = fallback_dir / "base_render.mp4"
                srt_path = fallback_dir / "overlay.srt"
                cue_count = _write_cues_srt(
                    srt_path,
                    cues,
                    fps=_project_fps(loaded),
                    frame_offset=int(render_in or 0),
                )
            cmd.append(str(cmd_source))
            cmd.extend(
                [
                    "-consumer",
                    f"avformat:{render_target}",
                    f"vcodec={rendered_params.get('vcodec', 'libx264')}",
                    f"acodec={rendered_params.get('acodec', 'aac')}",
                    f"ab={rendered_params.get('audio_bitrate', '192k')}",
                    f"crf={rendered_params.get('crf', '18')}",
                    f"preset={rendered_params.get('preset', 'fast')}",
                ]
            )
            job_id = f"job_{uuid4().hex[:12]}"
            RENDER_JOBS[job_id] = {
                "status": "running",
                "startedAt": datetime.utcnow().isoformat() + "Z",
                "type": "project",
                "request": {"method": "render.project", "params": dict(params)},
            }
            try:
                rendered_duration = _render_and_probe_duration(cmd, render_target)
            except Exception as exc:
                RENDER_JOBS[job_id].update(
                    {"status": "failed", "error": str(exc), "endedAt": datetime.utcnow().isoformat() + "Z"}
                )
                if tmp_project_path is not None:
                    shutil.rmtree(tmp_project_path.parent, ignore_errors=True)
                if fallback_dir is not None:
                    shutil.rmtree(fallback_dir, ignore_errors=True)
                raise
            if fallback_dir is not None and cue_count > 0:
                ffmpeg = _resolve_bin("ffmpeg")
                srt_path = fallback_dir / "overlay.srt"
                burn_cmd = [
                    str(ffmpeg),
                    "-y",
                    "-i",
                    "base_render.mp4",
                    "-vf",
                    "subtitles=overlay.srt",
                    "-c:v",
                    str(rendered_params.get("vcodec", "libx264")),
                    "-preset",
                    str(rendered_params.get("preset", "fast")),
                    "-crf",
                    str(rendered_params.get("crf", "18")),
                    "-c:a",
                    "copy",
                    str(output),
                ]
                _render_and_probe_duration(burn_cmd, output, cwd=fallback_dir)
                rendered_duration = _probe_media_duration_seconds(output) or rendered_duration
                shutil.rmtree(fallback_dir, ignore_errors=True)
            RENDER_JOBS[job_id].update({"status": "completed", "endedAt": datetime.utcnow().isoformat() + "Z"})
            if tmp_project_path is not None:
                shutil.rmtree(tmp_project_path.parent, ignore_errors=True)
            return {
                "project": str(source),
                "output": str(output),
                "durationSeconds": rendered_duration,
                "jobId": job_id,
                "status": "completed",
            }
        if method == "render.status":
            job_id = str(params["job_id"])
            status = RENDER_JOBS.get(job_id)
            if status is None:
                raise BridgeOperationError("NOT_FOUND", f"Render job not found: {job_id}")
            return {"jobId": job_id, **status}
        if method == "render.latest":
            job_type = params.get("type")
            status_filter = params.get("status")
            jobs = [{"jobId": jid, **info} for jid, info in RENDER_JOBS.items()]
            if job_type:
                jobs = [j for j in jobs if j.get("type") == str(job_type)]
            if status_filter:
                jobs = [j for j in jobs if j.get("status") == str(status_filter)]
            if not jobs:
                raise BridgeOperationError("NOT_FOUND", "No render jobs found matching filter")
            jobs.sort(key=lambda x: x.get("startedAt", ""), reverse=True)
            return jobs[0]
        if method == "render.retry":
            job_id = str(params["job_id"])
            previous = RENDER_JOBS.get(job_id)
            if previous is None:
                raise BridgeOperationError("NOT_FOUND", f"Render job not found: {job_id}")
            request = previous.get("request")
            if not isinstance(request, dict):
                raise BridgeOperationError("INVALID_INPUT", f"Render job '{job_id}' has no retryable request")
            retry_method = str(request.get("method", ""))
            retry_params = dict(request.get("params", {}))
            if retry_method not in {"render.clip", "render.project"}:
                raise BridgeOperationError("INVALID_INPUT", f"Render job '{job_id}' is not retryable")
            output_override = params.get("output")
            if output_override:
                retry_params["output"] = str(output_override)
            return execute(retry_method, retry_params)
        if method == "render.cancel":
            job_id = str(params["job_id"])
            status = RENDER_JOBS.get(job_id)
            if status is None:
                raise BridgeOperationError("NOT_FOUND", f"Render job not found: {job_id}")
            if status.get("status") == "running":
                status["status"] = "canceled"
                status["endedAt"] = datetime.utcnow().isoformat() + "Z"
                return _mutation_payload({"jobId": job_id, "status": "canceled"}, changed=True)
            return _mutation_payload({"jobId": job_id, "status": status.get("status")}, changed=False, idempotent=True)
        if method == "render.list_jobs":
            jobs = [{"jobId": jid, **info} for jid, info in RENDER_JOBS.items()]
            jobs.sort(key=lambda x: x.get("startedAt", ""), reverse=True)
            return {"jobs": jobs}
        if method == "render.wait":
            job_id = str(params["job_id"])
            timeout_seconds = float(params.get("timeout_seconds", 120))
            poll_interval = float(params.get("poll_interval_seconds", 0.2))
            deadline = time.perf_counter() + timeout_seconds
            while time.perf_counter() <= deadline:
                status = RENDER_JOBS.get(job_id)
                if status is None:
                    raise BridgeOperationError("NOT_FOUND", f"Render job not found: {job_id}")
                if status.get("status") in {"completed", "failed", "canceled"}:
                    return {"jobId": job_id, **status}
                time.sleep(max(0.05, poll_interval))
            raise BridgeOperationError("ERROR", f"Timed out waiting for render job: {job_id}")
        if method in {"export.edl", "export.xml", "export.otio"}:
            loaded = _load(params["project"])
            output = Path(params["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            timeline = loaded.get_clips_on_timeline()
            if method == "export.edl":
                lines = ["TITLE: harness-kdenlive export", "FCM: NON-DROP FRAME"]
                for idx, clip in enumerate(timeline, start=1):
                    lines.append(
                        f"{idx:03d}  AX       V     C        {clip.timeline_start:08d} {clip.timeline_end:08d} {clip.timeline_start:08d} {clip.timeline_end:08d}"
                    )
                output.write_text("\n".join(lines) + "\n", encoding="utf-8")
            elif method == "export.xml":
                root = etree.Element("harness_export", type="xml", project=str(loaded.project_path))
                for clip in timeline:
                    etree.SubElement(
                        root,
                        "clip",
                        ref=clip.instance_id,
                        producer=clip.producer_id,
                        track=clip.track_id,
                        start=str(clip.timeline_start),
                        end=str(clip.timeline_end),
                    )
                etree.ElementTree(root).write(str(output), encoding="utf-8", xml_declaration=True, pretty_print=True)
            else:
                otio = {
                    "OTIO_SCHEMA": "Timeline.1",
                    "name": loaded.project_path.stem,
                    "tracks": [
                        {
                            "OTIO_SCHEMA": "Track.1",
                            "children": [
                                {
                                    "OTIO_SCHEMA": "Clip.1",
                                    "name": c.instance_id,
                                    "metadata": {
                                        "producer_id": c.producer_id,
                                        "track_id": c.track_id,
                                        "start": c.timeline_start,
                                        "end": c.timeline_end,
                                    },
                                }
                                for c in timeline
                            ],
                        }
                    ],
                }
                output.write_text(json.dumps(otio, indent=2), encoding="utf-8")
            return _mutation_payload({"project": str(loaded.project_path), "output": str(output), "format": method.split(".")[1]})
        if method == "batch.execute":
            action = params.get("action")
            steps = params.get("steps")
            if action:
                steps = [{"method": action, "params": params.get("params", {})}]
            if not isinstance(steps, list) or not steps:
                raise BridgeOperationError("INVALID_INPUT", "steps must be a non-empty list")
            stop_on_error = bool(params.get("stop_on_error", True))
            results: List[Dict[str, Any]] = []
            for index, step in enumerate(steps):
                if not isinstance(step, dict) or "method" not in step:
                    raise BridgeOperationError("INVALID_INPUT", f"Invalid step at index {index}")
                step_method = str(step["method"])
                step_params = dict(step.get("params", {}))
                try:
                    step_result = execute(step_method, step_params)
                    results.append({"index": index, "method": step_method, "ok": True, "result": step_result})
                except BridgeOperationError as exc:
                    error = {"code": exc.code, "message": exc.message}
                    results.append({"index": index, "method": step_method, "ok": False, "error": error})
                    if stop_on_error:
                        return {
                            "ok": False,
                            "stopOnError": stop_on_error,
                            "completedSteps": index,
                            "totalSteps": len(steps),
                            "results": results,
                        }
            return {
                "ok": all(r.get("ok") for r in results),
                "stopOnError": stop_on_error,
                "completedSteps": len(results),
                "totalSteps": len(steps),
                "results": results,
            }
        raise BridgeOperationError("INVALID_INPUT", f"Unknown method: {method}")
    except ValueError as exc:
        raise BridgeOperationError("INVALID_INPUT", str(exc)) from exc
