import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from harness_kdenlive import __version__
from harness_kdenlive.bridge.client import BridgeClient, BridgeClientError
from harness_kdenlive.bridge.protocol import ERROR_CODES, PROTOCOL_VERSION
from harness_kdenlive.bridge.server import run_bridge_server

app = typer.Typer(add_completion=False, help="Bridge-first CLI for Kdenlive editing")
bridge_app = typer.Typer(add_completion=False, help="Bridge lifecycle and verification")
app.add_typer(bridge_app, name="bridge")


def _print(payload: Dict[str, Any]) -> None:
    typer.echo(json.dumps(payload, indent=2))


def _ok(command: str, data: Dict[str, Any]) -> None:
    _print({"ok": True, "protocolVersion": PROTOCOL_VERSION, "command": command, "data": data})


def _fail(command: str, code: str, message: str, retryable: bool = False) -> None:
    _print(
        {
            "ok": False,
            "protocolVersion": PROTOCOL_VERSION,
            "command": command,
            "error": {"code": code, "message": message, "retryable": retryable},
        }
    )
    raise SystemExit(ERROR_CODES.get(code, 1))


def _bridge_client() -> BridgeClient:
    return BridgeClient()


def _call_bridge(
    command: str,
    method: str,
    params: Dict[str, Any],
    timeout_seconds: float = 30,
) -> Dict[str, Any]:
    client = _bridge_client()
    try:
        return client.call(method, params, timeout_seconds=timeout_seconds)
    except BridgeClientError as exc:
        _fail(command, exc.code, exc.message, retryable=exc.code == "BRIDGE_UNAVAILABLE")
    except Exception as exc:
        _fail(command, "ERROR", str(exc))
    raise RuntimeError("unreachable")


def _ensure_bridge_ready(command: str) -> None:
    _call_bridge(command, "system.health", {}, timeout_seconds=5)


def _json_arg(command: str, raw: str) -> Dict[str, Any]:
    try:
        val = json.loads(raw)
    except json.JSONDecodeError as exc:
        _fail(command, "INVALID_INPUT", f"Invalid JSON: {exc}")
    if not isinstance(val, dict):
        _fail(command, "INVALID_INPUT", "JSON value must be an object")
    return val


def _bridge_state_dir() -> Path:
    root = Path(os.getenv("LOCALAPPDATA", Path.home()))
    state_dir = root / "harness-kdenlive"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def _bridge_pid_file() -> Path:
    return _bridge_state_dir() / "bridge.pid"


@bridge_app.command("serve")
def bridge_serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(41739, "--port"),
) -> None:
    run_bridge_server(host, port)


@bridge_app.command("start")
def bridge_start(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(41739, "--port"),
) -> None:
    pid_file = _bridge_pid_file()
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
            os.kill(pid, 0)
            _ok("bridge.start", {"status": "already-running", "pid": pid, "host": host, "port": port})
            return
        except Exception:
            pid_file.unlink(missing_ok=True)

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    process = subprocess.Popen(
        [sys.executable, "-m", "harness_kdenlive", "bridge", "serve", "--host", host, "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    pid_file.write_text(str(process.pid), encoding="utf-8")
    os.environ["HARNESS_KDENLIVE_BRIDGE_URL"] = f"http://{host}:{port}"

    for _ in range(30):
        time.sleep(0.1)
        try:
            status = BridgeClient(f"http://{host}:{port}").health()
            if status.get("ok"):
                _ok("bridge.start", {"status": "started", "pid": process.pid, "host": host, "port": port})
                return
        except BridgeClientError:
            continue
    _fail("bridge.start", "BRIDGE_UNAVAILABLE", "Bridge process started but health check failed")


@bridge_app.command("stop")
def bridge_stop() -> None:
    pid_file = _bridge_pid_file()
    if not pid_file.exists():
        _ok("bridge.stop", {"status": "not-running"})
        return
    pid = int(pid_file.read_text(encoding="utf-8").strip())
    try:
        os.kill(pid, signal.SIGTERM)
        pid_file.unlink(missing_ok=True)
        _ok("bridge.stop", {"status": "stopped", "pid": pid})
    except Exception as exc:
        _fail("bridge.stop", "ERROR", str(exc))


@bridge_app.command("status")
def bridge_status() -> None:
    client = _bridge_client()
    try:
        health = client.health()
        _ok("bridge.status", {"running": True, "health": health, "url": client.url})
    except BridgeClientError as exc:
        _fail("bridge.status", exc.code, exc.message, retryable=True)


@bridge_app.command("verify")
def bridge_verify(
    iterations: int = typer.Option(25, "--iterations", min=1, max=500),
    max_failures: int = typer.Option(0, "--max-failures", min=0),
) -> None:
    client = _bridge_client()
    failures = 0
    latencies_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            client.call("system.health", {})
        except BridgeClientError:
            failures += 1
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(round(elapsed_ms, 3))
        time.sleep(0.02)
    stable = failures <= max_failures
    data = {
        "stable": stable,
        "iterations": iterations,
        "failures": failures,
        "maxFailuresAllowed": max_failures,
        "latencyMs": {
            "min": min(latencies_ms),
            "max": max(latencies_ms),
            "avg": round(sum(latencies_ms) / len(latencies_ms), 3),
        },
    }
    if not stable:
        _ok("bridge.verify", data)
        raise SystemExit(ERROR_CODES["ERROR"])
    _ok("bridge.verify", data)


@bridge_app.command("soak")
def bridge_soak(
    iterations: int = typer.Option(100, "--iterations", min=1, max=10000),
    duration_seconds: float = typer.Option(5.0, "--duration-seconds", min=0.1, max=300.0),
    action: str = typer.Option("system.health", "--action"),
) -> None:
    _ok(
        "bridge.soak",
        _call_bridge(
            "bridge.soak",
            "system.soak",
            {
                "iterations": iterations,
                "duration_seconds": duration_seconds,
                "action": action,
                "action_params": {},
            },
            timeout_seconds=max(10, duration_seconds + 5),
        ),
    )


@app.command("actions")
def actions() -> None:
    _ok("actions", _call_bridge("actions", "system.actions", {}))


@app.command("doctor")
def doctor(
    report_on_failure: bool = True,
    include_render: bool = True,
    report_url: Optional[str] = None,
) -> None:
    _ensure_bridge_ready("doctor")
    data = _call_bridge(
        "doctor",
        "system.doctor",
        {
            "report_on_failure": report_on_failure,
            "include_render": include_render,
            "report_url": report_url,
        },
        timeout_seconds=180,
    )
    _ok("doctor", data)
    if not data.get("healthy", False):
        raise SystemExit(ERROR_CODES["ERROR"])


@app.command("inspect")
def inspect_project(project: Path) -> None:
    _ok("inspect", _call_bridge("inspect", "project.inspect", {"project": str(project)}))


@app.command("validate")
def validate_project(
    project: Path,
    check_files: bool = True,
) -> None:
    data = _call_bridge(
        "validate",
        "project.validate",
        {"project": str(project), "check_files": check_files},
    )
    _ok("validate", data)
    if not data.get("isValid", False):
        raise SystemExit(ERROR_CODES["VALIDATION_FAILED"])


@app.command("diff")
def diff_projects(source: Path, target: Path) -> None:
    _ok("diff", _call_bridge("diff", "project.diff", {"source": str(source), "target": str(target)}))


@app.command("plan-edit")
def plan_edit(project: Path, action: str, params_json: str = "{}") -> None:
    _ensure_bridge_ready("plan-edit")
    _ok(
        "plan-edit",
        _call_bridge(
            "plan-edit",
            "project.plan_edit",
            {"project": str(project), "action": action, "params": _json_arg("plan-edit", params_json)},
        ),
    )


@app.command("undo")
def undo(project: Path, snapshot_id: Optional[str] = None) -> None:
    _ensure_bridge_ready("undo")
    _ok("undo", _call_bridge("undo", "project.undo", {"project": str(project), "snapshot_id": snapshot_id}))


@app.command("redo")
def redo(project: Path) -> None:
    _ensure_bridge_ready("redo")
    _ok("redo", _call_bridge("redo", "project.redo", {"project": str(project)}))


@app.command("recalc-bounds")
def recalc_bounds(project: Path, output: Optional[Path] = None) -> None:
    _ensure_bridge_ready("recalc-bounds")
    _ok(
        "recalc-bounds",
        _call_bridge(
            "recalc-bounds",
            "project.recalculate_timeline_bounds",
            {"project": str(project), "output": str(output) if output else None},
        ),
    )


@app.command("autosave")
def autosave(project: Path, interval_seconds: int = 60, enabled: bool = True) -> None:
    _ensure_bridge_ready("autosave")
    _ok(
        "autosave",
        _call_bridge(
            "autosave",
            "project.autosave",
            {"project": str(project), "interval_seconds": interval_seconds, "enabled": enabled},
        ),
    )


@app.command("pack-project")
def pack_project(project: Path, output_dir: Path, media_dir_name: str = "media") -> None:
    _ensure_bridge_ready("pack-project")
    _ok(
        "pack-project",
        _call_bridge(
            "pack-project",
            "project.pack",
            {"project": str(project), "output_dir": str(output_dir), "media_dir_name": media_dir_name},
        ),
    )


@app.command("create-project")
def create_project(
    output: Path,
    title: Optional[str] = None,
    width: int = 1920,
    height: int = 1080,
    fps: float = 30.0,
    overwrite: bool = False,
) -> None:
    _ensure_bridge_ready("create-project")
    _ok(
        "create-project",
        _call_bridge(
            "create-project",
            "project.create",
            {
                "output": str(output),
                "title": title,
                "width": width,
                "height": height,
                "fps": fps,
                "overwrite": overwrite,
            },
        ),
    )


@app.command("clone-project")
def clone_project(source: Path, target: Path, overwrite: bool = False) -> None:
    _ensure_bridge_ready("clone-project")
    _ok(
        "clone-project",
        _call_bridge(
            "clone-project",
            "project.clone",
            {"source": str(source), "target": str(target), "overwrite": overwrite},
        ),
    )


@app.command("add-clip")
def add_clip(
    project: Path,
    clip_id: str,
    track_id: str,
    position: int,
    in_point: str = "0",
    out_point: Optional[str] = None,
    output: Optional[Path] = None,
    dry_run: bool = False,
) -> None:
    _ensure_bridge_ready("add-clip")
    _ok(
        "add-clip",
        _call_bridge(
            "add-clip",
            "timeline.add_clip",
            {
                "project": str(project),
                "clip_id": clip_id,
                "track_id": track_id,
                "position": position,
                "in_point": in_point,
                "out_point": out_point,
                "output": str(output) if output else None,
                "dry_run": dry_run,
            },
        ),
    )


@app.command("move-clip")
def move_clip(
    project: Path,
    clip_ref: str,
    track_id: str,
    position: int,
    output: Optional[Path] = None,
    dry_run: bool = False,
) -> None:
    _ensure_bridge_ready("move-clip")
    _ok(
        "move-clip",
        _call_bridge(
            "move-clip",
            "timeline.move_clip",
            {
                "project": str(project),
                "clip_ref": clip_ref,
                "track_id": track_id,
                "position": position,
                "output": str(output) if output else None,
                "dry_run": dry_run,
            },
        ),
    )


@app.command("trim-clip")
def trim_clip(
    project: Path,
    clip_ref: str,
    in_point: Optional[str] = None,
    out_point: Optional[str] = None,
    output: Optional[Path] = None,
    dry_run: bool = False,
) -> None:
    _ensure_bridge_ready("trim-clip")
    _ok(
        "trim-clip",
        _call_bridge(
            "trim-clip",
            "timeline.trim_clip",
            {
                "project": str(project),
                "clip_ref": clip_ref,
                "in_point": in_point,
                "out_point": out_point,
                "output": str(output) if output else None,
                "dry_run": dry_run,
            },
        ),
    )


@app.command("remove-clip")
def remove_clip(
    project: Path,
    clip_ref: str,
    close_gap: bool = False,
    output: Optional[Path] = None,
    dry_run: bool = False,
) -> None:
    _ensure_bridge_ready("remove-clip")
    _ok(
        "remove-clip",
        _call_bridge(
            "remove-clip",
            "timeline.remove_clip",
            {
                "project": str(project),
                "clip_ref": clip_ref,
                "close_gap": close_gap,
                "output": str(output) if output else None,
                "dry_run": dry_run,
            },
        ),
    )


@app.command("snapshot")
def snapshot(project: Path, description: str) -> None:
    _ensure_bridge_ready("snapshot")
    _ok(
        "snapshot",
        _call_bridge(
            "snapshot",
            "project.snapshot",
            {"project": str(project), "description": description},
        ),
    )


@app.command("import-asset")
def import_asset(
    project: Path,
    media: Path,
    producer_id: Optional[str] = None,
    output: Optional[Path] = None,
    dry_run: bool = False,
) -> None:
    _ensure_bridge_ready("import-asset")
    _ok(
        "import-asset",
        _call_bridge(
            "import-asset",
            "asset.import",
            {
                "project": str(project),
                "media": str(media),
                "producer_id": producer_id,
                "output": str(output) if output else None,
                "dry_run": dry_run,
            },
        ),
    )


@app.command("add-text")
def add_text(
    project: Path,
    text: str,
    duration_frames: int = 90,
    track_id: Optional[str] = None,
    position: int = 0,
    font: str = "DejaVu Sans",
    size: int = 64,
    color: str = "#ffffff",
    output: Optional[Path] = None,
    dry_run: bool = False,
) -> None:
    _ensure_bridge_ready("add-text")
    _ok(
        "add-text",
        _call_bridge(
            "add-text",
            "asset.create_text",
            {
                "project": str(project),
                "text": text,
                "duration_frames": duration_frames,
                "track_id": track_id,
                "position": position,
                "font": font,
                "size": size,
                "color": color,
                "output": str(output) if output else None,
                "dry_run": dry_run,
            },
        ),
    )


@app.command("update-text")
def update_text(
    project: Path,
    producer_id: str,
    text: Optional[str] = None,
    font: Optional[str] = None,
    size: Optional[int] = None,
    color: Optional[str] = None,
    duration_frames: Optional[int] = None,
    output: Optional[Path] = None,
) -> None:
    _ensure_bridge_ready("update-text")
    _ok(
        "update-text",
        _call_bridge(
            "update-text",
            "asset.update_text",
            {
                "project": str(project),
                "producer_id": producer_id,
                "text": text,
                "font": font,
                "size": size,
                "color": color,
                "duration_frames": duration_frames,
                "output": str(output) if output else None,
            },
        ),
    )


@app.command("asset-metadata")
def asset_metadata(project: Path, producer_id: str) -> None:
    _ensure_bridge_ready("asset-metadata")
    _ok(
        "asset-metadata",
        _call_bridge(
            "asset-metadata",
            "asset.metadata",
            {"project": str(project), "producer_id": producer_id},
        ),
    )


@app.command("replace-asset")
def replace_asset(
    project: Path,
    producer_id: str,
    media: Path,
    update_name: bool = True,
    output: Optional[Path] = None,
) -> None:
    _ensure_bridge_ready("replace-asset")
    _ok(
        "replace-asset",
        _call_bridge(
            "replace-asset",
            "asset.replace",
            {
                "project": str(project),
                "producer_id": producer_id,
                "media": str(media),
                "update_name": update_name,
                "output": str(output) if output else None,
            },
        ),
    )


@app.command("list-bin")
def list_bin(project: Path) -> None:
    _ensure_bridge_ready("list-bin")
    _ok("list-bin", _call_bridge("list-bin", "bin.list", {"project": str(project)}))


@app.command("create-bin-folder")
def create_bin_folder(
    project: Path,
    name: str,
    parent_id: int = -1,
    output: Optional[Path] = None,
) -> None:
    _ensure_bridge_ready("create-bin-folder")
    _ok(
        "create-bin-folder",
        _call_bridge(
            "create-bin-folder",
            "bin.create_folder",
            {
                "project": str(project),
                "name": name,
                "parent_id": parent_id,
                "output": str(output) if output else None,
            },
        ),
    )


@app.command("move-asset-to-folder")
def move_asset_to_folder(
    project: Path,
    producer_id: str,
    folder_id: int,
    output: Optional[Path] = None,
) -> None:
    _ensure_bridge_ready("move-asset-to-folder")
    _ok(
        "move-asset-to-folder",
        _call_bridge(
            "move-asset-to-folder",
            "bin.move_asset",
            {
                "project": str(project),
                "producer_id": producer_id,
                "folder_id": folder_id,
                "output": str(output) if output else None,
            },
        ),
    )


@app.command("set-effect-keyframes")
def set_effect_keyframes(
    project: Path,
    clip_ref: str,
    effect_id: str,
    parameter: str,
    keyframes_json: str,
    output: Optional[Path] = None,
) -> None:
    _ensure_bridge_ready("set-effect-keyframes")
    _ok(
        "set-effect-keyframes",
        _call_bridge(
            "set-effect-keyframes",
            "effect.keyframes",
            {
                "project": str(project),
                "clip_ref": clip_ref,
                "effect_id": effect_id,
                "parameter": parameter,
                "keyframes": json.loads(keyframes_json),
                "output": str(output) if output else None,
            },
        ),
    )


@app.command("version")
def version() -> None:
    _ok(
        "version",
        {
            "version": __version__,
            "package": "harnessgg-kdenlive",
            "cli": "harness-kdenlive",
        },
    )


@app.command("render-clip")
def render_clip(
    source: Path,
    output: Path,
    duration_seconds: float,
    start_seconds: float = 0.0,
    preset_name: str = "h264",
) -> None:
    _ensure_bridge_ready("render-clip")
    _ok(
        "render-clip",
        _call_bridge(
            "render-clip",
            "render.clip",
            {
                "source": str(source),
                "output": str(output),
                "duration_seconds": duration_seconds,
                "start_seconds": start_seconds,
                "preset_name": preset_name,
            },
            timeout_seconds=600,
        ),
    )


@app.command("split-clip")
def split_clip(
    project: Path,
    clip_ref: str,
    position: int,
    output: Optional[Path] = None,
    dry_run: bool = False,
) -> None:
    _ensure_bridge_ready("split-clip")
    _ok(
        "split-clip",
        _call_bridge(
            "split-clip",
            "timeline.split_clip",
            {
                "project": str(project),
                "clip_ref": clip_ref,
                "position": position,
                "output": str(output) if output else None,
                "dry_run": dry_run,
            },
        ),
    )


@app.command("ripple-delete")
def ripple_delete(
    project: Path,
    clip_ref: str,
    output: Optional[Path] = None,
    dry_run: bool = False,
) -> None:
    _ensure_bridge_ready("ripple-delete")
    _ok(
        "ripple-delete",
        _call_bridge(
            "ripple-delete",
            "timeline.ripple_delete",
            {
                "project": str(project),
                "clip_ref": clip_ref,
                "output": str(output) if output else None,
                "dry_run": dry_run,
            },
        ),
    )


@app.command("insert-gap")
def insert_gap(
    project: Path,
    track_id: str,
    position: int,
    length: int,
    output: Optional[Path] = None,
    dry_run: bool = False,
) -> None:
    _ensure_bridge_ready("insert-gap")
    _ok(
        "insert-gap",
        _call_bridge(
            "insert-gap",
            "timeline.insert_gap",
            {
                "project": str(project),
                "track_id": track_id,
                "position": position,
                "length": length,
                "output": str(output) if output else None,
                "dry_run": dry_run,
            },
        ),
    )


@app.command("remove-all-gaps")
def remove_all_gaps(
    project: Path,
    track_id: str,
    output: Optional[Path] = None,
    dry_run: bool = False,
) -> None:
    _ensure_bridge_ready("remove-all-gaps")
    _ok(
        "remove-all-gaps",
        _call_bridge(
            "remove-all-gaps",
            "timeline.remove_all_gaps",
            {
                "project": str(project),
                "track_id": track_id,
                "output": str(output) if output else None,
                "dry_run": dry_run,
            },
        ),
    )


@app.command("stitch-clips")
def stitch_clips(
    project: Path,
    track_id: str,
    clip_ids: List[str],
    position: Optional[int] = None,
    gap: int = 0,
    duration_frames: Optional[int] = None,
    output: Optional[Path] = None,
    dry_run: bool = False,
) -> None:
    _ensure_bridge_ready("stitch-clips")
    _ok(
        "stitch-clips",
        _call_bridge(
            "stitch-clips",
            "timeline.stitch_clips",
            {
                "project": str(project),
                "track_id": track_id,
                "clip_ids": clip_ids,
                "position": position,
                "gap": gap,
                "duration_frames": duration_frames,
                "output": str(output) if output else None,
                "dry_run": dry_run,
            },
        ),
    )


@app.command("list-clips")
def list_clips(
    project: Path,
    track_id: Optional[str] = None,
    producer_id: Optional[str] = None,
) -> None:
    _ok(
        "list-clips",
        _call_bridge(
            "list-clips",
            "timeline.list_clips",
            {"project": str(project), "track_id": track_id, "producer_id": producer_id},
        ),
    )


@app.command("resolve-clip")
def resolve_clip(
    project: Path,
    selector: str,
    track_id: Optional[str] = None,
    at_frame: Optional[int] = None,
) -> None:
    _ok(
        "resolve-clip",
        _call_bridge(
            "resolve-clip",
            "clip.resolve",
            {"project": str(project), "selector": selector, "track_id": track_id, "at_frame": at_frame},
        ),
    )


@app.command("select-zone")
def select_zone(project: Path, zone_in: int = 0, zone_out: int = 0, output: Optional[Path] = None) -> None:
    _ensure_bridge_ready("select-zone")
    _ok(
        "select-zone",
        _call_bridge(
            "select-zone",
            "timeline.select_zone",
            {"project": str(project), "zone_in": zone_in, "zone_out": zone_out, "output": str(output) if output else None},
        ),
    )


@app.command("detect-gaps")
def detect_gaps(project: Path, track_id: Optional[str] = None) -> None:
    _ok(
        "detect-gaps",
        _call_bridge("detect-gaps", "timeline.detect_gaps", {"project": str(project), "track_id": track_id}),
    )


@app.command("time-remap")
def time_remap(
    project: Path,
    clip_ref: str,
    speed: float,
    output: Optional[Path] = None,
) -> None:
    _ensure_bridge_ready("time-remap")
    _ok(
        "time-remap",
        _call_bridge(
            "time-remap",
            "timeline.time_remap",
            {
                "project": str(project),
                "clip_ref": clip_ref,
                "speed": speed,
                "output": str(output) if output else None,
            },
        ),
    )


@app.command("nudge-clip")
def nudge_clip(project: Path, clip_ref: str, delta_frames: int, output: Optional[Path] = None) -> None:
    _ensure_bridge_ready("nudge-clip")
    _ok(
        "nudge-clip",
        _call_bridge(
            "nudge-clip",
            "timeline.nudge_clip",
            {
                "project": str(project),
                "clip_ref": clip_ref,
                "delta_frames": delta_frames,
                "output": str(output) if output else None,
            },
        ),
    )


@app.command("slip-clip")
def slip_clip(project: Path, clip_ref: str, delta_frames: int, output: Optional[Path] = None) -> None:
    _ensure_bridge_ready("slip-clip")
    _ok(
        "slip-clip",
        _call_bridge(
            "slip-clip",
            "timeline.slip_clip",
            {
                "project": str(project),
                "clip_ref": clip_ref,
                "delta_frames": delta_frames,
                "output": str(output) if output else None,
            },
        ),
    )


@app.command("slide-clip")
def slide_clip(project: Path, clip_ref: str, delta_frames: int, output: Optional[Path] = None) -> None:
    _ensure_bridge_ready("slide-clip")
    _ok(
        "slide-clip",
        _call_bridge(
            "slide-clip",
            "timeline.slide_clip",
            {
                "project": str(project),
                "clip_ref": clip_ref,
                "delta_frames": delta_frames,
                "output": str(output) if output else None,
            },
        ),
    )


@app.command("ripple-insert")
def ripple_insert(
    project: Path,
    track_id: str,
    position: int,
    length: int = 1,
    clip_id: Optional[str] = None,
    in_point: str = "0",
    out_point: Optional[str] = None,
    output: Optional[Path] = None,
) -> None:
    _ensure_bridge_ready("ripple-insert")
    _ok(
        "ripple-insert",
        _call_bridge(
            "ripple-insert",
            "timeline.ripple_insert",
            {
                "project": str(project),
                "track_id": track_id,
                "position": position,
                "length": length,
                "clip_id": clip_id,
                "in_point": in_point,
                "out_point": out_point,
                "output": str(output) if output else None,
            },
        ),
    )


@app.command("group-clips")
def group_clips(project: Path, clip_refs: List[str], group_id: Optional[str] = None, output: Optional[Path] = None) -> None:
    _ensure_bridge_ready("group-clips")
    _ok(
        "group-clips",
        _call_bridge(
            "group-clips",
            "timeline.group_clips",
            {
                "project": str(project),
                "clip_refs": clip_refs,
                "group_id": group_id,
                "output": str(output) if output else None,
            },
        ),
    )


@app.command("ungroup-clips")
def ungroup_clips(project: Path, clip_refs: List[str], output: Optional[Path] = None) -> None:
    _ensure_bridge_ready("ungroup-clips")
    _ok(
        "ungroup-clips",
        _call_bridge(
            "ungroup-clips",
            "timeline.ungroup_clips",
            {"project": str(project), "clip_refs": clip_refs, "output": str(output) if output else None},
        ),
    )


@app.command("transform-clip")
def transform_clip(
    project: Path,
    clip_ref: str,
    geometry: Optional[str] = None,
    rotate: Optional[float] = None,
    scale: Optional[float] = None,
    opacity: Optional[float] = None,
    keyframes_json: Optional[str] = None,
    output: Optional[Path] = None,
) -> None:
    _ensure_bridge_ready("transform-clip")
    _ok(
        "transform-clip",
        _call_bridge(
            "transform-clip",
            "timeline.transform",
            {
                "project": str(project),
                "clip_ref": clip_ref,
                "geometry": geometry,
                "rotate": rotate,
                "scale": scale,
                "opacity": opacity,
                "keyframes": json.loads(keyframes_json) if keyframes_json else None,
                "output": str(output) if output else None,
            },
        ),
    )


@app.command("list-effects")
def list_effects(project: Path, clip_ref: str) -> None:
    _ensure_bridge_ready("list-effects")
    _ok("list-effects", _call_bridge("list-effects", "effect.list", {"project": str(project), "clip_ref": clip_ref}))


@app.command("apply-effect")
def apply_effect(
    project: Path,
    clip_ref: str,
    service: str,
    effect_id: Optional[str] = None,
    properties_json: str = "{}",
    output: Optional[Path] = None,
) -> None:
    _ensure_bridge_ready("apply-effect")
    _ok(
        "apply-effect",
        _call_bridge(
            "apply-effect",
            "effect.apply",
            {
                "project": str(project),
                "clip_ref": clip_ref,
                "service": service,
                "effect_id": effect_id,
                "properties": _json_arg("apply-effect", properties_json),
                "output": str(output) if output else None,
            },
        ),
    )


@app.command("update-effect")
def update_effect(
    project: Path,
    clip_ref: str,
    effect_id: str,
    properties_json: str,
    output: Optional[Path] = None,
) -> None:
    _ensure_bridge_ready("update-effect")
    _ok(
        "update-effect",
        _call_bridge(
            "update-effect",
            "effect.update",
            {
                "project": str(project),
                "clip_ref": clip_ref,
                "effect_id": effect_id,
                "properties": _json_arg("update-effect", properties_json),
                "output": str(output) if output else None,
            },
        ),
    )


@app.command("remove-effect")
def remove_effect(project: Path, clip_ref: str, effect_id: str, output: Optional[Path] = None) -> None:
    _ensure_bridge_ready("remove-effect")
    _ok(
        "remove-effect",
        _call_bridge(
            "remove-effect",
            "effect.remove",
            {
                "project": str(project),
                "clip_ref": clip_ref,
                "effect_id": effect_id,
                "output": str(output) if output else None,
            },
        ),
    )


@app.command("list-transitions")
def list_transitions(project: Path) -> None:
    _ensure_bridge_ready("list-transitions")
    _ok("list-transitions", _call_bridge("list-transitions", "transition.list", {"project": str(project)}))


@app.command("apply-transition")
def apply_transition(
    project: Path,
    in_frame: int = 0,
    out_frame: int = 0,
    service: str = "mix",
    transition_id: Optional[str] = None,
    properties_json: str = "{}",
    output: Optional[Path] = None,
) -> None:
    _ensure_bridge_ready("apply-transition")
    _ok(
        "apply-transition",
        _call_bridge(
            "apply-transition",
            "transition.apply",
            {
                "project": str(project),
                "in_frame": in_frame,
                "out_frame": out_frame,
                "service": service,
                "transition_id": transition_id,
                "properties": _json_arg("apply-transition", properties_json),
                "output": str(output) if output else None,
            },
        ),
    )


@app.command("remove-transition")
def remove_transition(project: Path, transition_id: str, output: Optional[Path] = None) -> None:
    _ensure_bridge_ready("remove-transition")
    _ok(
        "remove-transition",
        _call_bridge(
            "remove-transition",
            "transition.remove",
            {"project": str(project), "transition_id": transition_id, "output": str(output) if output else None},
        ),
    )


@app.command("apply-wipe")
def apply_wipe(
    project: Path,
    in_frame: int = 0,
    out_frame: int = 0,
    preset: str = "circle",
    transition_id: Optional[str] = None,
    softness: float = 0.05,
    invert: bool = False,
    output: Optional[Path] = None,
) -> None:
    _ensure_bridge_ready("apply-wipe")
    _ok(
        "apply-wipe",
        _call_bridge(
            "apply-wipe",
            "transition.wipe",
            {
                "project": str(project),
                "in_frame": in_frame,
                "out_frame": out_frame,
                "preset": preset,
                "transition_id": transition_id,
                "softness": softness,
                "invert": invert,
                "output": str(output) if output else None,
            },
        ),
    )


@app.command("add-music-bed")
def add_music_bed(
    project: Path,
    media: Path,
    track_id: str = "playlist1",
    position: int = 0,
    duration_frames: Optional[int] = None,
    producer_id: Optional[str] = None,
    output: Optional[Path] = None,
) -> None:
    _ensure_bridge_ready("add-music-bed")
    _ok(
        "add-music-bed",
        _call_bridge(
            "add-music-bed",
            "audio.add_music",
            {
                "project": str(project),
                "media": str(media),
                "track_id": track_id,
                "position": position,
                "duration_frames": duration_frames,
                "producer_id": producer_id,
                "output": str(output) if output else None,
            },
        ),
    )


@app.command("duck-audio")
def duck_audio(
    project: Path,
    track_id: str,
    duck_gain: float = 0.3,
    output: Optional[Path] = None,
) -> None:
    _ensure_bridge_ready("duck-audio")
    _ok(
        "duck-audio",
        _call_bridge(
            "duck-audio",
            "audio.duck",
            {
                "project": str(project),
                "track_id": track_id,
                "duck_gain": duck_gain,
                "output": str(output) if output else None,
            },
        ),
    )


@app.command("audio-fade")
def audio_fade(
    project: Path,
    clip_ref: str,
    fade_type: str = "in",
    frames: int = 24,
    output: Optional[Path] = None,
) -> None:
    _ensure_bridge_ready("audio-fade")
    _ok(
        "audio-fade",
        _call_bridge(
            "audio-fade",
            "audio.fade",
            {
                "project": str(project),
                "clip_ref": clip_ref,
                "fade_type": fade_type,
                "frames": frames,
                "output": str(output) if output else None,
            },
        ),
    )


@app.command("grade-clip")
def grade_clip(
    project: Path,
    clip_ref: str,
    lift: Optional[float] = None,
    gamma: Optional[float] = None,
    gain: Optional[float] = None,
    saturation: Optional[float] = None,
    temperature: Optional[float] = None,
    lut_path: Optional[str] = None,
    output: Optional[Path] = None,
) -> None:
    _ensure_bridge_ready("grade-clip")
    _ok(
        "grade-clip",
        _call_bridge(
            "grade-clip",
            "color.grade",
            {
                "project": str(project),
                "clip_ref": clip_ref,
                "lift": lift,
                "gamma": gamma,
                "gain": gain,
                "saturation": saturation,
                "temperature": temperature,
                "lut_path": lut_path,
                "output": str(output) if output else None,
            },
        ),
    )


@app.command("add-track")
def add_track(
    project: Path,
    track_type: str = "video",
    name: Optional[str] = None,
    index: Optional[int] = None,
    track_id: Optional[str] = None,
    output: Optional[Path] = None,
) -> None:
    _ensure_bridge_ready("add-track")
    _ok(
        "add-track",
        _call_bridge(
            "add-track",
            "track.add",
            {
                "project": str(project),
                "track_type": track_type,
                "name": name,
                "index": index,
                "track_id": track_id,
                "output": str(output) if output else None,
            },
        ),
    )


@app.command("remove-track")
def remove_track(
    project: Path,
    track_id: str,
    force: bool = False,
    output: Optional[Path] = None,
) -> None:
    _ensure_bridge_ready("remove-track")
    _ok(
        "remove-track",
        _call_bridge(
            "remove-track",
            "track.remove",
            {
                "project": str(project),
                "track_id": track_id,
                "force": force,
                "output": str(output) if output else None,
            },
        ),
    )


@app.command("reorder-track")
def reorder_track(
    project: Path,
    track_id: str,
    index: int,
    output: Optional[Path] = None,
) -> None:
    _ensure_bridge_ready("reorder-track")
    _ok(
        "reorder-track",
        _call_bridge(
            "reorder-track",
            "track.reorder",
            {
                "project": str(project),
                "track_id": track_id,
                "index": index,
                "output": str(output) if output else None,
            },
        ),
    )


@app.command("resolve-track")
def resolve_track(project: Path, selector: str) -> None:
    _ok(
        "resolve-track",
        _call_bridge("resolve-track", "track.resolve", {"project": str(project), "selector": selector}),
    )


@app.command("track-mute")
def track_mute(project: Path, track_id: str, output: Optional[Path] = None) -> None:
    _ensure_bridge_ready("track-mute")
    _ok(
        "track-mute",
        _call_bridge(
            "track-mute",
            "track.mute",
            {"project": str(project), "track_id": track_id, "output": str(output) if output else None},
        ),
    )


@app.command("track-unmute")
def track_unmute(project: Path, track_id: str, output: Optional[Path] = None) -> None:
    _ensure_bridge_ready("track-unmute")
    _ok(
        "track-unmute",
        _call_bridge(
            "track-unmute",
            "track.unmute",
            {"project": str(project), "track_id": track_id, "output": str(output) if output else None},
        ),
    )


@app.command("track-lock")
def track_lock(project: Path, track_id: str, output: Optional[Path] = None) -> None:
    _ensure_bridge_ready("track-lock")
    _ok(
        "track-lock",
        _call_bridge(
            "track-lock",
            "track.lock",
            {"project": str(project), "track_id": track_id, "output": str(output) if output else None},
        ),
    )


@app.command("track-unlock")
def track_unlock(project: Path, track_id: str, output: Optional[Path] = None) -> None:
    _ensure_bridge_ready("track-unlock")
    _ok(
        "track-unlock",
        _call_bridge(
            "track-unlock",
            "track.unlock",
            {"project": str(project), "track_id": track_id, "output": str(output) if output else None},
        ),
    )


@app.command("track-show")
def track_show(project: Path, track_id: str, output: Optional[Path] = None) -> None:
    _ensure_bridge_ready("track-show")
    _ok(
        "track-show",
        _call_bridge(
            "track-show",
            "track.show",
            {"project": str(project), "track_id": track_id, "output": str(output) if output else None},
        ),
    )


@app.command("track-hide")
def track_hide(project: Path, track_id: str, output: Optional[Path] = None) -> None:
    _ensure_bridge_ready("track-hide")
    _ok(
        "track-hide",
        _call_bridge(
            "track-hide",
            "track.hide",
            {"project": str(project), "track_id": track_id, "output": str(output) if output else None},
        ),
    )


@app.command("render-project")
def render_project(
    project: Path,
    output: Path,
    start_seconds: Optional[float] = None,
    duration_seconds: Optional[float] = None,
    zone_in: Optional[int] = None,
    zone_out: Optional[int] = None,
    preset_name: str = "h264",
) -> None:
    _ensure_bridge_ready("render-project")
    _ok(
        "render-project",
        _call_bridge(
            "render-project",
            "render.project",
            {
                "project": str(project),
                "output": str(output),
                "start_seconds": start_seconds,
                "duration_seconds": duration_seconds,
                "zone_in": zone_in,
                "zone_out": zone_out,
                "preset_name": preset_name,
            },
            timeout_seconds=600,
        ),
    )


@app.command("render-status")
def render_status(job_id: str) -> None:
    _ensure_bridge_ready("render-status")
    _ok("render-status", _call_bridge("render-status", "render.status", {"job_id": job_id}))


@app.command("render-latest")
def render_latest(type: Optional[str] = None, status: Optional[str] = None) -> None:
    _ok("render-latest", _call_bridge("render-latest", "render.latest", {"type": type, "status": status}))


@app.command("render-retry")
def render_retry(job_id: str, output: Optional[Path] = None) -> None:
    _ensure_bridge_ready("render-retry")
    _ok(
        "render-retry",
        _call_bridge(
            "render-retry",
            "render.retry",
            {"job_id": job_id, "output": str(output) if output else None},
            timeout_seconds=600,
        ),
    )


@app.command("render-cancel")
def render_cancel(job_id: str) -> None:
    _ensure_bridge_ready("render-cancel")
    _ok("render-cancel", _call_bridge("render-cancel", "render.cancel", {"job_id": job_id}))


@app.command("render-list-jobs")
def render_list_jobs() -> None:
    _ensure_bridge_ready("render-list-jobs")
    _ok("render-list-jobs", _call_bridge("render-list-jobs", "render.list_jobs", {}))


@app.command("render-wait")
def render_wait(
    job_id: str,
    timeout_seconds: float = 120.0,
    poll_interval_seconds: float = 0.2,
) -> None:
    _ensure_bridge_ready("render-wait")
    _ok(
        "render-wait",
        _call_bridge(
            "render-wait",
            "render.wait",
            {
                "job_id": job_id,
                "timeout_seconds": timeout_seconds,
                "poll_interval_seconds": poll_interval_seconds,
            },
            timeout_seconds=max(30, timeout_seconds + 5),
        ),
    )


@app.command("list-sequences")
def list_sequences(project: Path) -> None:
    _ensure_bridge_ready("list-sequences")
    _ok("list-sequences", _call_bridge("list-sequences", "sequence.list", {"project": str(project)}))


@app.command("copy-sequence")
def copy_sequence(project: Path, source_id: str, new_id: Optional[str] = None, output: Optional[Path] = None) -> None:
    _ensure_bridge_ready("copy-sequence")
    _ok(
        "copy-sequence",
        _call_bridge(
            "copy-sequence",
            "sequence.copy",
            {"project": str(project), "source_id": source_id, "new_id": new_id, "output": str(output) if output else None},
        ),
    )


@app.command("set-active-sequence")
def set_active_sequence(project: Path, sequence_id: str, output: Optional[Path] = None) -> None:
    _ensure_bridge_ready("set-active-sequence")
    _ok(
        "set-active-sequence",
        _call_bridge(
            "set-active-sequence",
            "sequence.set_active",
            {"project": str(project), "sequence_id": sequence_id, "output": str(output) if output else None},
        ),
    )


@app.command("export-edl")
def export_edl(project: Path, output: Path) -> None:
    _ensure_bridge_ready("export-edl")
    _ok("export-edl", _call_bridge("export-edl", "export.edl", {"project": str(project), "output": str(output)}))


@app.command("export-xml")
def export_xml(project: Path, output: Path) -> None:
    _ensure_bridge_ready("export-xml")
    _ok("export-xml", _call_bridge("export-xml", "export.xml", {"project": str(project), "output": str(output)}))


@app.command("export-otio")
def export_otio(project: Path, output: Path) -> None:
    _ensure_bridge_ready("export-otio")
    _ok("export-otio", _call_bridge("export-otio", "export.otio", {"project": str(project), "output": str(output)}))


@app.command("normalize-audio")
def normalize_audio(project: Path, clip_ref: str, target_db: float = -14.0, output: Optional[Path] = None) -> None:
    _ensure_bridge_ready("normalize-audio")
    _ok(
        "normalize-audio",
        _call_bridge(
            "normalize-audio",
            "audio.normalize",
            {"project": str(project), "clip_ref": clip_ref, "target_db": target_db, "output": str(output) if output else None},
        ),
    )


@app.command("remove-silence")
def remove_silence(
    project: Path,
    clip_ref: str,
    threshold_db: float = -35.0,
    min_duration_frames: int = 6,
    output: Optional[Path] = None,
) -> None:
    _ensure_bridge_ready("remove-silence")
    _ok(
        "remove-silence",
        _call_bridge(
            "remove-silence",
            "audio.remove_silence",
            {
                "project": str(project),
                "clip_ref": clip_ref,
                "threshold_db": threshold_db,
                "min_duration_frames": min_duration_frames,
                "output": str(output) if output else None,
            },
        ),
    )


@app.command("audio-pan")
def audio_pan(project: Path, clip_ref: str, pan: float, output: Optional[Path] = None) -> None:
    _ensure_bridge_ready("audio-pan")
    _ok(
        "audio-pan",
        _call_bridge(
            "audio-pan",
            "audio.pan",
            {"project": str(project), "clip_ref": clip_ref, "pan": pan, "output": str(output) if output else None},
        ),
    )


@app.command("batch")
def batch(
    steps_json: str,
    stop_on_error: bool = True,
) -> None:
    _ensure_bridge_ready("batch")
    try:
        steps = json.loads(steps_json)
    except json.JSONDecodeError as exc:
        _fail("batch", "INVALID_INPUT", f"Invalid JSON: {exc}")
    _ok(
        "batch",
        _call_bridge(
            "batch",
            "batch.execute",
            {"steps": steps, "stop_on_error": stop_on_error},
        ),
    )


def main() -> None:
    app()
