# Bridge Protocol

Transport: HTTP JSON-RPC style over localhost.

- Health endpoint: `GET /health`
- RPC endpoint: `POST /rpc`

Request:

```json
{
  "id": "inspect",
  "method": "inspect",
  "params": {
    "project": "C:/path/project.kdenlive"
  }
}
```

Success response:

```json
{
  "ok": true,
  "protocolVersion": "1.0",
  "id": "inspect",
  "result": {}
}
```

Error response:

```json
{
  "ok": false,
  "protocolVersion": "1.0",
  "error": {
    "code": "INVALID_INPUT",
    "message": "Unknown method: xyz"
  }
}
```

Implemented methods:

- `system.health`
- `system.version`
- `system.actions`
- `system.doctor`
- `system.soak`
- `project.create`
- `project.clone`
- `project.plan_edit`
- `project.undo`
- `project.redo`
- `project.autosave`
- `project.pack`
- `project.recalculate_timeline_bounds`
- `project.inspect`
- `project.validate`
- `project.diff`
- `project.snapshot`
- `asset.import`
- `asset.create_text`
- `asset.update_text`
- `asset.metadata`
- `asset.replace`
- `bin.list`
- `bin.create_folder`
- `bin.move_asset`
- `effect.list`
- `effect.apply`
- `effect.update`
- `effect.remove`
- `effect.keyframes`
- `transition.list`
- `transition.apply`
- `transition.remove`
- `transition.wipe`
- `timeline.add_clip`
- `timeline.move_clip`
- `timeline.trim_clip`
- `timeline.remove_clip`
- `timeline.split_clip`
- `timeline.ripple_delete`
- `timeline.insert_gap`
- `timeline.remove_all_gaps`
- `timeline.stitch_clips`
- `timeline.list_clips`
- `timeline.select_zone`
- `timeline.detect_gaps`
- `clip.resolve`
- `timeline.time_remap`
- `timeline.transform`
- `timeline.nudge_clip`
- `timeline.slip_clip`
- `timeline.slide_clip`
- `timeline.ripple_insert`
- `timeline.group_clips`
- `timeline.ungroup_clips`
- `sequence.list`
- `sequence.copy`
- `sequence.set_active`
- `audio.add_music`
- `audio.duck`
- `audio.fade`
- `audio.normalize`
- `audio.remove_silence`
- `audio.pan`
- `color.grade`
- `track.add`
- `track.remove`
- `track.reorder`
- `track.resolve`
- `track.mute`
- `track.unmute`
- `track.lock`
- `track.unlock`
- `track.show`
- `track.hide`
- `render.clip`
- `render.project`
- `render.status`
- `render.latest`
- `render.retry`
- `render.cancel`
- `render.list_jobs`
- `render.wait`
- `export.edl`
- `export.xml`
- `export.otio`
- `batch.execute`

Behavior notes:
- `asset.create_text` may fallback to subtitle-sidecar producer mode when `qtext` is not available.
- `render.project` may fallback to `ffmpeg` subtitle burn-in for harness text overlays when MLT text producers are unavailable.
