# Command Spec (LLM)

## Global rules

1. CLI commands emit exactly one JSON object to stdout.
2. Commands are bridge-only except `version`. If bridge is unavailable, bridge-backed calls fail with `BRIDGE_UNAVAILABLE`.
3. Start bridge with `harness-kdenlive bridge start`.
4. Mutating commands perform bridge health preflight (`system.health`) before edit/render calls.
5. `asset.create_text` may return warnings and use subtitle-sidecar fallback when `qtext` is unavailable.
6. `render.project` may internally burn text overlays with `ffmpeg` when local MLT text producers are unavailable.

## Exit codes

- `0`: success
- `1`: internal error
- `2`: file not found
- `3`: validation failed
- `4`: invalid input
- `5`: bridge unavailable

## Bridge commands

- `harness-kdenlive bridge start [--host <ip>] [--port <int>]`
- `harness-kdenlive bridge serve [--host <ip>] [--port <int>]`
- `harness-kdenlive bridge status`
- `harness-kdenlive bridge stop`
- `harness-kdenlive bridge verify [--iterations <int>] [--max-failures <int>]`
- `harness-kdenlive bridge soak [--iterations <int>] [--duration-seconds <float>] [--action <method>]`

## Editing commands

- `harness-kdenlive actions`
- `harness-kdenlive doctor [--report-on-failure/--no-report-on-failure] [--include-render/--no-include-render] [--report-url <url>]`
- `harness-kdenlive create-project <output> [--title <str>] [--width <int>] [--height <int>] [--fps <float>] [--overwrite]`
- `harness-kdenlive clone-project <source> <target> [--overwrite]`
- `harness-kdenlive plan-edit <project> <action> [--params-json <json>]`
- `harness-kdenlive undo <project> [--snapshot-id <id>]`
- `harness-kdenlive redo <project>`
- `harness-kdenlive autosave <project> [--interval-seconds <int>] [--enabled/--no-enabled]`
- `harness-kdenlive pack-project <project> <output-dir> [--media-dir-name <name>]`
- `harness-kdenlive recalc-bounds <project> [--output <path>]`
- `harness-kdenlive inspect <project>`
- `harness-kdenlive validate <project> [--check-files/--no-check-files]`
- `harness-kdenlive diff <source> <target>`
- `harness-kdenlive import-asset <project> <media> [--producer-id <str>] [--output <path>] [--dry-run]`
- `harness-kdenlive add-text <project> <text> [--duration-frames <int>] [--track-id <str>] [--position <int>] [--font <str>] [--size <int>] [--color <hex>] [--output <path>] [--dry-run]`
- `harness-kdenlive update-text <project> <producer_id> [--text <str>] [--font <str>] [--size <int>] [--color <hex>] [--duration-frames <int>] [--output <path>]`
- `harness-kdenlive asset-metadata <project> <producer_id>`
- `harness-kdenlive replace-asset <project> <producer_id> <media> [--update-name/--no-update-name] [--output <path>]`
- `harness-kdenlive list-bin <project>`
- `harness-kdenlive create-bin-folder <project> <name> [--parent-id <int>] [--output <path>]`
- `harness-kdenlive move-asset-to-folder <project> <producer_id> <folder_id> [--output <path>]`
- `harness-kdenlive set-effect-keyframes <project> <clip_ref> <effect_id> <parameter> <keyframes-json> [--output <path>]`
- `harness-kdenlive add-clip <project> <clip_id> <track_id> <position> [--in-point <int>] [--out-point <int>] [--output <path>] [--dry-run]`
- `harness-kdenlive move-clip <project> <clip_ref> <track_id> <position> [--output <path>] [--dry-run]`
- `harness-kdenlive trim-clip <project> <clip_ref> [--in-point <int>] [--out-point <int>] [--output <path>] [--dry-run]`
- `harness-kdenlive remove-clip <project> <clip_ref> [--close-gap] [--output <path>] [--dry-run]`
- `harness-kdenlive split-clip <project> <clip_ref> <position> [--output <path>] [--dry-run]`
- `harness-kdenlive ripple-delete <project> <clip_ref> [--output <path>] [--dry-run]`
- `harness-kdenlive insert-gap <project> <track_id> <position> <length> [--output <path>] [--dry-run]`
- `harness-kdenlive remove-all-gaps <project> <track_id> [--output <path>] [--dry-run]`
- `harness-kdenlive stitch-clips <project> <track_id> <clip_id...> [--position <int>] [--gap <int>] [--duration-frames <int>] [--output <path>] [--dry-run]`
- `harness-kdenlive list-clips <project> [--track-id <str>] [--producer-id <str>]`
- `harness-kdenlive resolve-clip <project> <selector> [--track-id <str>] [--at-frame <int>]`
- `harness-kdenlive select-zone <project> [--zone-in <int>] [--zone-out <int>] [--output <path>]`
- `harness-kdenlive detect-gaps <project> [--track-id <str>]`
- `harness-kdenlive time-remap <project> <clip_ref> <speed> [--output <path>]`
- `harness-kdenlive transform-clip <project> <clip_ref> [--geometry <str>] [--rotate <float>] [--scale <float>] [--opacity <float>] [--keyframes-json <json>] [--output <path>]`
- `harness-kdenlive nudge-clip <project> <clip_ref> <delta_frames> [--output <path>]`
- `harness-kdenlive slip-clip <project> <clip_ref> <delta_frames> [--output <path>]`
- `harness-kdenlive slide-clip <project> <clip_ref> <delta_frames> [--output <path>]`
- `harness-kdenlive ripple-insert <project> <track_id> <position> [--length <int>] [--clip-id <str>] [--in-point <int>] [--out-point <int>] [--output <path>]`
- `harness-kdenlive group-clips <project> <clip_ref...> [--group-id <str>] [--output <path>]`
- `harness-kdenlive ungroup-clips <project> <clip_ref...> [--output <path>]`
- `harness-kdenlive list-sequences <project>`
- `harness-kdenlive copy-sequence <project> <source_id> [--new-id <id>] [--output <path>]`
- `harness-kdenlive set-active-sequence <project> <sequence_id> [--output <path>]`
- `harness-kdenlive list-effects <project> <clip_ref>`
- `harness-kdenlive apply-effect <project> <clip_ref> <service> [--effect-id <id>] [--properties-json <json>] [--output <path>]`
- `harness-kdenlive update-effect <project> <clip_ref> <effect_id> <properties-json> [--output <path>]`
- `harness-kdenlive remove-effect <project> <clip_ref> <effect_id> [--output <path>]`
- `harness-kdenlive list-transitions <project>`
- `harness-kdenlive apply-transition <project> [--in-frame <int>] [--out-frame <int>] [--service <str>] [--transition-id <id>] [--properties-json <json>] [--output <path>]`
- `harness-kdenlive remove-transition <project> <transition_id> [--output <path>]`
- `harness-kdenlive apply-wipe <project> [--in-frame <int>] [--out-frame <int>] [--preset <name>] [--transition-id <id>] [--softness <float>] [--invert] [--output <path>]`
- `harness-kdenlive add-music-bed <project> <media> [--track-id <id>] [--position <int>] [--duration-frames <int>] [--producer-id <id>] [--output <path>]`
- `harness-kdenlive duck-audio <project> <track_id> [--duck-gain <float>] [--output <path>]`
- `harness-kdenlive audio-fade <project> <clip_ref> [--fade-type in|out] [--frames <int>] [--output <path>]`
- `harness-kdenlive normalize-audio <project> <clip_ref> [--target-db <float>] [--output <path>]`
- `harness-kdenlive remove-silence <project> <clip_ref> [--threshold-db <float>] [--min-duration-frames <int>] [--output <path>]`
- `harness-kdenlive audio-pan <project> <clip_ref> <pan> [--output <path>]`
- `harness-kdenlive grade-clip <project> <clip_ref> [--lift <float>] [--gamma <float>] [--gain <float>] [--saturation <float>] [--temperature <float>] [--lut-path <path>] [--output <path>]`
- `harness-kdenlive add-track <project> [--track-type video|audio] [--name <str>] [--index <int>] [--track-id <str>] [--output <path>]`
- `harness-kdenlive remove-track <project> <track_id> [--force] [--output <path>]`
- `harness-kdenlive reorder-track <project> <track_id> <index> [--output <path>]`
- `harness-kdenlive resolve-track <project> <name-or-id>`
- `harness-kdenlive track-mute <project> <track_id> [--output <path>]`
- `harness-kdenlive track-unmute <project> <track_id> [--output <path>]`
- `harness-kdenlive track-lock <project> <track_id> [--output <path>]`
- `harness-kdenlive track-unlock <project> <track_id> [--output <path>]`
- `harness-kdenlive track-show <project> <track_id> [--output <path>]`
- `harness-kdenlive track-hide <project> <track_id> [--output <path>]`
- `harness-kdenlive snapshot <project> <description>`
- `harness-kdenlive render-clip <source.mp4> <output.mp4> <duration_seconds> [--start-seconds <float>] [--preset-name h264|hevc|prores]`
- `harness-kdenlive render-project <project.kdenlive> <output.mp4> [--start-seconds <float>] [--duration-seconds <float>] [--zone-in <int>] [--zone-out <int>] [--preset-name h264|hevc|prores]`
- `harness-kdenlive render-status <job_id>`
- `harness-kdenlive render-latest [--type project|clip] [--status running|completed|failed|canceled]`
- `harness-kdenlive render-retry <job_id> [--output <path>]`
- `harness-kdenlive render-cancel <job_id>`
- `harness-kdenlive render-list-jobs`
- `harness-kdenlive render-wait <job_id> [--timeout-seconds <float>] [--poll-interval-seconds <float>]`
- `harness-kdenlive export-edl <project> <output.edl>`
- `harness-kdenlive export-xml <project> <output.xml>`
- `harness-kdenlive export-otio <project> <output.otio>`
- `harness-kdenlive batch <steps-json> [--stop-on-error/--no-stop-on-error]`
- `harness-kdenlive version`

## Timeouts and Retry Guidance

- Default bridge call timeout: `30s`.
- Health and readiness calls: `5s`.
- Doctor: `180s`.
- Render calls: `600s`.
- Bridge soak timeout: `duration-seconds + 5` (minimum `10s`).
- Retry policy for transient bridge errors:
  - Retry only for `BRIDGE_UNAVAILABLE`.
  - Recommended backoff: `0.5s`, `1s`, `2s` (max 3 retries).
  - Re-run `harness-kdenlive bridge status` before retrying mutating commands.
