# Commands

`harness-kdenlive` is bridge-only. Start the bridge first.

```bash
harness-kdenlive bridge start
harness-kdenlive bridge status
```

## Core Commands

```bash
harness-kdenlive actions
harness-kdenlive doctor [--report-on-failure/--no-report-on-failure] [--include-render/--no-include-render] [--report-url <url>]
harness-kdenlive plan-edit <project.kdenlive> <action> [--params-json <json>]
harness-kdenlive undo <project.kdenlive> [--snapshot-id <id>]
harness-kdenlive redo <project.kdenlive>
harness-kdenlive autosave <project.kdenlive> [--interval-seconds 60] [--enabled/--no-enabled]
harness-kdenlive pack-project <project.kdenlive> <output-dir> [--media-dir-name media]
harness-kdenlive recalc-bounds <project.kdenlive> [--output out.kdenlive]
harness-kdenlive create-project <output.kdenlive> [--title <name>] [--width 1920] [--height 1080] [--fps 30] [--overwrite]
harness-kdenlive clone-project <source.kdenlive> <target.kdenlive> [--overwrite]
harness-kdenlive inspect <project.kdenlive>
harness-kdenlive validate <project.kdenlive> [--check-files/--no-check-files]
harness-kdenlive diff <source.kdenlive> <target.kdenlive>
harness-kdenlive import-asset <project.kdenlive> <media-path> [--producer-id <id>] [--output out.kdenlive] [--dry-run]
harness-kdenlive add-text <project.kdenlive> <text> [--duration-frames 90] [--track-id <id>] [--position 0] [--font "DejaVu Sans"] [--size 64] [--color "#ffffff"] [--output out.kdenlive] [--dry-run]
harness-kdenlive update-text <project.kdenlive> <producer_id> [--text <text>] [--font <font>] [--size <n>] [--color <hex>] [--duration-frames <n>] [--output out.kdenlive]
harness-kdenlive asset-metadata <project.kdenlive> <producer_id>
harness-kdenlive replace-asset <project.kdenlive> <producer_id> <media-path> [--update-name/--no-update-name] [--output out.kdenlive]
harness-kdenlive list-bin <project.kdenlive>
harness-kdenlive create-bin-folder <project.kdenlive> <name> [--parent-id <id>] [--output out.kdenlive]
harness-kdenlive move-asset-to-folder <project.kdenlive> <producer_id> <folder_id> [--output out.kdenlive]
harness-kdenlive set-effect-keyframes <project> <clip_ref> <effect_id> <parameter> <keyframes-json> [--output out.kdenlive]
harness-kdenlive add-clip <project> <clip_id> <track_id> <position> [--in-point 0] [--out-point 49] [--output out.kdenlive] [--dry-run]
harness-kdenlive move-clip <project> <clip_ref> <track_id> <position> [--output out.kdenlive] [--dry-run]
harness-kdenlive trim-clip <project> <clip_ref> [--in-point 0] [--out-point 49] [--output out.kdenlive] [--dry-run]
harness-kdenlive remove-clip <project> <clip_ref> [--close-gap] [--output out.kdenlive] [--dry-run]
harness-kdenlive split-clip <project> <clip_ref> <position> [--output out.kdenlive] [--dry-run]
harness-kdenlive ripple-delete <project> <clip_ref> [--output out.kdenlive] [--dry-run]
harness-kdenlive insert-gap <project> <track_id> <position> <length> [--output out.kdenlive] [--dry-run]
harness-kdenlive remove-all-gaps <project> <track_id> [--output out.kdenlive] [--dry-run]
harness-kdenlive stitch-clips <project> <track_id> <clip_id...> [--position <frame>] [--gap 0] [--duration-frames <n>] [--output out.kdenlive] [--dry-run]
harness-kdenlive list-clips <project> [--track-id <id>] [--producer-id <id>]
harness-kdenlive resolve-clip <project> <selector> [--track-id <id>] [--at-frame <n>]
harness-kdenlive select-zone <project> [--zone-in 0] [--zone-out 0] [--output out.kdenlive]
harness-kdenlive detect-gaps <project> [--track-id <id>]
harness-kdenlive time-remap <project> <clip_ref> <speed> [--output out.kdenlive]
harness-kdenlive transform-clip <project> <clip_ref> [--geometry <str>] [--rotate <float>] [--scale <float>] [--opacity <float>] [--keyframes-json <json>] [--output out.kdenlive]
harness-kdenlive nudge-clip <project> <clip_ref> <delta_frames> [--output out.kdenlive]
harness-kdenlive slip-clip <project> <clip_ref> <delta_frames> [--output out.kdenlive]
harness-kdenlive slide-clip <project> <clip_ref> <delta_frames> [--output out.kdenlive]
harness-kdenlive ripple-insert <project> <track_id> <position> [--length <n>] [--clip-id <id>] [--in-point <n>] [--out-point <n>] [--output out.kdenlive]
harness-kdenlive group-clips <project> <clip_ref...> [--group-id <id>] [--output out.kdenlive]
harness-kdenlive ungroup-clips <project> <clip_ref...> [--output out.kdenlive]
harness-kdenlive list-sequences <project>
harness-kdenlive copy-sequence <project> <source_id> [--new-id <id>] [--output out.kdenlive]
harness-kdenlive set-active-sequence <project> <sequence_id> [--output out.kdenlive]
harness-kdenlive list-effects <project> <clip_ref>
harness-kdenlive apply-effect <project> <clip_ref> <service> [--effect-id <id>] [--properties-json <json>] [--output out.kdenlive]
harness-kdenlive update-effect <project> <clip_ref> <effect_id> <properties-json> [--output out.kdenlive]
harness-kdenlive remove-effect <project> <clip_ref> <effect_id> [--output out.kdenlive]
harness-kdenlive list-transitions <project>
harness-kdenlive apply-transition <project> [--in-frame 0] [--out-frame 0] [--service mix] [--transition-id <id>] [--properties-json <json>] [--output out.kdenlive]
harness-kdenlive remove-transition <project> <transition_id> [--output out.kdenlive]
harness-kdenlive apply-wipe <project> [--in-frame 0] [--out-frame 0] [--preset circle|clock|barn|iris|linear] [--transition-id <id>] [--softness <float>] [--invert] [--output out.kdenlive]
harness-kdenlive add-music-bed <project> <media> [--track-id playlist1] [--position 0] [--duration-frames <n>] [--producer-id <id>] [--output out.kdenlive]
harness-kdenlive duck-audio <project> <track_id> [--duck-gain 0.3] [--output out.kdenlive]
harness-kdenlive audio-fade <project> <clip_ref> [--fade-type in|out] [--frames 24] [--output out.kdenlive]
harness-kdenlive normalize-audio <project> <clip_ref> [--target-db -14] [--output out.kdenlive]
harness-kdenlive remove-silence <project> <clip_ref> [--threshold-db -35] [--min-duration-frames 6] [--output out.kdenlive]
harness-kdenlive audio-pan <project> <clip_ref> <pan> [--output out.kdenlive]
harness-kdenlive grade-clip <project> <clip_ref> [--lift <float>] [--gamma <float>] [--gain <float>] [--saturation <float>] [--temperature <float>] [--lut-path <path>] [--output out.kdenlive]
harness-kdenlive add-track <project> [--track-type video|audio] [--name <name>] [--index <n>] [--track-id <id>] [--output out.kdenlive]
harness-kdenlive remove-track <project> <track_id> [--force] [--output out.kdenlive]
harness-kdenlive reorder-track <project> <track_id> <index> [--output out.kdenlive]
harness-kdenlive resolve-track <project> <name-or-id>
harness-kdenlive track-mute <project> <track_id> [--output out.kdenlive]
harness-kdenlive track-unmute <project> <track_id> [--output out.kdenlive]
harness-kdenlive track-lock <project> <track_id> [--output out.kdenlive]
harness-kdenlive track-unlock <project> <track_id> [--output out.kdenlive]
harness-kdenlive track-show <project> <track_id> [--output out.kdenlive]
harness-kdenlive track-hide <project> <track_id> [--output out.kdenlive]
harness-kdenlive snapshot <project> <description>
harness-kdenlive render-clip <source.mp4> <output.mp4> <duration_seconds> [--start-seconds 0] [--preset-name h264|hevc|prores]
harness-kdenlive render-project <project.kdenlive> <output.mp4> [--start-seconds <float>] [--duration-seconds <float>] [--zone-in <frame>] [--zone-out <frame>] [--preset-name h264|hevc|prores]
harness-kdenlive render-status <job_id>
harness-kdenlive render-latest [--type project|clip] [--status running|completed|failed|canceled]
harness-kdenlive render-retry <job_id> [--output <path>]
harness-kdenlive render-cancel <job_id>
harness-kdenlive render-list-jobs
harness-kdenlive render-wait <job_id> [--timeout-seconds 120] [--poll-interval-seconds 0.2]
harness-kdenlive export-edl <project.kdenlive> <output.edl>
harness-kdenlive export-xml <project.kdenlive> <output.xml>
harness-kdenlive export-otio <project.kdenlive> <output.otio>
harness-kdenlive batch <steps-json> [--stop-on-error/--no-stop-on-error]
harness-kdenlive version
```

Notes:
- `add-text` automatically falls back to subtitle-sidecar mode when `qtext` is unavailable on the local MLT build.
- `render-project` automatically burns harness text overlays with `ffmpeg` when local MLT text producers are unavailable/unreliable.

## Bridge Lifecycle

```bash
harness-kdenlive bridge start [--host 127.0.0.1] [--port 41739]
harness-kdenlive bridge serve [--host 127.0.0.1] [--port 41739]   # foreground
harness-kdenlive bridge status
harness-kdenlive bridge stop
```

## Bridge Verification

Use this to verify the bridge is stable and responsive.

```bash
harness-kdenlive bridge verify [--iterations 25] [--max-failures 0]
harness-kdenlive bridge soak [--iterations 100] [--duration-seconds 5] [--action system.health]
```

Returns non-zero when stability criteria fail.
