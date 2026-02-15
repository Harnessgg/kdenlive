# END State: Full Kdenlive CLI Feature Surface

This document defines the complete CLI capability target for `harness-kdenlive` so an agent can perform everything a human can do in Kdenlive, end-to-end, through deterministic JSON commands.

Status legend (as of this doc update):
- `[implemented]`: available via current documented CLI/bridge command surface.
- `[partial]`: available in limited form, via adjacent command, or lacking full semantics.
- `[missing]`: not currently exposed as a first-class CLI/bridge capability.

## 1. System and Bridge (`system.*`, `bridge.*`)

- [implemented] `system.health`: health check with version, uptime, capability flags.
- [implemented] `system.actions`: discover all supported action methods, schemas, and examples.
- [partial] `system.capabilities`: return feature gates by runtime (MLT, ffmpeg, frei0r, movit, speech tools).
- [missing] `system.ping`: low-latency round-trip probe.
- [partial] `system.metrics`: request count, latency percentiles, error rates, queue depth.
- [missing] `system.logs`: structured log tail/query with correlation ids.
- [partial] `system.schema`: return command/response schema versions.
- [implemented] `bridge.start`: start local bridge process.
- [implemented] `bridge.stop`: stop bridge process.
- [implemented] `bridge.status`: running state, bind address, pid, readiness.
- [missing] `bridge.restart`: restart with same config.
- [implemented] `bridge.verify`: stability and determinism checks.
- [implemented] `bridge.soak`: repeated command execution for reliability validation.
- [missing] `bridge.configure`: set host, port, timeout, worker limits.
- [missing] `bridge.auth`: optional local token/session controls.

## 2. Project Lifecycle (`project.*`)

- [implemented] `project.create`: create new project with resolution, fps, color space, audio rate.
- [missing] `project.open`: open project for inspection/mutation.
- [missing] `project.save`: save current project.
- [missing] `project.save_as`: save copy to new path.
- [implemented] `project.clone`: clone project and referenced metadata.
- [missing] `project.close`: close active handle/session.
- [implemented] `project.inspect`: summarize tracks, clips, sequences, effects, proxies, durations.
- [implemented] `project.validate`: schema and semantic validation.
- [missing] `project.repair`: auto-fix recoverable schema/timeline issues.
- [implemented] `project.diff`: structural diff between revisions/projects.
- [implemented] `project.snapshot`: point-in-time snapshot with description.
- [missing] `project.list_snapshots`: enumerate snapshots.
- [missing] `project.restore_snapshot`: restore selected snapshot.
- [implemented] `project.undo`: undo last operation.
- [implemented] `project.redo`: redo previously undone operation.
- [implemented] `project.autosave.configure`: enable/disable autosave and interval.
- [missing] `project.lock`: cooperative lock for multi-agent safety.
- [missing] `project.unlock`: release lock.
- [implemented] `project.pack`: copy project + media into relocatable package.
- [partial] `project.relink`: relink offline/moved media paths.
- [missing] `project.consolidate`: trim/copy used media ranges only.
- [missing] `project.archive`: create long-term archive bundle.
- [missing] `project.metadata.get`: read title/author/notes/custom metadata.
- [missing] `project.metadata.set`: write project metadata.

## 3. Asset and Bin Management (`asset.*`, `bin.*`)

- [implemented] `asset.import`: import one or many media files.
- [missing] `asset.import_sequence`: import image sequence as clip.
- [missing] `asset.import_folder`: recursive import with filters.
- [missing] `asset.remove`: remove from bin (optionally keep timeline refs blocked).
- [implemented] `asset.replace`: replace media while preserving timeline usage.
- [missing] `asset.duplicate`: duplicate bin item.
- [missing] `asset.rename`: rename asset/bin entry.
- [missing] `asset.tag.add`: add tags/labels.
- [missing] `asset.tag.remove`: remove tags/labels.
- [missing] `asset.color_label.set`: set color label.
- [implemented] `asset.metadata.get`: inspect stream metadata.
- [missing] `asset.metadata.refresh`: rescan technical metadata.
- [missing] `asset.hash`: content hash for dedupe and tracking.
- [missing] `asset.transcode`: preconvert to editing codec/profile.
- [missing] `asset.proxy.create`: generate proxy media.
- [missing] `asset.proxy.remove`: remove proxy.
- [missing] `asset.proxy.toggle`: toggle proxy use project-wide or per asset.
- [missing] `asset.audio.extract`: extract audio stream to new asset.
- [missing] `asset.thumbnail.regenerate`: refresh thumbnails.
- [missing] `asset.offline.list`: list missing/offline media.
- [missing] `asset.offline.reconnect`: reconnect offline entries.
- [implemented] `bin.list`: list folders and items.
- [implemented] `bin.folder.create`: create bin folder.
- [missing] `bin.folder.rename`: rename folder.
- [missing] `bin.folder.move`: move folder hierarchy.
- [missing] `bin.folder.delete`: delete folder (with safety rules).
- [implemented] `bin.item.move`: move asset into folder.
- [missing] `bin.search`: query by name, tag, type, metadata.
- [missing] `bin.sort`: sort by name/date/type/duration/label.

## 4. Sequences and Timeline Structure (`sequence.*`, `timeline.*`)

- [implemented] `sequence.list`: list all sequences/timelines.
- [missing] `sequence.create`: create new sequence with profile and track layout.
- [implemented] `sequence.clone`: duplicate sequence.
- [missing] `sequence.rename`: rename sequence.
- [missing] `sequence.delete`: remove sequence.
- [implemented] `sequence.activate`: set active sequence.
- [missing] `sequence.settings.get`: get resolution/fps/audio/layout settings.
- [missing] `sequence.settings.set`: mutate sequence settings.
- [implemented] `timeline.inspect`: return full clip/composition layout.
- [implemented] `timeline.bounds.recalculate`: recalc sequence bounds/duration.
- [missing] `timeline.timecode.convert`: frame/timecode conversion utilities.
- [implemented] `timeline.zone.set`: set in/out zone.
- [missing] `timeline.zone.clear`: clear zone.
- [missing] `timeline.marker.add`: add timeline marker.
- [missing] `timeline.marker.update`: edit marker name/color/comment.
- [missing] `timeline.marker.remove`: remove marker.
- [missing] `timeline.marker.list`: list markers.
- [missing] `timeline.guides.add`: add guide.
- [missing] `timeline.guides.remove`: remove guide.
- [missing] `timeline.guides.list`: list guides.
- [missing] `timeline.ruler.set_display`: frames/timecode/drop-frame modes.
- [missing] `timeline.snap.enable`: enable snapping.
- [missing] `timeline.snap.disable`: disable snapping.
- [missing] `timeline.ripple.enable`: enable ripple mode.
- [missing] `timeline.ripple.disable`: disable ripple mode.

## 5. Track Management (`track.*`)

- [implemented] `track.list`: list tracks with type/state/order.
- [implemented] `track.add`: add video/audio/subtitle track.
- [implemented] `track.remove`: remove track with safeguards.
- [implemented] `track.reorder`: move track index.
- [missing] `track.rename`: rename track.
- [implemented] `track.resolve`: resolve by id/name/index.
- [implemented] `track.mute`: mute audio track.
- [implemented] `track.unmute`: unmute audio track.
- [missing] `track.solo`: solo track.
- [missing] `track.unsolo`: clear solo.
- [implemented] `track.lock`: lock track from edits.
- [implemented] `track.unlock`: unlock track.
- [implemented] `track.hide`: hide video track output.
- [implemented] `track.show`: show video track output.
- [partial] `track.enable`: enable track processing.
- [partial] `track.disable`: disable track processing.
- [missing] `track.height.set`: set visual height metadata.
- [missing] `track.compositing_mode.set`: set blend/composite behavior.
- [missing] `track.target.set`: set target tracks for insert/overwrite operations.

## 6. Clip Editing and Timeline Operations (`timeline.clip.*`)

- [implemented] `timeline.clip.add`: place clip on track at position.
- [partial] `timeline.clip.insert`: insert with ripple.
- [missing] `timeline.clip.overwrite`: overwrite range without ripple.
- [implemented] `timeline.clip.move`: move clip between tracks/positions.
- [implemented] `timeline.clip.nudge`: move by delta frames.
- [implemented] `timeline.clip.trim`: set in/out points.
- [missing] `timeline.clip.trim_start`: trim head by absolute or delta.
- [missing] `timeline.clip.trim_end`: trim tail by absolute or delta.
- [implemented] `timeline.clip.slip`: change source in/out while keeping position.
- [implemented] `timeline.clip.slide`: move clip while preserving neighbors.
- [implemented] `timeline.clip.split`: razor cut at frame.
- [implemented] `timeline.clip.remove`: remove clip.
- [implemented] `timeline.clip.ripple_delete`: delete and close gap.
- [missing] `timeline.clip.lift`: remove without ripple.
- [missing] `timeline.clip.extract`: remove zone and ripple close.
- [missing] `timeline.clip.copy`: copy selected clips.
- [missing] `timeline.clip.paste`: paste at playhead/position.
- [missing] `timeline.clip.duplicate`: duplicate in place or offset.
- [missing] `timeline.clip.reverse`: reverse playback.
- [partial] `timeline.clip.speed.set`: set speed/time remap.
- [missing] `timeline.clip.freeze_frame`: create still segment.
- [missing] `timeline.clip.enable`: enable clip.
- [missing] `timeline.clip.disable`: disable clip.
- [implemented] `timeline.clip.group`: group clips.
- [implemented] `timeline.clip.ungroup`: ungroup clips.
- [missing] `timeline.clip.link_av`: link audio/video components.
- [missing] `timeline.clip.unlink_av`: unlink audio/video components.
- [missing] `timeline.clip.select`: select by ids/query/time range.
- [implemented] `timeline.clip.resolve`: resolve selector to concrete clip refs.
- [implemented] `timeline.clip.list`: list clips with timing and refs.
- [implemented] `timeline.gap.detect`: detect all gaps.
- [implemented] `timeline.gap.insert`: insert gap at position.
- [missing] `timeline.gap.remove`: remove specific gap.
- [implemented] `timeline.gap.remove_all`: remove all gaps on track/sequence.
- [missing] `timeline.align.to_playhead`: align selection to playhead.
- [implemented] `timeline.stitch`: stitch clips with optional gaps/transitions.
- [missing] `timeline.paste_attributes`: copy/paste effects and transforms.

## 7. Compositions, Transitions, and Blending (`timeline.transition.*`)

- [implemented] `timeline.transition.list`: list transitions/compositions in sequence.
- [implemented] `timeline.transition.add`: add transition by service/preset.
- [partial] `timeline.transition.update`: mutate in/out/duration/properties.
- [implemented] `timeline.transition.remove`: remove transition.
- [missing] `timeline.transition.move`: reposition transition.
- [missing] `timeline.transition.duplicate`: duplicate transition.
- [implemented] `timeline.transition.wipe.apply`: apply wipe preset.
- [missing] `timeline.transition.default.set`: set default transition style.
- [missing] `timeline.composition.add`: add composition clip spanning tracks.
- [missing] `timeline.composition.update`: update properties/keyframes.
- [missing] `timeline.composition.remove`: remove composition.

## 8. Effects, Filters, and Keyframes (`effect.*`)

- [partial] `effect.catalog.list`: list all available effects with parameter schema.
- [missing] `effect.search`: search effect catalog.
- [missing] `effect.favorite.add`: mark favorite.
- [missing] `effect.favorite.remove`: unmark favorite.
- [missing] `effect.favorite.list`: list favorites.
- [implemented] `effect.stack.list`: list applied effects on target.
- [implemented] `effect.stack.add`: apply effect.
- [implemented] `effect.stack.remove`: remove effect.
- [implemented] `effect.stack.update`: set multiple properties.
- [missing] `effect.stack.enable`: enable effect.
- [missing] `effect.stack.disable`: disable effect.
- [missing] `effect.stack.reorder`: reorder effect stack.
- [missing] `effect.stack.copy`: copy effect stack.
- [missing] `effect.stack.paste`: paste stack to target.
- [missing] `effect.preset.save`: save custom preset.
- [missing] `effect.preset.load`: load preset onto target.
- [missing] `effect.preset.delete`: delete preset.
- [missing] `effect.parameter.get`: read current parameter values.
- [missing] `effect.parameter.set`: set scalar parameter.
- [missing] `effect.parameter.reset`: reset to default.
- [implemented] `effect.keyframe.set`: set keyframes in bulk.
- [missing] `effect.keyframe.add`: add single keyframe.
- [missing] `effect.keyframe.update`: update keyframe value/interpolation.
- [missing] `effect.keyframe.remove`: remove keyframe.
- [missing] `effect.keyframe.clear`: clear all keyframes for param.
- [missing] `effect.keyframe.list`: list keyframes and interpolation.
- [missing] `effect.mask.attach`: attach mask asset/effect.
- [missing] `effect.mask.detach`: detach mask.

## 9. Text, Titles, and Subtitles (`text.*`, `subtitle.*`)

- [implemented] `text.title.create`: create title clip.
- [implemented] `text.title.update`: update text, style, font, color, shadow, outline.
- [missing] `text.title.layout`: set anchor, alignment, safe margins.
- [missing] `text.title.animate`: add transform/opacity keyframes.
- [missing] `text.template.apply`: apply saved title template.
- [missing] `subtitle.track.create`: create subtitle track.
- [missing] `subtitle.track.remove`: remove subtitle track.
- [missing] `subtitle.add`: add subtitle cue.
- [missing] `subtitle.update`: edit subtitle cue.
- [missing] `subtitle.remove`: remove subtitle cue.
- [missing] `subtitle.list`: list subtitle cues.
- [missing] `subtitle.import_srt`: import SRT/ASS/VTT.
- [missing] `subtitle.export_srt`: export subtitles.
- [missing] `subtitle.style.set`: set subtitle style defaults.
- [partial] `subtitle.burn.enable`: configure burn-in at render.
- [partial] `subtitle.burn.disable`: disable burn-in.
- [missing] `caption.auto.generate`: speech-to-text subtitle generation.
- [missing] `caption.auto.align`: align subtitles to waveform.

## 10. Audio Editing and Mixing (`audio.*`)

- [missing] `audio.waveform.build`: ensure waveform cache exists.
- [missing] `audio.levels.analyze`: loudness and peak analysis.
- [implemented] `audio.normalize`: normalize clip/track/sequence.
- [partial] `audio.gain.set`: set clip or track gain.
- [implemented] `audio.pan.set`: set stereo pan.
- [missing] `audio.balance.set`: set channel balance.
- [missing] `audio.channel.map`: map channel layout (mono/stereo/5.1).
- [implemented] `audio.fade.in`: apply fade-in.
- [implemented] `audio.fade.out`: apply fade-out.
- [missing] `audio.crossfade.add`: add crossfade between adjacent clips.
- [implemented] `audio.duck`: duck background audio from dialog sidechain.
- [implemented] `audio.silence.remove`: detect and cut silent regions.
- [missing] `audio.noise.reduce`: apply denoise effect preset.
- [missing] `audio.eq.apply`: apply EQ presets/params.
- [missing] `audio.compressor.apply`: apply dynamics control.
- [missing] `audio.limiter.apply`: apply limiter.
- [missing] `audio.reverb.apply`: apply room/reverb.
- [missing] `audio.pitch.shift`: pitch correction or creative pitch shift.
- [missing] `audio.speed.change`: time-stretch with pitch options.
- [missing] `audio.sync.by_waveform`: sync clips using waveform correlation.
- [missing] `audio.sync.by_timecode`: sync via LTC/VITC/metadata timecode.
- [missing] `audio.monitor.meter`: query current mixer meters.
- [missing] `audio.bus.create`: create mix bus.
- [missing] `audio.bus.route`: route tracks to bus.
- [missing] `audio.bus.remove`: remove bus.

## 11. Color and Image Operations (`color.*`)

- [implemented] `color.grade.basic`: lift/gamma/gain/saturation/temperature/tint.
- [missing] `color.lut.apply`: apply LUT file.
- [missing] `color.lut.remove`: remove LUT.
- [missing] `color.curves.set`: set RGB/luma curves.
- [missing] `color.wheels.set`: set shadows/mids/highlights wheels.
- [missing] `color.hsl.secondary`: isolate and grade hue range.
- [missing] `color.white_balance.auto`: auto white balance.
- [missing] `color.exposure.auto`: auto exposure normalize.
- [missing] `color.match.shot`: match source shot to reference.
- [missing] `color.scope.waveform`: query/analyze waveform data.
- [missing] `color.scope.vectorscope`: query/analyze vectorscope data.
- [missing] `color.scope.histogram`: query/analyze histogram data.
- [missing] `color.safe.broadcast`: apply legal range clamp/check.

## 12. Motion, Transform, and Compositing (`motion.*`)

- [implemented] `motion.transform.set`: position/scale/rotation/opacity.
- [implemented] `motion.transform.keyframe`: animated transform keyframes.
- [missing] `motion.crop.set`: crop values.
- [missing] `motion.rotate.set`: rotation and anchor.
- [missing] `motion.stabilize`: run stabilization analysis/apply.
- [missing] `motion.tracking.run`: run motion tracking.
- [missing] `motion.tracking.apply`: apply tracking data to effects/text.
- [missing] `motion.blend_mode.set`: set blend mode.
- [missing] `motion.chroma_key.apply`: apply chroma/luma keying.
- [missing] `motion.matte.attach`: attach alpha matte.
- [missing] `motion.mask.animate`: animate masks.

## 13. Monitoring, Playback, and QC (`monitor.*`, `qc.*`)

- [missing] `monitor.playhead.get`: get current playhead.
- [missing] `monitor.playhead.set`: set playhead.
- [missing] `monitor.jog`: jog/shuttle by speed/delta.
- [missing] `monitor.frame.step`: frame forward/back.
- [missing] `monitor.preview.range`: render preview range cache.
- [missing] `monitor.preview.clear`: clear preview cache.
- [missing] `monitor.thumbnail.at`: generate frame thumbnail at time.
- [missing] `qc.offline_assets`: report unresolved media.
- [missing] `qc.missing_effects`: report unavailable effect services.
- [missing] `qc.render_readiness`: preflight for render job.
- [missing] `qc.timeline_issues`: overlaps, invalid refs, bounds errors.
- [missing] `qc.report.generate`: output comprehensive QC report.

## 14. Export and Interchange (`export.*`, `import.*`)

- [implemented] `export.project.render`: render full project.
- [missing] `export.project.render_zone`: render zone-only.
- [implemented] `export.clip.render`: render source clip segment.
- [missing] `export.sequence.render`: render selected sequence.
- [partial] `export.preset.list`: list encoding presets.
- [missing] `export.preset.inspect`: inspect preset parameters.
- [missing] `export.preset.create`: create custom preset.
- [missing] `export.preset.update`: update preset.
- [missing] `export.preset.delete`: delete preset.
- [implemented] `export.job.status`: query render job.
- [implemented] `export.job.list`: list render jobs.
- [implemented] `export.job.wait`: wait for completion.
- [implemented] `export.job.cancel`: cancel running job.
- [implemented] `export.job.retry`: retry failed job.
- [missing] `export.thumbnail`: export still frame.
- [missing] `export.audio_only`: export WAV/FLAC/MP3 mixdown.
- [missing] `export.stems`: export per-track/per-bus stems.
- [implemented] `export.edl`: export EDL.
- [implemented] `export.xml`: export XML interchange.
- [implemented] `export.otio`: export OTIO.
- [missing] `import.edl`: import EDL to sequence.
- [missing] `import.xml`: import XML to sequence.
- [missing] `import.otio`: import OTIO to sequence.
- [missing] `export.report`: render manifest with settings, hashes, timing.

## 15. Batch, Automation, and Transactions (`batch.*`, `transaction.*`)

- [implemented] `batch.run`: execute ordered command list.
- [missing] `batch.validate`: static validation of batch script.
- [missing] `batch.dry_run`: simulate batch without mutation.
- [missing] `batch.resume`: resume from failed step.
- [missing] `batch.cancel`: stop running batch.
- [missing] `batch.status`: progress, per-step result, timing.
- [missing] `transaction.begin`: start explicit transaction.
- [missing] `transaction.commit`: commit transaction.
- [missing] `transaction.rollback`: rollback transaction.
- [partial] `transaction.preview_diff`: show pending changes before commit.
- [missing] `macro.record`: record executed actions.
- [missing] `macro.play`: replay saved macro.
- [missing] `macro.list`: list available macros.

## 16. Determinism, Safety, and Error Contract (Cross-cutting)

- Every command returns exactly one JSON object.
- Stable response envelope for success and error paths.
- Structured error codes with remediation hints.
- Deterministic ordering for list outputs.
- Explicit idempotency semantics per action.
- [missing] `dry_run` support for all mutating actions.
- [missing] `validate_before_write` option for all destructive edits.
- [missing] `confirm_destructive` guard for high-risk operations.
- Path normalization and sandbox-aware path policy.
- Correlation id and causation id for each action.
- Full action-level audit log.
- Machine-readable warning channel distinct from errors.
- Versioned schemas for commands and responses.
- Backward compatibility policy and deprecation metadata.

## 17. Testing and Quality Requirements (Cross-cutting)

- Unit tests for every action method.
- Bridge integration tests for every CLI command.
- Golden JSON output tests for determinism.
- Schema conformance tests against response schema.
- Round-trip tests for project load/save integrity.
- Fuzz tests for malformed XML/project inputs.
- Performance budgets for common actions.
- Soak tests and flaky-test quarantine policy.
- Compatibility matrix tests across Kdenlive/MLT versions.
- Docs sync checks to ensure command surface matches docs.

## 18. Documentation and Discoverability (Cross-cutting)

- Human docs for every command with examples.
- LLM docs with strict command grammar and JSON schema.
- Machine-readable action registry with parameter schemas.
- Migration notes for any command/shape changes.
- Quickstart workflows for common editing pipelines.
- Troubleshooting playbooks for bridge/render failures.

## Definition of Done for Full Parity

The package reaches end-state parity when an agent can execute complete real-world editing workflows (ingest, organize, edit, mix, grade, subtitle, QC, render, export, archive) with no GUI fallback, using only documented CLI/bridge actions and deterministic JSON responses.
