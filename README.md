# DEEP OCEAN ROBOT VISION - SONAR 4D

An underwater robotic-perception installation: a robot surveys **real NOAA bathymetry of Monterey Canyon** with a
**simulated** multibeam/forward-looking sonar, navigates with dead reckoning, corrects itself with **terrain-based pose-graph
SLAM**, remembers every scan in world space, and the whole thing is rendered in **TouchDesigner** as a neon / fluorescent
LED light show (scientific and cinematic modes).

> **Honest scope.** Only the *terrain* is real data (NOAA NCEI CUDEM 1/9 arc-second tiles). No raw multibeam / sonar
> sequence with calibration and navigation could be verified and downloaded here, so every sonar, DVL, AHRS, pressure
> value, the navigation error and the four inserted seabed objects are **SIMULATED** (Source E fallback) and labelled that
> way in every output, the manifest and the on-screen diagnostics.

## What is real and what is simulated

| Item | Status |
|---|---|
| Seabed elevation grid (survey tile 2.4 x 3.3 km @ 3.5 m, overview 17.8 x 16.6 km @ 17.5 m) | **REAL** NOAA NCEI CUDEM, EPSG:4326, NAVD88 |
| Multibeam soundings, FLS images, DVL / AHRS / depth, navigation drift | **SIMULATED** (ray-cast over the real terrain, physical noise models) |
| 4 seabed objects (14 x 9 m, 7.6-8.9 m tall) | **SYNTHETIC**, inserted into the truth terrain for validation |
| SLAM, mapping, obstacle candidates, occupancy, temporal memory | real algorithms, run on the simulated sequence |
| Evaluation numbers below | computed against the *simulated* ground truth |

## Verified results (this machine, this run)

| Stage | Result |
|---|---|
| Real bathymetry | 953 x 687 metric grid @ 3.5 m, elevation -1031.7 ... -565.4 m, 100 % valid, 326 k triangles; geographic -> local TM metres (tested against WGS84 geodesics, < 0.02 %) |
| Simulated survey | 6 912 pings x 96 beams (663 552 soundings), 34.5 km track, 13 833 s, terrain-following at 70 m altitude |
| Sonar processing | 553 362 soundings kept (98.4 % of detections); isolated spikes rejected, coherent relief preserved |
| Dead reckoning | horizontal error mean 6.7 m, max 29.8 m (heading bias + walk, DVL scale error, 2 bottom-lock dropouts) |
| SLAM | 1 127 keyframes, 833 loop candidates -> **501 verified loop closures** (ambiguous / asymmetric / low-overlap rejected: 332). Keyframe **ATE RMSE 8.95 -> 5.24 m**, mean 6.74 -> 3.92 m, max 29.8 -> 20.1 m |
| Map vs truth terrain (observed cells) | elevation RMSE **1.89 m (DR) -> 1.45 m (SLAM)**, coverage 87 % of the planned survey rectangle |
| Obstacle candidates | all 4 inserted objects recovered (recall 4/4) but **precision is poor**: 149 other compact relief candidates (natural features), none given a semantic label |
| Occupancy (10-20 m voxels) | 94 % unknown, 4.8 % free, 1.2 % occupied - unobserved water is never "free" |
| Learning (optional) | relief-patch auto-encoder: val MSE 0.0017 vs 0.086 zero-relief baseline; k-means terrain *regimes* (not semantic classes), silhouette 0.37 |
| Bridge | loopback stream with shuffled + dropped packets: 111 of 120 delivered strictly in order (7 dropped by design, 2 late) |
| Film | `output/videos/deep_ocean_robot_vision_4d.mp4`: 30.0 s, 900 frames, 1280x720 @ 30 fps H.264 (7 scenes per the storyboard; 1080p/4K needs a commercial licence) |
| TouchDesigner | project builds with 0 warnings; 0 GLSL errors; 17-36 fps at 1280x720 on integrated Intel UHD 620, 122 MB of GPU textures |
| Tests | see "Tests" below |

Things that did **not** get better: a single loop closure does not guarantee lower error against truth (see
`tests/test_pose_graph.py`), residual-based loop *screening* removed exactly the closures that needed the largest
corrections and made SLAM useless, so it is only a diagnostic now.

## Quick start

```bash
pip install -r requirements.txt
python acquisition/download_bathymetry.py --config config.yaml   # real NOAA tiles -> data/bathymetry
python scripts/inspect_dataset.py        --config config.yaml   # sources, tile inspection, licence, manifest
python scripts/build_bathymetry.py       --config config.yaml   # metric grids, meshes, height textures
python scripts/simulate_auv.py           --config config.yaml   # SIMULATED survey (labelled) + FLS images
python scripts/preprocess_sonar.py       --config config.yaml   # filtered world soundings (dead-reckoned)
python scripts/run_slam.py               --config config.yaml   # keyframes, loop closures, pose graph, versions
python scripts/build_map.py              --config config.yaml   # grids, mesh, obstacles, occupancy, TD export
python scripts/start_bridge.py           --config config.yaml [--selftest]
python -m pytest tests/
```
`scripts/run_slam.py --reuse-loops` reuses the cached loop detection (the slow part, ~7-20 min on a laptop).

### TouchDesigner

1. Open TouchDesigner (non-commercial licence is enough), open the Textport (Alt+T).
2. Paste and run:
   ```python
   DORV_DIR = r'C:\Users\Utsav\DEEP_OCEAN_ROBOT_VISION_4D'
   exec(open(DORV_DIR + r'\touchdesigner\build_project.py', encoding='utf-8').read())
   ```
   This (re)creates `/project1/DEEP_OCEAN_ROBOT_VISION_4D` (idempotent: the old network is destroyed first) and saves
   `touchdesigner/Deep_Ocean_Robot_Vision_4D.toe` (set `DORV_SAVE = False` first to skip saving).
3. Open the control panel (`UI_CONTROLS/control_window`) and the output window (`OUTPUT/output_window`).
   Keyboard in free-fly mode: W/A/S/D move, Q/E up/down, Shift fast, left mouse = look, wheel = speed.
4. Film: `python scripts/export_video.py --port <your MCP port>` renders the 30 s storyboard frame by frame and encodes H.264
   with ffmpeg.

Note: `build_project.py` needs `output/pointclouds/` (run `scripts/build_map.py` first). A bridge on port 9990 is how the
scripts in `touchdesigner/tools/` (`tdrun.py`, `shot.py`, `release_data.py`) talk to my TouchDesigner instance; it is not
needed to use the project.

## Architecture

```
acquisition/      sources, NOAA download, licence + dataset validation, manifest
preprocessing/    loaders, calibration, sync, coordinate frames, sound speed, beam geometry, filters
sonar/            multibeam, FLS, side-scan adapter, uncertainty, bottom detection, pipeline
navigation/       IMU, DVL, depth sensor models, dead reckoning, pose interpolation, trajectory
slam/             features/keyframes, NCC + ICP scan matching, loop closure, pose graph, optimiser, reintegration
mapping/          fusion, bathymetry grid, mesh, normals, contours, occupancy, obstacles, uncertainty, export
temporal_4d/      history buffers, scrubber, echoes, time explosion, map versions, TD export
simulation/       terrain ray-casting, survey planning, sensor + navigation error models (SIMULATED)
learning/         optional PyTorch relief auto-encoder + unsupervised terrain regimes
shaders/          13 GLSL files (neon palette), touchdesigner/  builder + runtime modules, optional_ros2/
```

### Coordinate frames (all tested in `tests/test_coordinates.py`)

| Frame | Axes |
|---|---|
| Geographic | WGS84 lon/lat, elevation m (up) |
| Local ENU (world) | transverse Mercator k=1 on WGS84, x East, y North, z Up (sea surface ~0) |
| NED | attitude convention only (`R_ENU_NED`) |
| Vehicle / sonar | FRD: x forward, y starboard, z down; `T_vehicle_sonar` mounts the head |
| TouchDesigner | x = East, y = Up - z_ref, z = -North (z_ref recorded in `td_meta.json`) |

`P_world = T_world_vehicle * T_vehicle_sonar * P_sonar`. Planar beam model `x = r sin(theta)`, `z = r cos(theta)` is used
only as the flat-array special case of `preprocessing/beam_geometry.py`. Two-way travel time is converted with a
Mackenzie sound-speed profile (`r = c dt / 2`, harmonic-mean `c`); calibrated ranges are never converted twice.

### SLAM

Keyframe every 30 m with a +-60 m submap -> candidate pairs by acoustic *footprint overlap* -> masked NCC of bathymetric
rasters (TERCOM-like, +-60 m) -> trimmed point-to-plane ICP about the patch centre -> verification: ambiguity ratio,
overlap, inlier fraction, RMSE, predicted translation sigma (observability), 5-start basin test, forward/backward
symmetry and a navigation gate. Pose graph: odometry + sonar-odometry + loop edges with depth, gravity and weak compass
priors, robust (Huber) least squares on SE(3) (own solver: GTSAM is not installed). Corrections are re-integrated into
every ping and sounding (`slam/map_reintegration.py`). If fewer than 5 loops are verified the pipeline reports
`slam_available = false` and the map is built from navigation only.

### Visual design (neon / fluorescent / LED)

Fully saturated ramps (hot magenta -> UV violet -> electric blue -> cyan -> acid green -> yellow), additive LED point
sprites with white-hot cores, glowing iso-lines and LED mesh on a near-black violet seabed, laser-curtain beam fan,
chase-light trajectory strips, plankton sparkle, scan-lines, chromatic fringe and bloom. **SCIENTIFIC** keeps colours a pure
function of measured attributes (no hue cycling, echo flashes, wavefront rings, sparkle or scan-lines); **CINEMATIC** adds
those and they are never presented as measurements. Vertical exaggeration and the vehicle scale (14x) are visualisation
settings.

### Controls

Layer toggles (sonar, beams, real terrain, reconstruction, wire mesh, vehicle, trajectory, DR path, simulated truth,
navigation uncertainty, loop closures, objects, simulated objects, map version, contours, LED grid, atmosphere, bloom),
sliders (point size, opacity, acoustic gain, max sonar range, beam fan shown, scan history 5-500, temporal decay, vertical
exaggeration, depth colour scale, camera speed, playback speed, glow, density, DR<->SLAM mix), playback (play, pause,
reverse, step +-, freeze time, reset, jump to scan, jump to map version), six cameras (vehicle POV, chase, orbit, free-fly,
overhead, cinematic), colour modes (intensity, depth, age, scan bands, uncertainty), temporal modes (off, echo, layers,
time explosion), and a live diagnostics block.

## Limitations (read before trusting anything)

* The sonar is simulated; real multibeam/FLS/DVL formats are only covered by loaders and adapters (`load_mblist_ascii`,
  ROS 2 helpers) that are unit-tested on synthetic input. **No real MB-System file and no ROS 2 topic has ever been
  processed.** `optional_ros2/` has never run (no rclpy here).
* The forward-looking-sonar simulator and `slam/sonar_features.fls_features` exist and are tested, but the SLAM baseline
  uses multibeam bathymetry only; FLS registration is not part of the pipeline.
* Navigation drift is a model I chose; absolute numbers depend on it. The SLAM gain (8.95 -> 5.24 m ATE RMSE) is
  modest because DR is already good relative to the 150 m line spacing and the loop edges (1-3 m error each) are far
  noisier than the odometry.
* Obstacle precision is poor; objects are unlabelled geometry, not classified.
* The 12 GB / 1080p / 30 fps targets were not met as stated: development hardware is integrated graphics, the free TD
  licence caps TOPs at 1280 px, 17-36 fps was measured at 1280x720. 1080p/4K needs a commercial licence.
* The reconstructed seabed shows observed cells only; the (small-gap) interpolated grid is exported separately and not
  rendered as measured.
* No live-camera/keyboard/mouse hardware test beyond synthetic injection; free-fly input was verified by unit tests only.

## Tests

`python -m pytest tests/` - dataset integrity, coordinates, bathymetry, sound speed, sonar geometry and uncertainty,
navigation, scan matching, pose graph, mapping, temporal memory, bridge, learning/ROS 2 helpers, and (when my TouchDesigner
instance is running) operator creation, texture loading, shader compilation, camera switching, timeline and video-export
configuration. Live-TD tests are *skipped*, not passed, when TouchDesigner is unreachable.
