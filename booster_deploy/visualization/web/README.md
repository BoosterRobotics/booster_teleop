# Browser visualization component

`booster_deploy.visualization.web` is the rendering boundary used by the teleoperation Dashboard.
It does not own robot control, policy state, recording, or tmux lifecycle.

## Data flow

```text
PICO/mocap HumanFrame ───────┐
reference_50hz ──────────────┼─ WebVisualizationHub ─ /ws/visualization ─ Three.js
sim/real controller snapshot ┘                    ├─ /api/visualization/model
                                                 └─ /api/visualization/geometry?v=<model>
```

- Producers and the hub are **latest-only**. A slow or disconnected browser cannot add latency to control.
- Simulation and real controllers publish local msgpack snapshots through
  `BOOSTER_WEB_VIS_ENDPOINT` (the dashboard launcher uses `tcp://127.0.0.1:5566`).
- The browser uses one renderer, one canvas, and three scissored viewports.
- Geometry arrives as **one atomic bundle** (`BVZ2`, ~12.4 MiB) rather than 32
  requests: a partial load used to leave the robot rendered as "trunk only" with
  no visible error.  The URL carries a model digest so it caches, and any failure
  shows `模型加载失败：…` instead of a silent half-robot.
- Model geometry comes from MuJoCo's compiled `mesh_vert/mesh_face/mesh_normal`
  tables (u16 indices, int8 normals), not raw STL transforms.
- Three.js r163 and OrbitControls are vendored below `static/vendor/`; no npm, build step, or CDN is required.

`/api/visualization/mesh/<id>` (single-mesh payload) is kept as a debugging aid and
is what the unit tests diff against MuJoCo's tables; the page itself always uses the
bundle.

## Files

| File | Responsibility |
|---|---|
| `store.py` | thread-safe one-frame-per-stream store and stale age |
| `protocol.py` | versioned float32 msgpack controller snapshot |
| `frames.py` | HumanFrame/reference/qpos → render payload |
| `model.py` | MuJoCo body manifest and compact compiled-mesh wire format |
| `runtime.py` | non-blocking sim/real publishers |
| `hub.py` | read-only subscribers and aggregation |
| `websocket.py` | minimal RFC6455 server frame helpers |
| `static/app.js` | Three.js scene, robot/human renderers, cameras, tracking grid, layouts |

Terminal mode intentionally keeps the legacy viser viewers. Dashboard mode does not start the three
viser panes and exposes only the existing dashboard port (default `8095`).
