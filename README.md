<div align="center">

# PsySense

**Real-time classroom/room behavioral analysis** — face-based identity, emotion, and
posture signals, fused over time and reviewed by a human before anything about a
person's profile changes.

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/frontend-React%20%2B%20Vite-61DAFB?logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/typed-TypeScript-3178C6?logo=typescript&logoColor=white)
![SQLite](https://img.shields.io/badge/storage-SQLite%20(WAL)-07405E?logo=sqlite&logoColor=white)
![Tests](https://img.shields.io/badge/backend%20tests-65%20passing-2ea043)

</div>

---

PsySense watches a camera feed, detects and tracks people, resolves who they are
from a face-embedding profile, samples emotion and posture signals, and fuses all
of that into windowed behavioral events — not raw video, not per-frame judgments.
A backend API and a React dashboard sit on top so a teacher/admin can actually see
and act on the data, enroll people, and review anything the identity system wants
to learn automatically before it's allowed to.

This README describes the system **as it actually exists in this repository** —
if anything here goes stale, the code is authoritative.

## Contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Running it](#running-it)
- [Testing](#testing)
- [Deployment](#deployment)
- [Privacy & security posture](#privacy--security-posture)
- [Known limitations / roadmap](#known-limitations--roadmap)
- [Troubleshooting](#troubleshooting)

## Highlights

- **Decoupled identity** — a tracker ID is a short-lived, frame-to-frame hint;
  `student_id` (resolved by ArcFace embedding search, FAISS or linear-scan
  fallback) is the only identity that ever gets written to the database. Losing
  and regaining track of someone doesn't create a new person.
- **Independent-frequency pipeline** — detection/tracking run every frame;
  identity resolution, emotion sampling, and pose sampling each run on their own
  configurable interval per tracked person, not in lockstep.
- **Graceful degradation everywhere** — if the emotion or pose service is down,
  the pipeline keeps running with that signal marked unavailable rather than
  crashing or fabricating a value.
- **Human-in-the-loop profile enrichment** — a face is only ever added to
  someone's identity profile in two ways: (1) an admin explicitly enrolling
  photos, or (2) the pipeline getting an *extremely* confident match, queuing it,
  and a human approving or rejecting the actual photo in the dashboard's Review
  page. Nothing updates a profile silently, and a rejected candidate leaves no
  trace — image and embedding both discarded.
- **Scoped reviewer accounts** — beyond the root admin, sub-admin "reviewer"
  accounts can be created and assigned to specific students; a reviewer can only
  see and act on candidates for the people they're assigned to, enforced
  server-side, not just hidden in the UI.
- **Typed, tested backend** — Pydantic config/domain models throughout, a single
  async DB writer serializing all SQLite writes (WAL mode, so readers never
  block), and 65 backend tests that run without a GPU, camera, or real ML models
  (fakes stand in for InsightFace/DeepFace/MediaPipe).

## Architecture

```mermaid
flowchart TD
    CAM["Camera (cv2.VideoCapture)"] --> YOLO["YOLO person detection"]
    YOLO --> TRACK["Tracker (ByteTrack | StrongSort)"]
    TRACK --> SCHED["main.py per-track scheduler<br/>(each stage on its own interval)"]

    SCHED --> IDMGR["IdentityManager<br/>(ArcFace embed + FAISS/linear search)"]
    SCHED --> DEEPFACE["deepface_server.py<br/>emotion, HTTP :8000"]
    SCHED --> MEDIAPIPE["mediapipe_server.py<br/>pose, HTTP :8001"]

    IDMGR --> FUSION["FusionEngine<br/>windowed averaging + behavior flags"]
    DEEPFACE --> FUSION
    MEDIAPIPE --> FUSION

    IDMGR -. "very high-confidence match" .-> PENDING[("pending_identity_candidates<br/>(raw photo + embedding)")]

    FUSION --> REPO["Repository"]
    REPO --> WRITER["AsyncDBWriter<br/>(single writer, WAL)"]
    WRITER --> DB[("SQLite")]
    PENDING --> DB

    DB --> API["bridge_api (FastAPI)<br/>JWT auth, admin + reviewer roles"]
    API --> WEB["React dashboard (Vite)"]
    WEB -->|"approve / reject"| API
    API -->|"approved candidate"| DB
```

See `psysense/services/identity/manager.py`'s docstring and
`psysense/tests/test_identity_manager.py::test_identity_decoupled_from_tracker_id`
for how tracker-ID decoupling is actually enforced and tested.

## Repository layout

```
psysense/
  main.py                    camera pipeline entrypoint
  deepface_server.py         emotion analysis HTTP service (port 8000)
  mediapipe_server.py        pose analysis HTTP service (port 8001)
  pose_analyzer.py           landmark -> pose-cue math, used by mediapipe_server
  utils.py                   image crop/encode helpers
  core/                      config loading, domain models, interfaces, exceptions
  db/                        schema, AsyncDBWriter (single writer, WAL), Repository
  services/
    identity/                 ArcFace embedder, quality gate, vector index, IdentityManager
    fusion/                   windowed averaging + behavior-flag engine
    tracking/                 ByteTrack/StrongSort wrapper
  api/                        bridge_api: FastAPI app, auth (JWT + roles), routers, schemas
    routers/                   auth, students, telemetry, identity-candidates, users
  scripts/create_admin.py     generates ADMIN_PASSWORD_HASH/JWT_SECRET for .env
  tests/                      pytest suite -- pipeline + full API, no real models required
  benchmarks/                 tracker A/B benchmark script
  config/config.yaml          all non-secret tunables
frontend/                    React + Vite + TypeScript dashboard
  src/api/                     fetch client, typed endpoint wrappers, mirrored types
  src/auth/                    JWT + role session context, route guards
  src/components/               shared UI (charts, dropzone, layout, badges)
  src/pages/                    one file per route
docker-compose.yml           bridge_api container definition
requirements*.txt            per-service Python dependencies (see Setup)
```

## Setup

### Python environments

Three separate venvs, historically kept apart because their dependency chains
conflict with each other:

| venv | installs from | runs |
|---|---|---|
| `main_env` | `requirements.txt` | `main.py` (camera pipeline) and `bridge_api` |
| `deepface_env` | `requirements-deepface.txt` | `deepface_server.py` |
| `mediapipe_env` | `requirements-mediapipe.txt` | `mediapipe_server.py` |

```powershell
python -m venv main_env
main_env\Scripts\activate
pip install -r requirements.txt
```

(repeat per venv with its own requirements file, from the repo root)

Python 3.10+ — InsightFace/FAISS wheel availability is the main constraint on
very new Python versions.

### Model files (not committed — see `.gitignore`)

- `yolov8n.pt` — YOLO person detector
- `osnet_x0_25_msmt17.pt` / `.onnx` — StrongSort ReID weights (only needed if
  `pipeline.tracker.kind: "strongsort"` in config.yaml; ByteTrack is the default
  and needs neither)
- `pose_landmarker_heavy.task` — MediaPipe pose model

InsightFace's `buffalo_l` model pack downloads itself on first use of the
identity subsystem — no manual step, but it needs network access the first time.

### Configuration

- `psysense/config/config.yaml` — every non-secret tunable: ports, camera
  source, quality/matching thresholds, human-review auto-update thresholds,
  intervals, database path, CORS origin, log level. Never put secrets here.
- `.env` (gitignored; copy `.env.example`) — only `ADMIN_USERNAME`,
  `ADMIN_PASSWORD_HASH`, `JWT_SECRET`, generated with:

  ```powershell
  cd psysense
  python -m scripts.create_admin
  ```

  This is the root admin — a break-glass account that always works. Additional
  "reviewer" accounts (scoped to specific students) are created afterward from
  the dashboard's Manage Reviewers page, not via environment variables.

### Frontend

```powershell
cd frontend
npm install
```

## Running it

Each of these is a separate process (separate venv where noted):

```powershell
# deepface_env
python psysense\deepface_server.py

# mediapipe_env
python psysense\mediapipe_server.py

# main_env -- camera pipeline
cd psysense
python main.py

# main_env -- backend API (.env loads automatically -- see api/asgi.py)
cd psysense
python -m uvicorn api.asgi:app --host 0.0.0.0 --port 8080

# frontend dev server
cd frontend
npm run dev
```

`main.py` does not open a display window by default
(`pipeline.display.headless` in config.yaml) — set it to `false` for a debug
overlay (bounding boxes, resolved identity, live emotion/pose readout) during
development.

The frontend is pinned to port `5173` (`--strictPort`, matching
`config.yaml`'s `api.frontend_url`) — if that port is taken, `npm run dev`
fails loudly instead of silently landing on a different port and breaking CORS.

## Testing

```powershell
cd psysense
python -m pytest -q
```

All 65 tests run without real ML models, a camera, or GPU — the identity/API
tests use fakes (`tests/test_identity_manager.py`'s `FakeEmbedder`,
`FakeQualityAssessor`) against a real temp-file SQLite database, so they still
exercise real persistence, concurrency, and role-authorization logic, just not
real inference.

```powershell
cd frontend
npm run build     # tsc -b && vite build -- this is also the type-check
```

## Deployment

**bridge_api runs on the same machine as the camera pipeline**, reading the same
SQLite file `main.py` writes. This was a deliberate choice over a separate cloud
server with a sync process — it avoids inventing a second database and a sync
engine for a single-deployment setup. Expose bridge_api to wherever the deployed
frontend runs via a reverse proxy or tunnel rather than opening the port
directly to the internet.

### Backend (Docker)

```powershell
docker compose build bridge_api
docker compose up -d bridge_api
```

See `docker-compose.yml` and `psysense/Dockerfile.api`. Requires `.env` at the
repo root. The pipeline and inference servers are **not** containerized (camera
device access and the three conflicting venvs don't map cleanly onto compose
services) — run those directly on the host.

### Frontend

```powershell
cd frontend
npm run build
```

Deploy the `dist/` output to Vercel/Netlify (both auto-detect a Vite build) or
any static host. Set `VITE_API_URL` to wherever bridge_api is reachable, and
`config.yaml`'s `api.frontend_url` to the deployed frontend's exact origin —
CORS is locked to that one origin, never `*`.

## Privacy & security posture

- Raw face images are **never** stored except transiently, for the human-review
  step of the auto-update queue — and even then only until an admin/reviewer
  approves (image discarded, embedding kept) or rejects (both discarded), or
  `identity.auto_update.pending_retention_days` (default 7) expires, whichever
  comes first.
- Enrollment photos themselves are never persisted either — only the extracted
  embedding vector.
- Every write to the database goes through one serialized async writer; no
  ad-hoc SQL exists outside `db/repository/repository.py`.
- JWTs carry a role claim (`admin` | `reviewer`); reviewer-scoped routes check
  the caller's actual assignment server-side on every request, not just at
  login.
- CORS is locked to a single configured origin. Never wildcarded.
- Passwords are hashed with `bcrypt` directly (no unmaintained compatibility
  shims); login attempts are rate-limited per username with a lockout window.

## Known limitations / roadmap

- **Single-tenant.** Sub-admin "reviewer" roles exist and are scoped per-student,
  but there is no organization/tenant concept yet — one shared student roster,
  no cross-org data isolation, no billing, no email invites. This is a
  deliberately deferred, separate design effort for a multi-business deployment.
- **Identity robustness depends on enrollment breadth.** A single-photo
  enrollment will not generalize across meaningfully different conditions
  (glasses on/off, very different lighting/angle) — enroll with several photos
  spanning real variation, or add more later from a student's detail page.
- **Emotion classification reflects DeepFace's underlying model**, which is
  known to bias toward "sad" for a neutral/resting expression (a FER-2013
  training-data artifact, not a PsySense bug) — treat emotion output as an
  inferred signal, not a diagnosis.
- **No CI pipeline configured** in this repository yet — tests are run locally.

## Troubleshooting

- **"config.yaml not found"** — `core/config.py` fails loudly by design; create
  `psysense/config/config.yaml` (copy the one in this repo) rather than relying
  on hardcoded defaults.
- **bridge_api won't start** — check `ADMIN_USERNAME`/`ADMIN_PASSWORD_HASH`/
  `JWT_SECRET` are present in `.env` (loaded automatically by `api/asgi.py`);
  it refuses to start without them.
- **Emotion/pose showing as unavailable** — the pipeline degrades gracefully if
  `deepface_server`/`mediapipe_server` are down; check their own process logs,
  not `main.py`'s.
- **A newly enrolled student isn't recognized yet** — the running pipeline picks
  up new enrollments on a periodic refresh
  (`pipeline.frequencies.identity_refresh_interval_sec`, default 30s), not
  instantly.
- **Slow/erratic emotion inference, or a `haarcascade` error from DeepFace** —
  `deepface_server.py` uses `detector_backend="skip"` (the frame is already a
  person-crop from YOLO, so a second face detector is redundant); if you've
  modified that, an incomplete OpenCV install can surface as a missing
  haarcascade XML file.
- **Nothing shows up in the Review page** — that's the expected empty state, not
  an error; candidates only appear after the pipeline gets a match well above
  the normal match threshold (`identity.auto_update.distance_threshold`).
- **Diagnosing a resolve() match failure** — set `logging.level: "DEBUG"` in
  `config.yaml` temporarily; `main.py` will then log the exact quality-gate
  reason or match distance for every resolution attempt instead of a bare
  "unknown".
