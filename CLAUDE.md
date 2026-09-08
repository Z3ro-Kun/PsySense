# PsySense — Claude Code Engineering Instructions

## 1. Mission

You are taking over an existing project called **PsySense**.

Your job is to bring the existing codebase from its current prototype/research state to a **reliable, maintainable, production-quality software product** without destroying working functionality.

Do not treat this as a greenfield project.

First understand the existing implementation, data flow, models, dependencies, and assumptions. Then improve the architecture incrementally.

The human owner makes:
- product decisions
- architectural decisions
- scope decisions
- technology decisions
- validation/acceptance decisions

You are responsible for:
- inspecting the existing code
- identifying problems
- implementing approved/implied improvements
- maintaining compatibility where reasonable
- writing robust code
- testing changes
- documenting important architectural decisions

Do not blindly rewrite working code merely because you prefer a different style.

---

# 2. What PsySense Is

PsySense is an AI-assisted classroom/student behavioral analysis system.

The current implementation processes visual information and attempts to infer:

- student identity
- facial/emotional state
- body/pose cues
- temporal/aggregated behavioral signals

The current pipeline uses computer-vision models and separate processing services.

The project currently contains components including:

- YOLO-based person detection
- StrongSORT/BoxMOT tracking
- DeepFace-based emotion analysis
- MediaPipe-based pose analysis
- identity processing
- aggregation logic
- SQLite persistence
- FastAPI bridge API
- a Python orchestration/client layer

The current code is a prototype and contains architectural, reliability, security, data-modeling, and maintainability issues.

Do not assume the current architecture is final.

---

# 3. CRITICAL FIRST STEP — UNDERSTAND BEFORE MODIFYING

Before making substantial changes:

1. Inspect the entire repository.
2. Inspect every Python source file.
3. Inspect dependency files.
4. Inspect configuration/environment files.
5. Inspect frontend files if present.
6. Inspect model-loading and inference code.
7. Trace the complete runtime pipeline.
8. Identify how processes/services communicate.
9. Identify all persisted data.
10. Identify all external model/API dependencies.
11. Identify startup/shutdown behavior.
12. Identify failure points.
13. Identify dead/duplicate/experimental code.
14. Identify assumptions that are currently undocumented.

Create a concise internal understanding of:

```text
Camera/Input
    ↓
Person Detection
    ↓
Tracking
    ↓
Identity
    ↓
Face / Emotion Analysis
    ↓
Pose Analysis
    ↓
Temporal Aggregation
    ↓
Persistence
    ↓
API
    ↓
Frontend / Consumer
```

The actual repository may differ from this diagram. Verify it from the code rather than assuming it.

---

# 4. Existing Architecture

The existing system is approximately distributed as follows:

### Main client/orchestrator

`Main.py`

Responsible for coordinating:

- YOLO detection
- StrongSORT tracking
- image cropping
- requests to analysis services
- result fusion
- aggregation
- database persistence

### DeepFace service

`deepface_server.py`

Provides emotion/face analysis through an HTTP service.

### MediaPipe service

`mediapipe_server.py`

Provides pose/body-landmark analysis through an HTTP service.

### Identity service

`identity_server.py`

Provides identity-related processing.

### Database

`db_handler.py`

Currently uses SQLite and stores aggregated emotion/pose information.

### API bridge

`bridge_api.py`

FastAPI service exposing persisted data to frontend/other consumers.

### Startup

`start_psysense.py`

Coordinates startup of required services/components.

### Analysis utilities

`pose_analyzer.py`

Contains pose-related analysis logic.

### Utilities

`utils.py`

Contains image encoding/cropping and aggregation helpers.

### Database utilities

`clear_db.py`
`check_database.py`

Used for database maintenance/debugging.

### Model conversion

`convert_pth_to_tflite.py`

Handles model conversion experiments/deployment support.

---

# 5. Core Engineering Principle

## Do not build a demo disguised as a product.

Every feature should be evaluated against:

- correctness
- reliability
- observability
- maintainability
- security
- privacy
- performance
- failure recovery
- testability
- reproducibility
- deployment practicality

If a proposed implementation creates a fragile system merely to make a feature appear to work, do not implement it that way.

---

# 6. Architecture Direction

The project should evolve toward a modular architecture.

Avoid a giant orchestrator containing:

- business logic
- inference logic
- networking
- database operations
- configuration
- aggregation
- error handling
- UI concerns

Separate responsibilities.

A reasonable target separation is:

```text
psysense/
│
├── app/
│   ├── core/
│   │   ├── config
│   │   ├── logging
│   │   ├── exceptions
│   │   └── lifecycle
│   │
│   ├── domain/
│   │   ├── models
│   │   ├── schemas
│   │   └── business rules
│   │
│   ├── inference/
│   │   ├── detection
│   │   ├── tracking
│   │   ├── face
│   │   ├── emotion
│   │   └── pose
│   │
│   ├── identity/
│   ├── aggregation/
│   ├── storage/
│   ├── services/
│   └── api/
│
├── tests/
├── scripts/
├── models/
├── configs/
├── docs/
└── ...
```

This is a target direction, not permission to blindly restructure the repository immediately.

Only restructure when it provides a concrete benefit.

---

# 7. Configuration

Do not hardcode configuration such as:

```python
http://127.0.0.1:8000
http://127.0.0.1:8001
emotion_data.db
model paths
ports
thresholds
timeouts
```

Use centralized configuration.

Configuration should support:

- development
- testing
- production

Secrets must never be committed.

Use environment variables or an appropriate configuration mechanism.

Never hardcode API keys, credentials, tokens, or private endpoints.

---

# 8. Service Communication

The current architecture uses HTTP between components.

Treat every network call as unreliable.

Every service request must account for:

- connection timeout
- read timeout
- malformed response
- HTTP errors
- service unavailable
- service startup race
- model initialization failure
- process crash
- invalid image
- oversized payload
- unexpected schema
- partial response

Never allow one unavailable analysis service to crash the entire application unless that service is explicitly classified as mandatory.

Prefer graceful degradation.

For example:

```text
Detection works
      ↓
Emotion service unavailable
      ↓
Emotion = unavailable
      ↓
Pose continues
      ↓
System remains operational
```

Do not fabricate missing inference results.

---

# 9. Startup and Shutdown

Startup must be deterministic.

The application should not assume:

> "I started the process, therefore the service is ready."

Use explicit health/readiness checks.

Startup flow should conceptually be:

```text
Start required service
        ↓
Wait for readiness
        ↓
Verify health
        ↓
Load/verify models
        ↓
Start dependent component
```

Shutdown must:

- stop child processes
- release camera/video resources
- release model resources where appropriate
- close database connections
- stop HTTP servers cleanly
- avoid orphan processes

Handle SIGINT/SIGTERM appropriately.

---

# 10. Database

The existing SQLite implementation is a prototype.

Improve it without prematurely introducing unnecessary infrastructure.

The database layer must have:

- schema versioning/migrations
- indexes for frequently queried fields
- parameterized queries
- proper connection lifecycle
- transaction handling
- validation
- consistent timestamps
- clear schema definitions

Do not store everything as opaque JSON if structured querying will be required.

JSON may be used for genuinely variable/metadata fields, but important analytical fields should be queryable.

Consider the distinction between:

```text
raw inference
derived signal
temporal aggregation
session
student
event
```

Do not mix these concepts into a single table.

---

# 11. Temporal Data

PsySense is not merely a collection of individual frame predictions.

Behavioral interpretation requires temporal context.

Design data flow so that the system can distinguish:

```text
frame-level observation
        ↓
short-term aggregation
        ↓
session-level behavior
        ↓
long-term behavioral patterns
```

Never treat one emotion prediction as a definitive statement about a student's psychological state.

Use probabilistic/observational language.

---

# 12. Vector Database / RAG Direction

A future/expanded PsySense architecture should use **vector retrieval where semantic historical/contextual retrieval is actually useful**, rather than trying to solve everything with SQL or raw JSON.

Potentially retrievable information includes:

- historical behavioral summaries
- session summaries
- contextual observations
- teacher notes
- intervention outcomes
- structured summaries transformed into embeddings
- relevant historical patterns

The vector layer must NOT become the primary source of truth.

Use:

```text
SQL / structured database
        +
vector store
        +
metadata filters
        +
retrieval layer
```

The structured database remains authoritative for structured facts.

Vector search is for semantic/contextual retrieval.

Every vector record should retain metadata sufficient to:

- identify source
- identify student/session
- identify timestamp
- enforce authorization
- filter by relevant context
- trace the retrieved information back to its source

Do not blindly embed personally identifiable information.

---

# 13. RAG Requirements

If RAG is implemented:

### Retrieval must be traceable.

Every retrieved chunk/result should have:

- source identifier
- source type
- timestamp
- relevant metadata
- confidence/relevance score where applicable

The generated answer should never imply that retrieved information is more certain than its source.

### Retrieval must be scoped.

Do not retrieve information across students or unauthorized contexts.

A query for Student A must never retrieve Student B's information.

Apply authorization and metadata filtering BEFORE or as part of retrieval, not only after generation.

### No hallucinated evidence.

If retrieval returns nothing relevant:

```text
No relevant historical evidence found.
```

is preferable to inventing context.

---

# 14. Identity

Identity is one of the most sensitive and failure-prone components.

Never assume face recognition is always correct.

Account for:

- unknown faces
- multiple similar faces
- low-quality images
- occlusion
- lighting changes
- camera angle
- false matches
- identity drift
- tracking ID changes

Use confidence thresholds and explicit `unknown` states.

Do not force every detection into a known student identity.

Identity confidence should be represented separately from behavioral/emotion confidence.

---

# 15. Emotion Analysis

Emotion classification must be treated as an **inference signal**, not a diagnosis.

Do not expose language such as:

- "this student is depressed"
- "this student has anxiety"
- "this student is mentally ill"

based solely on facial/pose inference.

Use observational terminology such as:

- detected facial expression
- inferred emotion distribution
- observed engagement cue
- behavioral signal
- confidence

Avoid converting uncertain model predictions into authoritative psychological claims.

---

# 16. Pose / Behavioral Signals

Pose analysis should similarly produce measurable signals.

Prefer:

```text
posture_score
attention_proxy
movement_level
head_orientation
body_orientation
confidence
```

over vague assertions.

Every derived signal should have a clear definition.

Document how each signal is calculated.

---

# 17. Aggregation

Do not simply average arbitrary model outputs without understanding what the values mean.

Aggregation must account for:

- confidence
- missing observations
- inconsistent frame rates
- tracking interruptions
- outliers
- temporal windows
- identity confidence
- service failures

Missing data must remain distinguishable from zero.

For example:

```text
emotion = null
```

must not silently become:

```text
emotion = 0
```

unless mathematically justified.

---

# 18. Data Quality

Every inference should have enough metadata to determine:

- when it happened
- what source generated it
- which model/version generated it
- which student/track it belongs to
- confidence
- whether it was inferred or observed
- whether the result was complete/partial

Model changes should not silently make historical data incomparable.

Store model/version metadata where useful.

---

# 19. Privacy and Security

PsySense processes highly sensitive educational/behavioral information.

Treat privacy as a first-class architectural concern.

Never:

- log raw face images unnecessarily
- log credentials
- expose database files through APIs
- expose unrestricted student data
- use wildcard CORS in production
- trust arbitrary student IDs from clients
- store unnecessary biometric data
- send sensitive data to external services without explicit justification

Development conveniences must not become production defaults.

The existing broad CORS configuration must be treated as temporary/development-only.

---

# 20. API Design

The FastAPI layer should use:

- typed request/response schemas
- validation
- consistent HTTP status codes
- meaningful error responses
- pagination for large datasets
- filtering
- explicit date/time semantics
- API versioning where appropriate
- health/readiness endpoints
- structured logging

Avoid endpoints that return unbounded datasets.

Do not expose internal exceptions directly to clients.

Bad:

```text
500 + raw Python exception
```

Better:

```json
{
  "error": {
    "code": "DATABASE_UNAVAILABLE",
    "message": "Unable to retrieve requested data."
  }
}
```

Keep internal diagnostic information in logs.

---

# 21. Observability

Replace uncontrolled `print()` debugging with structured logging.

Logs should make it possible to determine:

- service startup
- service shutdown
- inference failures
- API failures
- database failures
- model loading
- dropped frames
- processing latency
- identity failures
- degraded operation

Use appropriate log levels:

```text
DEBUG
INFO
WARNING
ERROR
CRITICAL
```

Never log sensitive personal information unless explicitly required and justified.

---

# 22. Performance

Computer vision is expensive.

Do not optimize blindly.

Measure first.

Potential bottlenecks include:

- model inference
- image encoding
- HTTP serialization
- repeated model loading
- unnecessary image copies
- synchronous network calls
- database writes every frame
- excessive logging
- redundant inference

Avoid writing to SQLite once per frame if batching/windowed persistence is appropriate.

Avoid sending unnecessarily large images between services.

Do not sacrifice correctness merely for arbitrary FPS numbers.

---

# 23. Concurrency

The current system contains threading/process/service interactions.

Be careful with:

- shared mutable state
- SQLite connections across threads
- model thread safety
- queues
- race conditions
- process termination
- resource ownership

Every resource should have a clear owner.

Do not share objects across processes unless the mechanism is explicitly safe.

---

# 24. Failure Matrix

Assume these failures will happen.

| Failure | Required behavior |
|---|---|
| Camera unavailable | Clear error + safe shutdown/retry |
| Camera disconnects | Detect and recover if possible |
| YOLO model missing | Fail clearly during startup |
| ReID model missing | Clear startup failure/degraded mode |
| DeepFace unavailable | Continue in degraded mode if possible |
| MediaPipe unavailable | Continue in degraded mode if possible |
| Identity unavailable | Mark identity unknown |
| Database unavailable | Do not silently lose data; expose failure |
| Malformed API response | Validate + log + degrade |
| Network timeout | Timeout + controlled retry |
| Service crashes | Detect + controlled restart where appropriate |
| Invalid frame | Drop frame safely |
| Empty detection | Normal condition, not an exception |
| Face not visible | Unknown/no-face state |
| Student identity uncertain | Do not force identity |
| Database corruption | Detect and fail safely |
| Model inference exception | Isolate failure from entire pipeline |
| Memory pressure | Backpressure/drop strategy |
| Queue overflow | Explicit policy, metrics/logging |
| Shutdown signal | Clean resource release |

No infinite retry loops.

Use bounded retries and backoff where retries are appropriate.

---

# 25. Testing

Do not consider "it runs" to mean "it works."

Build tests around:

### Unit tests

Test:

- image utilities
- aggregation
- confidence handling
- schemas
- database functions
- API validation
- configuration
- failure handling

### Integration tests

Test:

```text
API → database
orchestrator → service
service → inference
startup → health check
```

### End-to-end tests

Where practical:

```text
input/video
    ↓
detection
    ↓
tracking
    ↓
analysis
    ↓
aggregation
    ↓
database
    ↓
API
```

Use mocks/fixtures for expensive ML models where appropriate.

Do not require a GPU for ordinary unit tests.

---

# 26. Type Safety

Use type hints consistently.

Prefer explicit types over:

```python
data = {}
result = {}
```

when the structure is known.

Use Pydantic/dataclasses/models where appropriate.

Avoid `Any` unless genuinely necessary.

---

# 27. Error Handling

Do not use:

```python
except Exception:
    pass
```

and do not silently swallow failures.

Every exception should be:

- handled intentionally
- propagated intentionally
- logged appropriately
- converted into a safe user-facing error where necessary

Do not use exceptions for ordinary control flow.

---

# 28. Dependencies

Audit dependencies.

For each dependency consider:

- Is it actually used?
- Is it still maintained?
- Is the version compatible?
- Is it required at runtime?
- Is it GPU-specific?
- Is it platform-specific?
- Does it introduce licensing concerns?

Do not upgrade every dependency automatically.

ML dependencies are particularly prone to compatibility breakage.

Change them deliberately.

---

# 29. Model Management

Models must not be treated as random files sitting in the project directory.

Track:

- model name
- version
- source
- expected format
- checksum where appropriate
- runtime requirements
- preprocessing requirements

Do not automatically download arbitrary model files at runtime without explicit design.

Handle missing models with actionable errors.

---

# 30. Reproducibility

A new developer/machine should be able to understand:

1. What needs to be installed.
2. Which Python version is required.
3. Which models are required.
4. How services start.
5. How the database is initialized.
6. How tests run.
7. How the application runs.
8. What environment variables are required.

Document this.

---

# 31. Git Hygiene

Never:

- commit secrets
- commit generated databases
- commit huge model binaries unless explicitly intended
- commit caches
- commit virtual environments
- commit temporary debugging files

Maintain an appropriate `.gitignore`.

Make changes in logically separable commits when the user is managing commits.

Do not rewrite Git history unless explicitly instructed.

---

# 32. Documentation

Documentation should describe the **actual system**, not an imaginary architecture.

At minimum maintain:

```text
README
Architecture documentation
Setup instructions
Environment configuration
API documentation
Model requirements
Troubleshooting
Testing instructions
```

If architecture changes, update the documentation.

---

# 33. Refactoring Rules

Before refactoring a component:

1. Identify what it currently does.
2. Identify who calls it.
3. Identify what behavior depends on it.
4. Identify implicit contracts.
5. Add tests where practical.
6. Refactor incrementally.
7. Verify behavior after each major step.

Do not perform enormous rewrites where a controlled migration is possible.

---

# 34. Do Not Preserve Bad Architecture for Sentiment

The fact that code already exists does NOT mean it should remain.

If you find:

- duplicated logic
- unsafe networking
- invalid concurrency
- bad database design
- insecure CORS
- hardcoded configuration
- untestable functions
- tightly coupled services
- unnecessary complexity

call it out and fix it where appropriate.

However, replace it with a demonstrably better design—not merely a different design.

---

# 35. Productization Standard

The final system should behave like software someone else can actually use.

A successful implementation should have:

- deterministic setup
- clear configuration
- predictable startup
- predictable shutdown
- health checks
- meaningful errors
- robust failure handling
- structured logs
- tests
- documentation
- clean API contracts
- maintainable modules
- reproducible environments

A feature is not finished merely because its happy path works.

---

# 36. Development Workflow

For every significant task:

### Phase 1 — Inspect

Understand the existing implementation.

### Phase 2 — Plan

State:

- what is wrong
- why it is wrong
- what will change
- what files will change
- what risks exist
- how it will be tested

### Phase 3 — Implement

Make the smallest coherent set of changes.

### Phase 4 — Validate

Run:

- syntax/type checks where applicable
- unit tests
- integration tests
- application startup checks

### Phase 5 — Review

Check for:

- regressions
- security issues
- race conditions
- error handling
- unnecessary complexity
- documentation drift

### Phase 6 — Report

Summarize:

```text
Changed
Tested
Known limitations
Next recommended step
```

---

# 37. Important Constraint — Do Not Invent Requirements

Do not invent product requirements.

If something is ambiguous and materially affects architecture, stop and ask.

Do not silently decide that PsySense needs:

- a cloud backend
- a paid external API
- a particular database
- authentication provider
- Kubernetes
- microservices
- a particular vector database
- a particular frontend framework

unless there is an actual project requirement or an explicit architectural decision.

Prefer the simplest architecture that satisfies the real requirements.

---

# 38. AI / LLM Usage

If LLM functionality is introduced:

Never allow the LLM to become the source of truth.

The LLM should operate over:

```text
validated structured data
+
retrieved contextual information
+
explicit system rules
```

not arbitrary raw application state.

LLM outputs must be treated as untrusted generated content.

Validate structured outputs.

Do not allow model-generated text to directly perform privileged operations.

---

# 39. Behavioral Interpretation Guardrail

PsySense deals with human behavior.

The system should distinguish:

```text
Observation
    ↓
Inference
    ↓
Interpretation
```

These are not equivalent.

For example:

```text
Observation:
head turned away from screen

Inference:
low attention proxy

Interpretation:
possible disengagement
```

Do NOT collapse this into:

```text
Student is not interested.
```

Avoid deterministic psychological conclusions from visual signals.

---

# 40. Current Strategic Direction

The original PsySense concept became too broad when treated as a complete AI platform.

The broader architectural lessons remain valuable:

- contextual reasoning
- tiered retrieval
- RAG
- structured + semantic data
- traceability
- human-in-the-loop decisions
- security
- local/cloud separation
- modular agents
- observability

However, do not expand scope merely because these ideas exist.

The immediate priority is:

> Make the existing PsySense system technically sound, reliable, maintainable, and demonstrable.

Then expand deliberately.

---

# 41. Definition of Done

Do not declare a task complete because the code compiles.

A change is complete when:

- the intended behavior works
- existing behavior has not been unnecessarily broken
- failure paths are considered
- relevant tests pass
- configuration is sane
- logs are useful
- security implications are addressed
- documentation is updated where necessary
- no obvious technical debt was introduced
- the implementation matches the actual architecture

---

# 42. Final Rule

Be critical.

If an approach is technically weak, say so.

If an existing implementation is fundamentally flawed, explain why and propose a better migration path.

Do not optimize for making the human owner happy.

Optimize for building a system that is:

**correct, robust, maintainable, secure, testable, explainable, and realistically deployable.**

Before major implementation work, inspect the repository and establish the current state. Do not assume this document describes every detail of the existing implementation; the source code is authoritative for current behavior.