"""
api/asgi.py -- production ASGI entrypoint.

    uvicorn api.asgi:app --host 0.0.0.0 --port 8080

Kept separate from api/main.py: this is the one place create_app() is
called with no overrides, so it's also the one place that eagerly loads
real models and requires ADMIN_USERNAME/ADMIN_PASSWORD_HASH/JWT_SECRET to
already be set in the environment -- and, correspondingly, the one place
that loads .env, so those variables don't have to be exported by hand in
whatever terminal happens to launch this process. Without this, a stale
terminal with old `$env:` values (or one where they were never set at
all) silently wins over an updated .env file, which is exactly the kind
of confusing failure this is meant to prevent.
"""
from pathlib import Path

from dotenv import load_dotenv

# .env lives at the repo root, one level above psysense/ -- load it
# explicitly rather than relying on CWD, since this is commonly launched
# from inside psysense/ (see README's run instructions).
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from api.main import create_app  # noqa: E402 (must follow load_dotenv)

app = create_app()
