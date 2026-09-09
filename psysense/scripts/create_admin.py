"""
scripts/create_admin.py

One-time setup helper: prompts for the dashboard admin password and
prints the ADMIN_PASSWORD_HASH env var to put in .env. Never writes the
plaintext or the hash to disk itself -- you copy it into .env yourself,
which stays gitignored (see api/auth.py for why this can't just be a
config.yaml value).

Password entry is deliberately VISIBLE (plain input(), not getpass) --
this is a local, single-operator setup script, not a shared login
prompt, and hidden entry with no way to double-check what you typed was
a real source of confusion (mistyped/mis-remembered passwords with no
way to tell whether a later login failure was the password or something
else). The script also immediately re-verifies the hash it just
generated against what you typed, so a bug or typo is caught right here
-- not discovered later as an unexplained login failure.

Run from psysense/:
    python -m scripts.create_admin
"""
from __future__ import annotations

import secrets

from api.auth import hash_password, verify_password


def main() -> None:
    username = input("Admin username: ").strip()
    if not username:
        raise SystemExit("Username must not be empty.")

    password = input("Admin password (visible -- this is a local setup script, not a login prompt): ").strip()
    confirm = input("Confirm password: ").strip()
    if password != confirm:
        raise SystemExit("Passwords did not match.")
    if len(password) < 8:
        raise SystemExit("Password must be at least 8 characters.")

    password_hash = hash_password(password)

    # Immediately prove the hash we're about to print actually verifies
    # against the password you typed -- this should never fail, but if
    # it somehow did, better to find out now than after copying a bad
    # hash into .env and getting a confusing 401 later.
    if not verify_password(password, password_hash):
        raise SystemExit(
            "Internal error: the generated hash did not verify against the password just entered. "
            "Do not use this hash -- report this as a bug."
        )

    jwt_secret = secrets.token_urlsafe(48)

    print("\nVerification: OK -- this hash correctly matches the password you entered.\n")
    print("Add these to your .env (do not commit .env):\n")
    print(f"ADMIN_USERNAME={username}")
    print(f"ADMIN_PASSWORD_HASH={password_hash}")
    print(f"JWT_SECRET={jwt_secret}")
    print(
        "\nRestart bridge_api after updating .env -- api/asgi.py loads .env once at "
        "process startup, so an already-running process keeps using the old values."
    )


if __name__ == "__main__":
    main()
