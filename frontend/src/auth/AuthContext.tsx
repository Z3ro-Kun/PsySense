// auth/AuthContext.tsx -- holds the JWT + role (persisted to localStorage
// for convenience across reloads; scoped to this artifact/app's own
// origin, not shared with anything else), and reacts to the API client's
// 401 handler by clearing the session.
import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { setAuthToken, setUnauthorizedHandler } from "../api/client";
import { authApi } from "../api/endpoints";

const STORAGE_KEY = "psysense_session";

export type Role = "admin" | "reviewer";

interface StoredSession {
  token: string;
  role: Role;
}

function readStoredSession(): StoredSession | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (typeof parsed?.token === "string" && (parsed.role === "admin" || parsed.role === "reviewer")) return parsed;
    return null;
  } catch {
    return null; // private browsing / storage disabled, or corrupt value -- fall back to session-only auth
  }
}

function writeStoredSession(session: StoredSession | null): void {
  try {
    if (session) localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
    else localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore -- session still held in memory for this tab via React state
  }
}

interface AuthContextValue {
  isAuthenticated: boolean;
  role: Role | null;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<StoredSession | null>(readStoredSession);

  // Called during render (not inside an effect) so the token is set on
  // the shared api client BEFORE any child's own effect runs its first
  // fetch -- child effects fire before a parent's effect, so syncing this
  // only in useEffect here would race the very first page load.
  setAuthToken(session?.token ?? null);

  useEffect(() => {
    setUnauthorizedHandler(() => setSession(null));
  }, []);

  useEffect(() => {
    writeStoredSession(session);
  }, [session]);

  const login = async (username: string, password: string) => {
    const resp = await authApi.login(username, password);
    setSession({ token: resp.access_token, role: resp.role as Role });
  };

  const logout = () => setSession(null);

  const value = useMemo<AuthContextValue>(
    () => ({ isAuthenticated: session !== null, role: session?.role ?? null, login, logout }),
    [session],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
