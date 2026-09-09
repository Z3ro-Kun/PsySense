// api/client.ts -- thin fetch wrapper: injects the bearer token, parses
// the {"error": {code, message}} envelope bridge_api always returns on
// failure (see psysense/api/main.py's exception handlers), and notifies
// a registered handler on 401 so AuthContext can clear the session and
// redirect to /login without this module importing React/router.
import type { ApiErrorBody } from "./types";

const API_BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8080";

export class ApiError extends Error {
  status: number;
  code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

let token: string | null = null;
let onUnauthorized: (() => void) | null = null;

export function setAuthToken(next: string | null): void {
  token = next;
}

export function setUnauthorizedHandler(handler: () => void): void {
  onUnauthorized = handler;
}

interface RequestOptions {
  method?: string;
  json?: unknown;
  form?: FormData;
  params?: object;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const url = new URL(path, API_BASE_URL);
  if (options.params) {
    for (const [key, value] of Object.entries(options.params as Record<string, unknown>)) {
      if (value !== undefined && value !== null && value !== "") url.searchParams.set(key, String(value));
    }
  }

  const headers: Record<string, string> = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;

  let body: BodyInit | undefined;
  if (options.form) {
    body = options.form; // browser sets multipart Content-Type + boundary
  } else if (options.json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(options.json);
  }

  const resp = await fetch(url.toString(), { method: options.method ?? "GET", headers, body });

  if (resp.status === 401) {
    onUnauthorized?.();
  }

  if (!resp.ok) {
    let code = "ERROR";
    let message = `Request failed (${resp.status})`;
    try {
      const parsed = (await resp.json()) as ApiErrorBody;
      code = parsed.error?.code ?? code;
      message = parsed.error?.message ?? message;
    } catch {
      // Non-JSON error body (e.g. a proxy 502) -- fall back to the defaults above.
    }
    throw new ApiError(resp.status, code, message);
  }

  if (resp.status === 204) return undefined as T;
  return (await resp.json()) as T;
}

export const api = {
  get: <T>(path: string, params?: object) => request<T>(path, { method: "GET", params }),
  post: <T>(path: string, json?: unknown) => request<T>(path, { method: "POST", json }),
  postForm: <T>(path: string, form: FormData) => request<T>(path, { method: "POST", form }),
  put: <T>(path: string, json?: unknown) => request<T>(path, { method: "PUT", json }),
  del: <T>(path: string) => request<T>(path, { method: "DELETE" }),
};
