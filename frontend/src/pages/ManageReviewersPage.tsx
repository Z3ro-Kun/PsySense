import { useEffect, useState, type FormEvent } from "react";
import { usersApi, studentsApi } from "../api/endpoints";
import { useAsync } from "../hooks/useAsync";
import { ApiError } from "../api/client";
import type { StudentOut, UserOut } from "../api/types";
import "./ManageReviewersPage.css";

export function ManageReviewersPage() {
  const users = useAsync(() => usersApi.list(), []);
  const [expandedUserId, setExpandedUserId] = useState<string | null>(null);
  const [showCreate, setShowCreate] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [createError, setCreateError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);

  const handleCreate = async (e: FormEvent) => {
    e.preventDefault();
    setCreateError(null);
    if (password !== confirmPassword) {
      setCreateError("Passwords do not match.");
      return;
    }
    setCreating(true);
    try {
      await usersApi.create(username, password, "reviewer");
      setUsername("");
      setPassword("");
      setConfirmPassword("");
      setShowCreate(false);
      users.refetch();
    } catch (err) {
      setCreateError(err instanceof ApiError ? err.message : "Could not create this account.");
    } finally {
      setCreating(false);
    }
  };

  const handleToggleActive = async (user: UserOut) => {
    if (user.active) await usersApi.deactivate(user.user_id);
    else await usersApi.reactivate(user.user_id);
    users.refetch();
  };

  return (
    <div>
      <div className="toolbar">
        <h1 className="page-title" style={{ marginBottom: 0 }}>Reviewers</h1>
        <button className="btn" onClick={() => setShowCreate((v) => !v)}>
          {showCreate ? "Cancel" : "+ New reviewer"}
        </button>
      </div>
      <p style={{ color: "var(--text-muted)", fontSize: 13, marginTop: -12, marginBottom: 20 }}>
        A reviewer can only see and act on identity candidates for the students assigned to them below --
        never the whole system.
      </p>

      {showCreate && (
        <form className="card" onSubmit={handleCreate} style={{ maxWidth: 420 }}>
          <div className="form-field">
            <label htmlFor="ruser">Username</label>
            <input id="ruser" type="text" value={username} onChange={(e) => setUsername(e.target.value)} required />
          </div>
          <div className="form-field">
            <label htmlFor="rpass">Password</label>
            <input id="rpass" type="password" value={password} onChange={(e) => setPassword(e.target.value)} required minLength={8} />
          </div>
          <div className="form-field">
            <label htmlFor="rpass2">Confirm password</label>
            <input id="rpass2" type="password" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} required />
          </div>
          {createError && <p className="error-text">{createError}</p>}
          <button className="btn" type="submit" disabled={creating}>
            {creating ? "Creating…" : "Create reviewer"}
          </button>
        </form>
      )}

      <div className="card">
        {users.loading && <p className="empty-state">Loading…</p>}
        {users.error && <p className="error-text">{users.error}</p>}
        {users.data && users.data.length === 0 && <p className="empty-state">No reviewer accounts yet.</p>}
        {users.data && users.data.length > 0 && (
          <table className="data-table">
            <thead>
              <tr>
                <th>Username</th>
                <th>Role</th>
                <th>Status</th>
                <th>Assigned</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {users.data.map((u) => (
                <tr key={u.user_id}>
                  <td>{u.username}</td>
                  <td style={{ textTransform: "capitalize" }}>{u.role}</td>
                  <td>{u.active ? "Active" : "Deactivated"}</td>
                  <td>{u.assigned_student_count}</td>
                  <td>
                    <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
                      {u.role === "reviewer" && (
                        <button
                          className="btn btn-secondary"
                          onClick={() => setExpandedUserId(expandedUserId === u.user_id ? null : u.user_id)}
                        >
                          {expandedUserId === u.user_id ? "Close" : "Manage students"}
                        </button>
                      )}
                      <button className="btn btn-secondary" onClick={() => handleToggleActive(u)}>
                        {u.active ? "Deactivate" : "Reactivate"}
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {expandedUserId && <ReviewerAssignmentEditor key={expandedUserId} userId={expandedUserId} />}
    </div>
  );
}

function ReviewerAssignmentEditor({ userId }: { userId: string }) {
  const [assignedIds, setAssignedIds] = useState<Set<string>>(new Set());
  const [assignedStudents, setAssignedStudents] = useState<StudentOut[]>([]);
  const [search, setSearch] = useState("");
  const [searchResults, setSearchResults] = useState<StudentOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      try {
        const { student_ids } = await usersApi.getAssignments(userId);
        if (cancelled) return;
        setAssignedIds(new Set(student_ids));
        const students = await Promise.all(
          student_ids.map((id) => studentsApi.get(id).catch(() => null)),
        );
        if (!cancelled) setAssignedStudents(students.filter((s): s is StudentOut => s !== null));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [userId]);

  useEffect(() => {
    let cancelled = false;
    studentsApi.list({ search: search || undefined, limit: 20 }).then((res) => {
      if (!cancelled) setSearchResults(res.items);
    });
    return () => {
      cancelled = true;
    };
  }, [search]);

  const toggle = (student: StudentOut) => {
    setSaved(false);
    setAssignedIds((prev) => {
      const next = new Set(prev);
      if (next.has(student.student_id)) {
        next.delete(student.student_id);
        setAssignedStudents((s) => s.filter((x) => x.student_id !== student.student_id));
      } else {
        next.add(student.student_id);
        // Idempotent on purpose: two toggle() calls for the same student
        // in quick succession (e.g. a fast double-click, or two clicks
        // landing on the same row before React re-renders in between)
        // must not add a second chip -- assignedIds (a Set) already
        // de-dupes correctly via functional updates, but this array push
        // didn't check membership before, so it could drift out of sync
        // with the Set and show a duplicate.
        setAssignedStudents((s) => (s.some((x) => x.student_id === student.student_id) ? s : [...s, student]));
      }
      return next;
    });
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      await usersApi.setAssignments(userId, [...assignedIds]);
      setSaved(true);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not save assignments.");
    } finally {
      setSaving(false);
    }
  };

  if (loading) return <p className="empty-state">Loading assignments…</p>;

  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>Assigned students</h3>
      {assignedStudents.length === 0 && <p className="empty-state">No students assigned yet.</p>}
      <div className="assignment-chips">
        {assignedStudents.map((s) => (
          <span key={s.student_id} className="assignment-chip">
            {s.display_name}
            <button type="button" onClick={() => toggle(s)} aria-label={`Unassign ${s.display_name}`}>
              ×
            </button>
          </span>
        ))}
      </div>

      <input
        type="text"
        placeholder="Search students to assign…"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        className="assignment-search"
      />
      <ul className="assignment-list">
        {searchResults.map((s) => (
          <li key={s.student_id}>
            <input
              type="checkbox"
              checked={assignedIds.has(s.student_id)}
              onChange={() => toggle(s)}
              id={`assign-${s.student_id}`}
            />
            <label htmlFor={`assign-${s.student_id}`}>{s.display_name}</label>
          </li>
        ))}
        {searchResults.length === 0 && <li className="empty-state">No students match.</li>}
      </ul>

      {error && <p className="error-text">{error}</p>}
      {saved && !error && <p style={{ color: "var(--status-good)", fontSize: 13 }}>Saved.</p>}
      <button className="btn" onClick={handleSave} disabled={saving} style={{ marginTop: 8 }}>
        {saving ? "Saving…" : "Save assignments"}
      </button>
    </div>
  );
}
