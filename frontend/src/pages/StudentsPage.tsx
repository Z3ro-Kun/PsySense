import { useState } from "react";
import { Link } from "react-router-dom";
import { studentsApi } from "../api/endpoints";
import { useAsync } from "../hooks/useAsync";

const PAGE_SIZE = 20;

export function StudentsPage() {
  const [search, setSearch] = useState("");
  const [offset, setOffset] = useState(0);

  const students = useAsync(
    () => studentsApi.list({ search: search || undefined, limit: PAGE_SIZE, offset }),
    [search, offset],
  );

  return (
    <div>
      <div className="toolbar">
        <h1 className="page-title" style={{ marginBottom: 0 }}>Students</h1>
        <Link to="/students/enroll" className="btn" style={{ textDecoration: "none" }}>
          + Enroll student
        </Link>
      </div>

      <div className="card">
        <input
          type="text"
          placeholder="Search by name…"
          value={search}
          onChange={(e) => {
            setSearch(e.target.value);
            setOffset(0);
          }}
          style={{
            padding: "8px 10px", borderRadius: 6, border: "1px solid var(--border)",
            background: "var(--surface-1)", color: "var(--text-primary)", width: "100%", marginBottom: 12,
          }}
        />

        {students.loading && <p className="empty-state">Loading…</p>}
        {students.error && <p className="error-text">{students.error}</p>}
        {students.data && students.data.items.length === 0 && (
          <p className="empty-state">
            {search ? "No students match that search." : "No students enrolled yet."}
          </p>
        )}
        {students.data && students.data.items.length > 0 && (
          <>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>External ref</th>
                  <th>Enrolled</th>
                </tr>
              </thead>
              <tbody>
                {students.data.items.map((student) => (
                  <tr key={student.student_id}>
                    <td>
                      <Link to={`/students/${student.student_id}`}>{student.display_name}</Link>
                    </td>
                    <td>{student.external_ref ?? "—"}</td>
                    <td>{new Date(student.enrolled_at).toLocaleDateString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="pagination">
              <button
                className="btn btn-secondary"
                disabled={offset === 0}
                onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
              >
                Previous
              </button>
              <span>
                {offset + 1}–{Math.min(offset + PAGE_SIZE, students.data.total)} of {students.data.total}
              </span>
              <button
                className="btn btn-secondary"
                disabled={offset + PAGE_SIZE >= students.data.total}
                onClick={() => setOffset(offset + PAGE_SIZE)}
              >
                Next
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
