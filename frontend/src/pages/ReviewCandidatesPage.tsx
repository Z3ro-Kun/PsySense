import { useState } from "react";
import { Link } from "react-router-dom";
import { identityCandidatesApi } from "../api/endpoints";
import { useAsync } from "../hooks/useAsync";
import { ApiError } from "../api/client";
import "./ReviewCandidatesPage.css";

export function ReviewCandidatesPage() {
  const candidates = useAsync(() => identityCandidatesApi.list({ limit: 50 }), []);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleDecision = async (candidateId: string, decision: "approve" | "reject") => {
    setBusyId(candidateId);
    setError(null);
    try {
      if (decision === "approve") await identityCandidatesApi.approve(candidateId);
      else await identityCandidatesApi.reject(candidateId);
      candidates.refetch();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not save that decision.");
    } finally {
      setBusyId(null);
    }
  };

  return (
    <div>
      <h1 className="page-title">Review identity candidates</h1>
      <p style={{ color: "var(--text-muted)", fontSize: 13, marginTop: -12, marginBottom: 20 }}>
        The pipeline queues a photo here only after a very high-confidence match -- nothing is ever
        added to a student's profile without you confirming it's really them.
      </p>

      {candidates.loading && <p className="empty-state">Loading…</p>}
      {candidates.error && <p className="error-text">{candidates.error}</p>}
      {error && <p className="error-text">{error}</p>}
      {candidates.data && candidates.data.items.length === 0 && (
        <p className="empty-state">No candidates waiting for review.</p>
      )}

      <div className="candidate-grid">
        {candidates.data?.items.map((c) => (
          <div key={c.candidate_id} className="card candidate-card">
            <img
              className="candidate-image"
              src={`data:image/jpeg;base64,${c.image_b64}`}
              alt={`Candidate photo for ${c.student_display_name}`}
            />
            <div className="candidate-meta">
              <Link to={`/students/${c.student_id}`}>{c.student_display_name}</Link>
              <span style={{ color: "var(--text-muted)" }}>
                match distance {c.distance.toFixed(3)} · {new Date(c.captured_at).toLocaleString()}
              </span>
            </div>
            <div className="candidate-actions">
              <button
                className="btn"
                disabled={busyId === c.candidate_id}
                onClick={() => handleDecision(c.candidate_id, "approve")}
              >
                Approve
              </button>
              <button
                className="btn btn-secondary"
                disabled={busyId === c.candidate_id}
                onClick={() => handleDecision(c.candidate_id, "reject")}
              >
                Reject
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
