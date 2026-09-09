import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { studentsApi, telemetryApi } from "../api/endpoints";
import { useAsync } from "../hooks/useAsync";
import { LineChart } from "../components/LineChart";
import { SeverityBadge } from "../components/SeverityBadge";
import { DateRangeFilter, DEFAULT_RANGE_LABEL, sinceForLabel } from "../components/DateRangeFilter";
import { PhotoDropzone } from "../components/PhotoDropzone";
import { buildEmotionSeries, buildPoseSeries } from "../lib/chartData";
import { ApiError } from "../api/client";
import type { EnrollmentResult } from "../api/types";

export function StudentDetailPage() {
  const { studentId } = useParams<{ studentId: string }>();
  const navigate = useNavigate();
  const [rangeLabel, setRangeLabel] = useState(DEFAULT_RANGE_LABEL);
  const since = sinceForLabel(rangeLabel);

  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const [addingPhotos, setAddingPhotos] = useState(false);
  const [newImages, setNewImages] = useState<File[]>([]);
  const [submittingPhotos, setSubmittingPhotos] = useState(false);
  const [addPhotosError, setAddPhotosError] = useState<string | null>(null);
  const [addPhotosResult, setAddPhotosResult] = useState<EnrollmentResult | null>(null);

  const student = useAsync(() => studentsApi.get(studentId!), [studentId]);
  const emotions = useAsync(() => telemetryApi.studentEmotions(studentId!, { since, limit: 200 }), [studentId, since]);
  const pose = useAsync(() => telemetryApi.studentPose(studentId!, { since, limit: 200 }), [studentId, since]);
  const behaviorEvents = useAsync(
    () => telemetryApi.studentBehaviorEvents(studentId!, { since, limit: 50 }),
    [studentId, since],
  );

  if (student.error) return <p className="error-text">{student.error}</p>;

  const emotionChart = emotions.data ? buildEmotionSeries(emotions.data.items) : null;
  const poseChart = pose.data ? buildPoseSeries(pose.data.items) : null;

  const handleDelete = async () => {
    setDeleting(true);
    setDeleteError(null);
    try {
      await studentsApi.delete(studentId!);
      navigate("/students", { replace: true });
    } catch (err) {
      setDeleteError(err instanceof ApiError ? err.message : "Could not delete this student.");
      setDeleting(false);
    }
  };

  const handleAddPhotos = async () => {
    if (newImages.length === 0) {
      setAddPhotosError("Add at least one photo.");
      return;
    }
    setSubmittingPhotos(true);
    setAddPhotosError(null);
    try {
      const res = await studentsApi.addPhotos(studentId!, newImages);
      setAddPhotosResult(res);
      setNewImages([]);
    } catch (err) {
      setAddPhotosError(err instanceof ApiError ? err.message : "Could not add these photos.");
    } finally {
      setSubmittingPhotos(false);
    }
  };

  return (
    <div>
      <div className="toolbar">
        <div>
          <h1 className="page-title" style={{ marginBottom: 0 }}>{student.data?.display_name ?? "…"}</h1>
          {student.data?.external_ref && <p style={{ color: "var(--text-muted)", margin: 0 }}>{student.data.external_ref}</p>}
        </div>

        {!confirmingDelete && (
          <div style={{ display: "flex", gap: 8 }}>
            <button
              className="btn btn-secondary"
              onClick={() => setAddingPhotos((v) => !v)}
              disabled={!student.data}
            >
              {addingPhotos ? "Cancel" : "+ Add photos"}
            </button>
            <button className="btn btn-secondary" onClick={() => setConfirmingDelete(true)} disabled={!student.data}>
              Delete student
            </button>
          </div>
        )}
        {confirmingDelete && (
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontSize: 13, color: "var(--text-secondary)" }}>
              Delete {student.data?.display_name}? This removes their face data and cannot be undone.
            </span>
            <button
              className="btn"
              style={{ background: "var(--status-critical)" }}
              onClick={handleDelete}
              disabled={deleting}
            >
              {deleting ? "Deleting…" : "Confirm delete"}
            </button>
            <button className="btn btn-secondary" onClick={() => setConfirmingDelete(false)} disabled={deleting}>
              Cancel
            </button>
          </div>
        )}
      </div>
      {deleteError && <p className="error-text">{deleteError}</p>}

      {addingPhotos && (
        <div className="card">
          <h2>Add photos</h2>
          <p style={{ color: "var(--text-muted)", fontSize: 12, marginTop: -8 }}>
            Adds to {student.data?.display_name}'s existing profile -- different angles or lighting
            help matching generalize, rather than creating a separate duplicate student.
          </p>
          <PhotoDropzone images={newImages} onImagesChange={setNewImages} />
          {addPhotosError && <p className="error-text">{addPhotosError}</p>}
          {addPhotosResult && (
            <p style={{ fontSize: 13, color: "var(--text-secondary)" }}>
              {addPhotosResult.accepted_images} photo(s) added.
              {addPhotosResult.skipped_images.length > 0 && ` ${addPhotosResult.skipped_images.length} skipped.`}
            </p>
          )}
          <button className="btn" onClick={handleAddPhotos} disabled={submittingPhotos}>
            {submittingPhotos ? "Adding…" : "Add photos"}
          </button>
        </div>
      )}

      <div style={{ marginBottom: 20 }}>
        <DateRangeFilter value={rangeLabel} onChange={setRangeLabel} />
      </div>

      <div className="card">
        <h2>Emotion trend</h2>
        <p style={{ color: "var(--text-muted)", fontSize: 12, marginTop: -8 }}>
          Windowed averages from the Fusion Engine -- an inferred emotion distribution, not a diagnosis.
        </p>
        {emotions.loading && <p className="empty-state">Loading…</p>}
        {emotions.error && <p className="error-text">{emotions.error}</p>}
        {emotionChart && (
          <LineChart x={emotionChart.x} series={emotionChart.series} yDomain={[0, 1]} yFormat={(v) => v.toFixed(2)} />
        )}
      </div>

      <div className="card">
        <h2>Pose signals</h2>
        <p style={{ color: "var(--text-muted)", fontSize: 12, marginTop: -8 }}>
          Behavioral proxies derived from posture landmarks -- observed signals, not conclusions about attention or intent.
        </p>
        {pose.loading && <p className="empty-state">Loading…</p>}
        {pose.error && <p className="error-text">{pose.error}</p>}
        {poseChart && (
          <LineChart x={poseChart.x} series={poseChart.series} yDomain={[0, 1]} yFormat={(v) => v.toFixed(2)} />
        )}
      </div>

      <div className="card">
        <h2>Behavior events</h2>
        {behaviorEvents.loading && <p className="empty-state">Loading…</p>}
        {behaviorEvents.error && <p className="error-text">{behaviorEvents.error}</p>}
        {behaviorEvents.data && behaviorEvents.data.items.length === 0 && (
          <p className="empty-state">No behavior events flagged in this range.</p>
        )}
        {behaviorEvents.data && behaviorEvents.data.items.length > 0 && (
          <table className="data-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Event</th>
                <th>Severity</th>
                <th>Detail</th>
              </tr>
            </thead>
            <tbody>
              {behaviorEvents.data.items.map((event) => (
                <tr key={event.event_id}>
                  <td>{new Date(event.timestamp).toLocaleString()}</td>
                  <td>{event.event_type.replaceAll("_", " ")}</td>
                  <td>
                    <SeverityBadge severity={event.severity} />
                  </td>
                  <td>{typeof event.payload?.detail === "string" ? event.payload.detail : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
