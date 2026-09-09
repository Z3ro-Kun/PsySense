import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { studentsApi } from "../api/endpoints";
import { ApiError } from "../api/client";
import type { EnrollmentResult } from "../api/types";
import { PhotoDropzone } from "../components/PhotoDropzone";

export function EnrollStudentPage() {
  const navigate = useNavigate();
  const [displayName, setDisplayName] = useState("");
  const [externalRef, setExternalRef] = useState("");
  const [images, setImages] = useState<File[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<EnrollmentResult | null>(null);

  const handleSubmit = async (evt: FormEvent) => {
    evt.preventDefault();
    setError(null);
    if (images.length === 0) {
      setError("Add at least one photo.");
      return;
    }
    setSubmitting(true);
    try {
      const res = await studentsApi.enroll(displayName, externalRef || undefined, images);
      setResult(res);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Enrollment failed.");
    } finally {
      setSubmitting(false);
    }
  };

  if (result) {
    return (
      <div className="card" style={{ maxWidth: 480 }}>
        <h2>Enrolled {result.student.display_name}</h2>
        <p>{result.accepted_images} photo(s) accepted.</p>
        {result.skipped_images.length > 0 && (
          <div>
            <p>{result.skipped_images.length} photo(s) skipped:</p>
            <ul>
              {result.skipped_images.map((reason, i) => (
                <li key={i} className="empty-state" style={{ textAlign: "left", padding: 0 }}>
                  {reason}
                </li>
              ))}
            </ul>
          </div>
        )}
        <div style={{ display: "flex", gap: 8, marginTop: 16 }}>
          <button className="btn" onClick={() => navigate(`/students/${result.student.student_id}`)}>
            View student
          </button>
          <button
            className="btn btn-secondary"
            onClick={() => {
              setResult(null);
              setDisplayName("");
              setExternalRef("");
              setImages([]);
            }}
          >
            Enroll another
          </button>
        </div>
      </div>
    );
  }

  return (
    <div>
      <h1 className="page-title">Enroll a student</h1>
      <p style={{ color: "var(--text-muted)", fontSize: 13, marginTop: -12, marginBottom: 20 }}>
        Add several photos in one go (different angles, glasses on/off if that varies) -- a richer
        initial profile matches more reliably than a single photo. You can also add more photos to
        this student later from their detail page.
      </p>
      <form className="card" style={{ maxWidth: 480 }} onSubmit={handleSubmit}>
        <div className="form-field">
          <label htmlFor="displayName">Display name</label>
          <input id="displayName" type="text" value={displayName} onChange={(e) => setDisplayName(e.target.value)} required />
        </div>
        <div className="form-field">
          <label htmlFor="externalRef">External reference (optional)</label>
          <input id="externalRef" type="text" value={externalRef} onChange={(e) => setExternalRef(e.target.value)} placeholder="e.g. roll number" />
        </div>

        <PhotoDropzone images={images} onImagesChange={setImages} />

        {error && <p className="error-text">{error}</p>}

        <button className="btn" type="submit" disabled={submitting} style={{ marginTop: 12 }}>
          {submitting ? "Enrolling…" : "Enroll student"}
        </button>
      </form>
    </div>
  );
}
