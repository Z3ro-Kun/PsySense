import { Link } from "react-router-dom";
import { telemetryApi } from "../api/endpoints";
import { useAsync } from "../hooks/useAsync";
import { SeverityBadge } from "../components/SeverityBadge";

export function DashboardPage() {
  const alerts = useAsync(() => telemetryApi.recentBehaviorEvents({ limit: 20 }), []);
  const sessions = useAsync(() => telemetryApi.listSessions({ limit: 20 }), []);

  return (
    <div>
      <h1 className="page-title">Dashboard</h1>

      <div className="card">
        <h2>Recent behavior alerts</h2>
        {alerts.loading && <p className="empty-state">Loading…</p>}
        {alerts.error && <p className="error-text">{alerts.error}</p>}
        {alerts.data && alerts.data.items.length === 0 && (
          <p className="empty-state">No behavior events flagged yet.</p>
        )}
        {alerts.data && alerts.data.items.length > 0 && (
          <table className="data-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Student</th>
                <th>Event</th>
                <th>Severity</th>
              </tr>
            </thead>
            <tbody>
              {alerts.data.items.map((event) => (
                <tr key={event.event_id}>
                  <td>{new Date(event.timestamp).toLocaleString()}</td>
                  <td>
                    {event.student_id ? (
                      <Link to={`/students/${event.student_id}`}>{event.student_id.slice(0, 8)}</Link>
                    ) : (
                      "Unknown"
                    )}
                  </td>
                  <td>{event.event_type.replaceAll("_", " ")}</td>
                  <td>
                    <SeverityBadge severity={event.severity} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="card">
        <h2>Recent sessions</h2>
        {sessions.loading && <p className="empty-state">Loading…</p>}
        {sessions.error && <p className="error-text">{sessions.error}</p>}
        {sessions.data && sessions.data.items.length === 0 && (
          <p className="empty-state">No tracked sessions yet -- the camera pipeline may not be running.</p>
        )}
        {sessions.data && sessions.data.items.length > 0 && (
          <table className="data-table">
            <thead>
              <tr>
                <th>Started</th>
                <th>Student</th>
                <th>Camera</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {sessions.data.items.map((session) => (
                <tr key={session.session_id}>
                  <td>{new Date(session.start_time).toLocaleString()}</td>
                  <td>
                    {session.student_id ? (
                      <Link to={`/students/${session.student_id}`}>{session.student_id.slice(0, 8)}</Link>
                    ) : (
                      "Unresolved"
                    )}
                  </td>
                  <td>{session.camera_id}</td>
                  <td>{session.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
