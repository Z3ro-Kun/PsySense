import { SEVERITY_COLORS } from "../lib/colors";
import "./SeverityBadge.css";

const ICONS: Record<string, string> = { info: "ℹ", warning: "▲", critical: "⛔" };

export function SeverityBadge({ severity }: { severity: string }) {
  // Status colors are never carried by hue alone (dataviz skill's
  // non-negotiable) -- icon + label ship alongside the color.
  return (
    <span className="severity-badge" style={{ color: SEVERITY_COLORS[severity] ?? "var(--text-muted)" }}>
      <span aria-hidden="true">{ICONS[severity] ?? "•"}</span> {severity}
    </span>
  );
}
