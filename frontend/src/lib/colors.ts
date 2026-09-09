// lib/colors.ts -- fixed category -> color-var assignments. Per the
// dataviz skill's non-negotiable, hues are assigned in a FIXED order per
// label, never reassigned based on which labels happen to be present in
// a given chart (that would make a series change color when a filter
// changes, which breaks "color follows the entity, never its rank").

export const EMOTION_COLORS: Record<string, string> = {
  happy: "var(--series-yellow)",
  sad: "var(--series-blue)",
  angry: "var(--series-red)",
  fear: "var(--series-violet)",
  disgust: "var(--series-green)",
  surprise: "var(--series-orange)",
  neutral: "var(--series-aqua)",
};
const FALLBACK_EMOTION_COLOR = "var(--series-magenta)";

export function colorForEmotion(label: string): string {
  return EMOTION_COLORS[label] ?? FALLBACK_EMOTION_COLOR;
}

export const POSE_COLORS: Record<"slumped_score" | "rigidity_score" | "fidgeting_score", string> = {
  slumped_score: "var(--series-blue)",
  rigidity_score: "var(--series-orange)",
  fidgeting_score: "var(--series-aqua)",
};

export const SEVERITY_COLORS: Record<string, string> = {
  info: "var(--text-muted)",
  warning: "var(--status-warning)",
  critical: "var(--status-critical)",
};
