// lib/chartData.ts -- turns API rows into the {x, series} shape
// components/LineChart.tsx expects. Each event row already carries every
// series' value at one timestamp (see StudentDetailPage), so this is
// just a reshape, not an alignment problem.
import type { ChartSeries } from "../components/LineChart";
import type { EmotionEventOut, PoseEventOut } from "../api/types";
import { colorForEmotion, POSE_COLORS } from "./colors";

export function buildEmotionSeries(events: EmotionEventOut[]): { x: number[]; series: ChartSeries[] } {
  // Events come back newest-first from the API; charts read left-to-right.
  const ordered = [...events].reverse();
  const x = ordered.map((e) => new Date(e.timestamp).getTime());

  const labels = new Set<string>();
  for (const e of ordered) for (const key of Object.keys(e.emotions)) labels.add(key);

  const series: ChartSeries[] = Array.from(labels).map((label) => ({
    key: label,
    label,
    color: colorForEmotion(label),
    values: ordered.map((e) => e.emotions[label] ?? null),
  }));

  return { x, series };
}

export function buildPoseSeries(events: PoseEventOut[]): { x: number[]; series: ChartSeries[] } {
  const ordered = [...events].reverse();
  const x = ordered.map((e) => new Date(e.timestamp).getTime());

  const series: ChartSeries[] = [
    { key: "slumped_score", label: "Slumped", color: POSE_COLORS.slumped_score, values: ordered.map((e) => e.slumped_score) },
    { key: "rigidity_score", label: "Rigidity", color: POSE_COLORS.rigidity_score, values: ordered.map((e) => e.rigidity_score) },
    { key: "fidgeting_score", label: "Fidgeting", color: POSE_COLORS.fidgeting_score, values: ordered.map((e) => e.fidgeting_score) },
  ];

  return { x, series };
}
