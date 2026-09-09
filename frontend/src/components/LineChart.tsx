// components/LineChart.tsx -- a small generic multi-series time-line
// chart (used for both the emotion trend and the pose trend on the
// student detail page). All series here already share the same x-axis
// (they come from the same rows -- see StudentDetailPage), so hover only
// needs one nearest-index lookup, not per-series alignment.
//
// Follows the dataviz skill's non-negotiables: one axis (never dual),
// thin 2px lines, a legend whenever there's more than one series, a
// hover crosshair + tooltip, and category colors assigned by the caller
// (lib/colors.ts) rather than cycled here.
import { useMemo, useState } from "react";
import "./LineChart.css";

export interface ChartSeries {
  key: string;
  label: string;
  color: string; // a var(--series-*) reference from lib/colors.ts
  values: (number | null)[]; // same length as `x`
}

interface LineChartProps {
  x: number[]; // epoch ms, ascending
  series: ChartSeries[];
  height?: number;
  yDomain?: [number, number];
  yFormat?: (v: number) => string;
  xFormat?: (t: number) => string;
  emptyLabel?: string;
}

const PADDING = { top: 12, right: 16, bottom: 24, left: 36 };

export function LineChart({
  x,
  series,
  height = 220,
  yDomain = [0, 1],
  yFormat = (v) => v.toFixed(2),
  xFormat = (t) => new Date(t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
  emptyLabel = "No data in this range.",
}: LineChartProps) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const width = 640;
  const plotW = width - PADDING.left - PADDING.right;
  const plotH = height - PADDING.top - PADDING.bottom;

  const xScale = useMemo(() => {
    if (x.length === 0) return (i: number) => i;
    const [xMin, xMax] = [x[0], x[x.length - 1]];
    const span = xMax - xMin || 1;
    return (value: number) => PADDING.left + ((value - xMin) / span) * plotW;
  }, [x, plotW]);

  const yScale = (value: number) => {
    const [yMin, yMax] = yDomain;
    const span = yMax - yMin || 1;
    return PADDING.top + plotH - ((value - yMin) / span) * plotH;
  };

  if (x.length === 0) {
    return (
      <div className="linechart-empty" style={{ height }}>
        {emptyLabel}
      </div>
    );
  }

  const gridLines = 4;
  const handleMove = (evt: React.MouseEvent<SVGRectElement>) => {
    const rect = evt.currentTarget.getBoundingClientRect();
    const px = evt.clientX - rect.left;
    const ratio = Math.min(1, Math.max(0, (px - PADDING.left) / plotW));
    const idx = Math.round(ratio * (x.length - 1));
    setHoverIndex(idx);
  };

  return (
    <div className="linechart-root">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Trend chart">
        {Array.from({ length: gridLines + 1 }, (_, i) => {
          const value = yDomain[0] + ((yDomain[1] - yDomain[0]) * i) / gridLines;
          const y = yScale(value);
          return (
            <g key={i}>
              <line x1={PADDING.left} x2={width - PADDING.right} y1={y} y2={y} className="linechart-grid" />
              <text x={PADDING.left - 6} y={y + 3} textAnchor="end" className="linechart-axis-label">
                {yFormat(value)}
              </text>
            </g>
          );
        })}
        <line
          x1={PADDING.left} x2={width - PADDING.right}
          y1={PADDING.top + plotH} y2={PADDING.top + plotH}
          className="linechart-baseline"
        />

        {series.map((s) => {
          const points = s.values
            .map((v, i) => (v === null ? null : `${xScale(x[i])},${yScale(v)}`))
            .filter((p): p is string => p !== null);
          if (points.length === 0) return null;
          return <polyline key={s.key} points={points.join(" ")} style={{ stroke: s.color }} className="linechart-line" />;
        })}

        {hoverIndex !== null && (
          <line
            x1={xScale(x[hoverIndex])} x2={xScale(x[hoverIndex])}
            y1={PADDING.top} y2={PADDING.top + plotH}
            className="linechart-crosshair"
          />
        )}
        {hoverIndex !== null &&
          series.map((s) => {
            const v = s.values[hoverIndex];
            if (v === null) return null;
            return (
              <circle key={s.key} cx={xScale(x[hoverIndex])} cy={yScale(v)} r={3.5} style={{ fill: s.color }} />
            );
          })}

        {/* Hit-target overlay, wider than the marks themselves per the skill's interaction guidance. */}
        <rect
          x={PADDING.left} y={PADDING.top} width={plotW} height={plotH}
          fill="transparent"
          onMouseMove={handleMove}
          onMouseLeave={() => setHoverIndex(null)}
        />
      </svg>

      {hoverIndex !== null && (
        <div className="linechart-tooltip">
          <div className="linechart-tooltip-time">{xFormat(x[hoverIndex])}</div>
          {series.map((s) => {
            const v = s.values[hoverIndex];
            return (
              <div key={s.key} className="linechart-tooltip-row">
                <span className="linechart-swatch" style={{ background: s.color }} />
                <span className="linechart-tooltip-label">{s.label}</span>
                <span className="linechart-tooltip-value">{v === null ? "—" : yFormat(v)}</span>
              </div>
            );
          })}
        </div>
      )}

      {series.length >= 2 && (
        <div className="linechart-legend">
          {series.map((s) => (
            <span key={s.key} className="linechart-legend-item">
              <span className="linechart-swatch" style={{ background: s.color }} />
              {s.label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
