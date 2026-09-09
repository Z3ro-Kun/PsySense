// components/DateRangeFilter.tsx -- preset rows, per the dataviz skill's
// filter-controls guidance (date-range = preset list, not a raw picker).
import "./DateRangeFilter.css";

export interface DateRange {
  label: string;
  since?: string; // ISO, undefined = no lower bound
}

const PRESETS: DateRange[] = [
  { label: "Last 24 hours", since: new Date(Date.now() - 24 * 3600 * 1000).toISOString() },
  { label: "Last 7 days", since: new Date(Date.now() - 7 * 24 * 3600 * 1000).toISOString() },
  { label: "Last 30 days", since: new Date(Date.now() - 30 * 24 * 3600 * 1000).toISOString() },
  { label: "All time", since: undefined },
];

export function DateRangeFilter({ value, onChange }: { value: string; onChange: (label: string) => void }) {
  return (
    <div className="daterange-filter" role="radiogroup" aria-label="Date range">
      {PRESETS.map((preset) => (
        <button
          key={preset.label}
          type="button"
          role="radio"
          aria-checked={value === preset.label}
          className={`daterange-option${value === preset.label ? " daterange-option-active" : ""}`}
          onClick={() => onChange(preset.label)}
        >
          {value === preset.label && <span aria-hidden="true">✓ </span>}
          {preset.label}
        </button>
      ))}
    </div>
  );
}

export function sinceForLabel(label: string): string | undefined {
  return PRESETS.find((p) => p.label === label)?.since;
}

export const DEFAULT_RANGE_LABEL = "Last 7 days";
