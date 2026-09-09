// api/types.ts -- mirrors psysense/api/schemas.py. Kept as a thin,
// hand-written mirror rather than codegen since the API surface is small
// and stable; revisit if it grows enough to drift.

export interface Paginated<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in_minutes: number;
  role: string;
}

export interface StudentOut {
  student_id: string;
  display_name: string;
  external_ref: string | null;
  enrolled_at: string;
  active: boolean;
}

export interface EnrollmentResult {
  student: StudentOut;
  accepted_images: number;
  skipped_images: string[];
}

export type SessionStatus = "active" | "closed" | "lost";

export interface SessionOut {
  session_id: string;
  student_id: string | null;
  tracker_id: number;
  camera_id: string;
  start_time: string;
  end_time: string | null;
  status: SessionStatus;
}

export interface EmotionEventOut {
  event_id: number;
  session_id: string | null;
  student_id: string | null;
  timestamp: string;
  emotions: Record<string, number>;
  dominant_emotion: string | null;
}

export interface PoseEventOut {
  event_id: number;
  session_id: string | null;
  student_id: string | null;
  timestamp: string;
  slumped_score: number;
  rigidity_score: number;
  fidgeting_score: number;
}

export type BehaviorSeverity = "info" | "warning" | "critical";

export interface BehaviorEventOut {
  event_id: number;
  session_id: string | null;
  student_id: string | null;
  timestamp: string;
  event_type: string;
  payload: Record<string, unknown>;
  severity: BehaviorSeverity;
}

export interface ApiErrorBody {
  error: { code: string; message: string };
}

export type UserRole = "admin" | "reviewer";

export interface UserOut {
  user_id: string;
  username: string;
  role: UserRole;
  active: boolean;
  created_at: string;
  assigned_student_count: number;
}

export interface AssignmentsOut {
  student_ids: string[];
}

export interface PendingCandidateOut {
  candidate_id: string;
  student_id: string;
  student_display_name: string;
  tracker_id: number | null;
  session_id: string | null;
  distance: number;
  captured_at: string;
  image_b64: string;
}
