// api/endpoints.ts -- one function per bridge_api route, typed. Pages
// call these instead of `api.get`/`api.post` directly with inline paths.
import { api } from "./client";
import type {
  AssignmentsOut,
  BehaviorEventOut,
  EmotionEventOut,
  EnrollmentResult,
  Paginated,
  PendingCandidateOut,
  PoseEventOut,
  SessionOut,
  StudentOut,
  TokenResponse,
  UserOut,
  UserRole,
} from "./types";

export interface TimeRangeParams {
  since?: string;
  until?: string;
  limit?: number;
  offset?: number;
}

export const authApi = {
  login: (username: string, password: string) => api.post<TokenResponse>("/api/v1/auth/login", { username, password }),
};

export const studentsApi = {
  list: (params?: { search?: string; limit?: number; offset?: number }) =>
    api.get<Paginated<StudentOut>>("/api/v1/students", params),
  get: (studentId: string) => api.get<StudentOut>(`/api/v1/students/${studentId}`),
  enroll: (displayName: string, externalRef: string | undefined, images: File[]) => {
    const form = new FormData();
    form.append("display_name", displayName);
    if (externalRef) form.append("external_ref", externalRef);
    for (const image of images) form.append("images", image);
    return api.postForm<EnrollmentResult>("/api/v1/students", form);
  },
  delete: (studentId: string) => api.del<void>(`/api/v1/students/${studentId}`),
  addPhotos: (studentId: string, images: File[]) => {
    const form = new FormData();
    for (const image of images) form.append("images", image);
    return api.postForm<EnrollmentResult>(`/api/v1/students/${studentId}/photos`, form);
  },
};

export const telemetryApi = {
  listSessions: (params?: TimeRangeParams & { student_id?: string }) =>
    api.get<Paginated<SessionOut>>("/api/v1/sessions", params),
  studentEmotions: (studentId: string, params?: TimeRangeParams) =>
    api.get<Paginated<EmotionEventOut>>(`/api/v1/students/${studentId}/emotions`, params),
  studentPose: (studentId: string, params?: TimeRangeParams) =>
    api.get<Paginated<PoseEventOut>>(`/api/v1/students/${studentId}/pose`, params),
  studentBehaviorEvents: (studentId: string, params?: TimeRangeParams & { severity?: string }) =>
    api.get<Paginated<BehaviorEventOut>>(`/api/v1/students/${studentId}/behavior-events`, params),
  recentBehaviorEvents: (params?: TimeRangeParams & { severity?: string }) =>
    api.get<Paginated<BehaviorEventOut>>("/api/v1/behavior-events", params),
};

export const identityCandidatesApi = {
  list: (params?: { limit?: number; offset?: number }) =>
    api.get<Paginated<PendingCandidateOut>>("/api/v1/identity-candidates", params),
  approve: (candidateId: string) => api.post<void>(`/api/v1/identity-candidates/${candidateId}/approve`),
  reject: (candidateId: string) => api.post<void>(`/api/v1/identity-candidates/${candidateId}/reject`),
};

export const usersApi = {
  list: () => api.get<UserOut[]>("/api/v1/users"),
  create: (username: string, password: string, role: UserRole) =>
    api.post<UserOut>("/api/v1/users", { username, password, role }),
  deactivate: (userId: string) => api.post<void>(`/api/v1/users/${userId}/deactivate`),
  reactivate: (userId: string) => api.post<void>(`/api/v1/users/${userId}/reactivate`),
  getAssignments: (userId: string) => api.get<AssignmentsOut>(`/api/v1/users/${userId}/assignments`),
  setAssignments: (userId: string, studentIds: string[]) =>
    api.put<AssignmentsOut>(`/api/v1/users/${userId}/assignments`, { student_ids: studentIds }),
};
