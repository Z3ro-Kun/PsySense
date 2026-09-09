import { Navigate } from "react-router-dom";
import type { ReactNode } from "react";
import { useAuth } from "./AuthContext";

/** Gates admin-only routes (students, enroll, delete, manage reviewers).
 * A logged-in reviewer hitting one of these is redirected to /review --
 * not /login, since they ARE authenticated, just not authorized for this
 * route. Assumes it's nested inside RequireAuth (so role is never null
 * here in practice), but treats null defensively the same as reviewer. */
export function RequireAdmin({ children }: { children: ReactNode }) {
  const { role } = useAuth();
  if (role !== "admin") return <Navigate to="/review" replace />;
  return <>{children}</>;
}
