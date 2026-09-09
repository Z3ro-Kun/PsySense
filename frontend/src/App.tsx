import { Navigate, Route, Routes } from "react-router-dom";
import { AuthProvider, useAuth } from "./auth/AuthContext";
import { RequireAuth } from "./auth/RequireAuth";
import { RequireAdmin } from "./auth/RequireAdmin";
import { Layout } from "./components/Layout";
import { LoginPage } from "./pages/LoginPage";
import { DashboardPage } from "./pages/DashboardPage";
import { StudentsPage } from "./pages/StudentsPage";
import { EnrollStudentPage } from "./pages/EnrollStudentPage";
import { StudentDetailPage } from "./pages/StudentDetailPage";
import { ReviewCandidatesPage } from "./pages/ReviewCandidatesPage";
import { ManageReviewersPage } from "./pages/ManageReviewersPage";

/** Admins land on the Dashboard; reviewers only ever have the Review
 * page to go to, so send them straight there instead of a Dashboard
 * that would otherwise need its own admin gate. */
function Home() {
  const { role } = useAuth();
  return role === "admin" ? <DashboardPage /> : <Navigate to="/review" replace />;
}

export function App() {
  return (
    <AuthProvider>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route
          path="/*"
          element={
            <RequireAuth>
              <Layout>
                <Routes>
                  <Route index element={<Home />} />
                  <Route path="students" element={<RequireAdmin><StudentsPage /></RequireAdmin>} />
                  <Route path="students/enroll" element={<RequireAdmin><EnrollStudentPage /></RequireAdmin>} />
                  <Route path="students/:studentId" element={<RequireAdmin><StudentDetailPage /></RequireAdmin>} />
                  <Route path="review" element={<ReviewCandidatesPage />} />
                  <Route path="admin/reviewers" element={<RequireAdmin><ManageReviewersPage /></RequireAdmin>} />
                  <Route path="*" element={<Navigate to="/" replace />} />
                </Routes>
              </Layout>
            </RequireAuth>
          }
        />
      </Routes>
    </AuthProvider>
  );
}
