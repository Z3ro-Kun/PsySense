import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";
import "./Layout.css";

export function Layout({ children }: { children: ReactNode }) {
  const { logout, role } = useAuth();
  const isAdmin = role === "admin";
  return (
    <div className="layout-root">
      <header className="layout-header">
        <span className="layout-brand">PsySense</span>
        <nav className="layout-nav">
          {isAdmin && (
            <NavLink to="/" end className={({ isActive }) => (isActive ? "active" : undefined)}>
              Dashboard
            </NavLink>
          )}
          {isAdmin && (
            <NavLink to="/students" className={({ isActive }) => (isActive ? "active" : undefined)}>
              Students
            </NavLink>
          )}
          <NavLink to="/review" className={({ isActive }) => (isActive ? "active" : undefined)}>
            Review
          </NavLink>
          {isAdmin && (
            <NavLink to="/admin/reviewers" className={({ isActive }) => (isActive ? "active" : undefined)}>
              Reviewers
            </NavLink>
          )}
        </nav>
        <button className="layout-logout" onClick={logout}>
          Log out
        </button>
      </header>
      <main className="layout-main">{children}</main>
    </div>
  );
}
