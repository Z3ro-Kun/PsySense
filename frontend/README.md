# PsySense dashboard

React + Vite + TypeScript frontend for PsySense's `bridge_api` backend.
See the repo root `README.md` for the full setup/run/deploy story — this
file only covers frontend-specific dev commands.

```
npm install
npm run dev      # dev server; set VITE_API_URL to point at bridge_api
npm run build    # type-check (tsc -b) + production build to dist/
```

## Structure

```
src/
  api/            fetch client, typed endpoint wrappers, TS types mirroring api/schemas.py
  auth/            JWT + role session context, RequireAuth / RequireAdmin route guards
  components/      shared UI (LineChart, Layout, SeverityBadge, DateRangeFilter, PhotoDropzone)
  pages/           one file per route
  lib/             category->color mappings, chart data reshaping
  styles/          design tokens (theme.css) + shared component styles
```

## Routes

| Route | Who | Page |
|---|---|---|
| `/login` | anyone | `LoginPage` |
| `/` | admin only (reviewers redirect to `/review`) | `DashboardPage` |
| `/students`, `/students/enroll`, `/students/:id` | admin only | `StudentsPage`, `EnrollStudentPage`, `StudentDetailPage` |
| `/review` | admin + reviewer (server-scoped per reviewer) | `ReviewCandidatesPage` |
| `/admin/reviewers` | admin only | `ManageReviewersPage` |

Colors are assigned by category (emotion label, pose signal, severity) in
`lib/colors.ts` and never reassigned based on what's present in a given
chart -- see that file's comment for why.
