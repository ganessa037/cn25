# MyKereta — Smart Car Management

A web app for managing vehicles, documents, expenses, and renewals, powered by AI.
**Frontend**: React + TypeScript + Vite + Tailwind + Recharts
**Backend (API Gateway)**: Node.js + Express + Passport (Google OAuth) + JWT + Prisma + PostgreSQL
**AIML Service**: FastAPI (Python) — OCR/extraction, predictions, training jobs

---

## Architecture

```bash
Frontend (Vite/React)
        │   (Bearer JWT)
        ▼
Node/Express API  ──► PostgreSQL (Prisma)
        │
        ├── /api/auth/*  (Google OAuth)
        ├── /api/vehicles, /api/expenses, /api/documents, /api/notifications
        └── /api/ml/*  (proxy to FastAPI)
                         ▲
                         │  (internal HTTP, e.g. http://127.0.0.1:8001)
                 FastAPI (AIML: OCR, prediction, training)
```

    •	The frontend talks only to the Node/Express API (never directly to Python).
    •	The Express server proxies ML requests to the FastAPI service under /api/ml/*.
    •	Data is scoped per user via JWT; Google OAuth issues the token on sign-in.

## 🚀 Quick Start (Local)

### 0) Prerequisites

- **Node.js** ≥ 18
- **npm** ≥ 9
- **PostgreSQL** 15/16 (running locally)
- A **Google OAuth 2.0 Client** (Web application)

> **Mac (Homebrew) Postgres tips**
>
> ```bash
> brew install postgresql@16
> brew services start postgresql@16
> createdb cn25
> psql -h 127.0.0.1 -U "$USER" -d cn25 -c 'SELECT 1;'
> ```

---

## 1) Backend Setup

### Create backend/.env:

```bash
cd backend
npm install

# Server
PORT=3000
FRONTEND_URL=http://localhost:5173

# Google OAuth
GOOGLE_CLIENT_ID=YOUR_GOOGLE_CLIENT_ID
GOOGLE_CLIENT_SECRET=YOUR_GOOGLE_CLIENT_SECRET
GOOGLE_CALLBACK_URL=http://127.0.0.1:3000/api/auth/google/callback

# Database (adjust user if needed)
DATABASE_URL=postgresql://YOUR_OS_USER@127.0.0.1:5432/cn25

# JWT
JWT_SECRET=change_me_dev_secret
```

### Generate Prisma client & sync schema:

```bash
npx prisma generate
npx prisma db push
```

### Run the API:

```bash
npm run dev
# Health check:
# open http://127.0.0.1:3000/health  ->  {"status":"ok"}
```

### API base: http://127.0.0.1:3000/api

### OAuth endpoints:

    •	GET /api/auth/google
    •	GET /api/auth/google/callback

## 2) Google OAuth (Console) — required for Sign-in

In Google Cloud Console → APIs & Services → Credentials:
• Authorized JavaScript origins
• http://localhost:5173
• Authorized redirect URIs
• http://127.0.0.1:3000/api/auth/google/callback

Copy the Client ID and Client secret into backend/.env.
(Keep 127.0.0.1 in the callback exactly as shown.)

## 3) Frontend Setup

```bash
cd project
npm install
```

### Create project/.env

```bash
VITE_GOOGLE_CLIENT_ID=YOUR_GOOGLE_CLIENT_ID
VITE_BACKEND_URL=http://127.0.0.1:3000
# (optional) If you prefer /api explicitly:
# VITE_API_URL=http://127.0.0.1:3000/api
```

### Run the web app:

```bash
npm run dev
# open http://localhost:5173
```

Sign in at /signin → Google → you’ll be redirected to /dashboard.
Your JWT is stored at localStorage.user.token and sent to the API.

## 4) What’s Included

    •	Landing page with header links (Features, Security, FAQ)
    •	Protected App Shell (Header + Sidebar + Footer)
    •	Sidebar: hover-to-reveal + Pin mode (layout reflows)
    •	Header (App): Settings, Profile, Exit (to /landing)
    •	Brand shows the KeretaKu logo (put your image at project/public/keretaku-icon.png)
    •	Pages
    •	Dashboard (default after sign-in): KPIs, renewals, recent activity (reads /api/vehicles, /api/expenses, /api/documents)
    •	Analytics: read-only charts (Recharts) for the last 12 months
    •	Expenses: full CRUD (uses Prisma enum categories)
    •	Vehicles: full CRUD with compliance/maintenance dates
    •	Documents: upload (FormData) + list + preview text
    •	Notifications: computed alerts (expired/expiring)

All API requests include Authorization: Bearer <token> and are scoped to the signed-in user on the server.

## 5) Monorepo Layout

```bash
cn25/
├─ backend/                      # Express API (ESM) + Prisma + Passport Google
│  ├─ src/
│  │  ├─ routes/                 # /api/* routes (auth, vehicles, expenses, documents, ml)
│  │  ├─ controllers/            # OAuth + handlers
│  │  ├─ services/               # mlService.js (talks to FastAPI)
│  │  └─ middleware/             # auth (JWT), errorHandler, upload
│  ├─ prisma/schema.prisma       # User, Vehicle, Document, Expense, ValidationResult
│  └─ .env
├─ project/                      # React + TS + Vite + Tailwind
│  ├─ src/
│  │  ├─ layout/                 # Header, Sidebar, AppLayout, Footer
│  │  ├─ features/Dashboard/     # Analytics, ExpenseTracker, VehicleManager, DocumentManager, Notifications
│  │  └─ pages/                  # LandingPage, SignInPage, DashboardPage
│  ├─ public/keretaku-icon.png   # logo used in header
│  └─ .env
└─ (separate ML repo)            # FastAPI app (Python)
   └─ app/main.py                # /health, /predict, /extract-text, /train, /jobs/:id
```

## 6) Testing Flow (smoke tests)

    1.	Backend running → GET /health returns {"status":"ok"}.
    2.	Front-end running → open http://localhost:5173/signin and sign in.
    3.	Add a Vehicle and an Expense via the UI.
    4.	Verify data made it to the DB:

```bash
psql -h 127.0.0.1 -U "$USER" -d cn25 \
  -c 'SELECT COUNT(*) FROM "Vehicle";' \
  -c 'SELECT COUNT(*) FROM "Expense";'
```

    5.	Open Analytics and Dashboard—charts and KPIs should refresh.

## 7) Production Build

### Frontend:

```bash
cd project
npm run build
npm run preview  # serves dist/ locally for a quick test
```

### Backend (typical PM2 example):

```bash
# inside backend/
NODE_ENV=production npm start
# or with PM2:
# pm2 start server.js --name keretaku-api
```

Serve the built frontend with your favorite static host (Nginx, Vercel, etc.) and set FRONTEND_URL accordingly.

## 🔧 Troubleshooting

OAuth 404 at /auth/google
Use /api/auth/google and ensure FRONTEND_URL matches http://localhost:5173.

Callback mismatch
In Google Console, set redirect URI exactly to
http://127.0.0.1:3000/api/auth/google/callback.

role "postgres" does not exist
Use your macOS user in DATABASE_URL or create the role:

```bash
-- psql
CREATE ROLE postgres WITH LOGIN SUPERUSER PASSWORD 'postgres';
```

(or simply DATABASE_URL=postgresql://$USER@127.0.0.1:5432/cn25)

Prisma error: table does not exist
Run:

```bash
cd backend
npx prisma generate
npx prisma db push
```

CORS or 401 after sign-in
• FRONTEND_URL in backend .env must match the browser origin.
• Frontend must send Authorization: Bearer <token> (we store it in localStorage.user.token).

## 📜 Scripts

```bash
Backend
	•	npm run dev — dev server with nodemon
	•	npm start — production server
	•	npx prisma generate — (re)generate client
	•	npx prisma db push — sync schema to DB
```

```bash
Frontend
	•	npm run dev — Vite dev server
	•	npm run build — production build
	•	npm run preview — serve production build locally
```
