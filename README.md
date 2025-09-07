# 🚗 MyKereta – Smart Vehicle-Management Platform  
Drive your life, not your paperwork

📄 [Presentation Slides](https://drive.google.com/drive/folders/1cqFF7POEtSZ79wl31A-WUBOUBdmzkWbW?usp=sharing)

**This is a project for CodeNection hackathon 2025, Track BIJAK.**

---

## 🌟 Project Overview  
KeretaKu fixes the “car-admin chaos” faced by Malaysian drivers:

| Problem | KeretaKu Fix |
|---------|--------------|
| Plate typos and wrong chassis numbers delay insurance | 📸 Plate-photo OCR cross-checks user input in real time |
| Hidden fuel & repair costs | 📊 Live dashboard shows true RM / km for every vehicle |
| Missed road-tax or insurance renewals | 🔔 Smart countdown alerts via e-mail / push |
| Scattered documents (Geran, IC, licence) | 🗂️ Encrypted vault with one-tap sharing |
| Re-typing the same details on every form | 🧩 Chrome extension auto-fills BJAK & JPJ pages |

---

## ✨ Key Features
- **Plate-Photo Validation** – Snap a picture; OCR verifies against typed plate.  
- **Doc Auto-Parse** – Upload JPJ card; make, model, year auto-populate.  
- **Expense Tracker** – Receipt OCR logs amount & station GPS in one tap.  
- **Smart Alerts** – Road-tax & insurance countdown with snooze / done.  
- **Chrome Auto-Fill** – One click completes insurance or loan forms.  
- **Fleet Mode** – Role-based access and cost-per-vehicle analytics.

---

## 🚀 Technical Stack
| Component | Technology |
|-----------|------------|
| **Frontend** | React + TypeScript, Vite, Tailwind CSS, TanStack Query |
| **Backend** | NestJS (Express), Passport (Google OAuth), JWT, Prisma ORM, PostgreSQL |
| **AI / ML** | FastAPI (Python), YOLOv8 (Ultralytics) & TrOCR (.pt from Hugging Face) |
| **Storage** | PostgreSQL (local / Supabase in production) |
| **Auth** | Google OAuth via Passport + JWT |
| **Dev Env** | .env config, Vite for front, Node.js for server |
| **ORM & Migrations** | Prisma Client + npx prisma migrate dev + prisma db push |
| **Deployment** | Vercel (Frontend) • Heroku (Backend) • FastAPI service (TBD: Render / AWS / Docker) |

---

## ⚡ Quick Start
👉 [see the Project README here](./project/README.md)
