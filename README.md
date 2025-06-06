# PsySense 🧠
**Tagline:** Empowering mental wellness through real-time emotion detection.

## 📝 Description
PsySense is an AI-powered tool designed to monitor students' emotional states via camera feeds in classrooms. By analyzing facial expressions and behavioral cues, it helps detect signs of psychological conditions like anxiety or depression and flags students who may be at future risk.

## 🔍 Features
- Real-time emotion detection via webcam feed
- Local database (SQLite) to store per-student emotion data
- Risk prediction for psychological illnesses
- UI to:
  - Search students by ID
  - View students with current psychological concerns
  - View students at risk of future psychological distress

## 💡 Problem It Solves
In many schools, mental health issues among students often go unnoticed until it's too late. Lack of continuous monitoring makes early intervention difficult. PsySense bridges that gap by offering a non-intrusive way to keep track of students’ emotional well-being in real time.

## ⚙️ Tech Stack

| Component        | Technology            |
|------------------|------------------------|
| Frontend         | React (Planned)        |
| Backend          | Python (FastAPI)       |
| Emotion Detection| OpenCV, DeepFace       |
| Database         | SQLite                 |
| Visualization    | Chart.js / React Charts (Planned) |
| DevOps           | Git, GitHub            |

## ⚠️ Challenges Faced
- Camera indexing issues (smartphone/laptop camera mix-up)
- False positives from face detection (e.g., tube lights, obstructions)
- Resetting SQLite's auto-increment IDs after clearing data
- Crash/blue-screen issues due to webcam + real-time processing
- Ensuring modular backend for future UI integration

## 🚀 Installation & Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Z3ro-Kun/your-repo-name.git
cd your-repo-name
