import express from "express";
import cors from "cors";
import helmet from "helmet";
import morgan from "morgan";
import vehiclesRouter from "./src/routes/vehicles.js";
import documentsRouter from "./src/routes/documents.js";
import expensesRouter from "./src/routes/expenses.js";
import authRouter from "./src/routes/auth.js";
import config from "./src/config/config.js";

const app = express();

// Trust proxy (useful if behind a proxy in prod)
app.set("trust proxy", 1);

// CORS must match your frontend origin exactly
app.use(
  cors({
    origin: config.frontendUrl,
    credentials: true,
    methods: ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
  })
);

app.use(helmet());
app.use(morgan("dev"));
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));

app.get("/health", (_req, res) => res.json({ ok: true }));

// --- Mount API routers ---
app.use("/api/auth", authRouter);
app.use("/api/vehicles", vehiclesRouter);
app.use("/api/documents", documentsRouter);
app.use("/api/expenses", expensesRouter);

export default app;