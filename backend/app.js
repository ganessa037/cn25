import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import vehiclesRouter from './src/routes/vehicles.js';
import documentsRouter from './src/routes/documents.js';
import expensesRouter from './src/routes/expenses.js';
import authRouter from "./src/routes/auth.js";
import 'dotenv/config';

const app = express();

app.use(express.json({ limit: "10mb" }));
// 允许前端域名
// app.use(cors({ origin: process.env.FRONTEND_URL?.split(",") || "*", credentials: true }));

app.use("/api/auth", authRouter);

app.use(helmet());
app.use(cors({
  origin: process.env.FRONTEND_URL?.split(',') || '*',
  credentials: true
}));
app.use(morgan('dev'));
app.use(express.json({ limit: '10mb' })); // 允许 base64 文本
app.use(express.urlencoded({ extended: true }));

app.get('/health', (_req, res) => res.json({ ok: true }));

app.use('/api/vehicles', vehiclesRouter);
app.use('/api/documents', documentsRouter);
app.use('/api/expenses', expensesRouter);

export default app;