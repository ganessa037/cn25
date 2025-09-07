import { Router } from 'express';
import authRoutes from './auth.js';
import vehicleRoutes from './vehicles.js';
import documentRoutes from './documents.js';
import expenseRoutes from './expenses.js';
import ml from "./ml.js";

const router = Router();

// Mount under /api
router.use('/auth', authRoutes);
router.use('/vehicles', vehicleRoutes);
router.use('/documents', documentRoutes);
router.use('/expenses', expenseRoutes);
router.use("/ml", ml);

// API info
router.get('/', (_req, res) => {
  res.json({
    message: 'Vehicle Validation API v1.0',
    version: '1.0.0',
    endpoints: {
      auth: '/api/auth',
      vehicles: '/api/vehicles',
      documents: '/api/documents',
      expenses: '/api/expenses',
    },
  });
});

export default router;