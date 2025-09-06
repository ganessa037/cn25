import { Router } from 'express';
import authRoutes from './auth.js';
import vehicleRoutes from './vehicles.js';
import documentRoutes from './documents.js';
import insuranceRoutes from './insurance.js';

const router = Router();

// Mount routes under /api (app.js already does app.use('/api', routes))
router.use('/auth', authRoutes);
router.use('/vehicles', vehicleRoutes);
router.use('/documents', documentRoutes);
router.use('/insurance', insuranceRoutes);

// API info
router.get('/', (_req, res) => {
  res.json({
    message: 'Vehicle Validation API v1.0',
    version: '1.0.0',
    endpoints: {
      auth: '/api/auth',
      vehicles: '/api/vehicles',
      documents: '/api/documents',
      insurance: '/api/insurance',
    },
  });
});

export default router;