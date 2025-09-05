const express = require('express');
const authRoutes = require('./auth');
const vehicleRoutes = require('./vehicles');
const documentRoutes = require('./documents');
const insuranceRoutes = require('./insurance');

const router = express.Router();

// Mount routes
router.use('/auth', authRoutes);
router.use('/vehicles', vehicleRoutes);
router.use('/documents', documentRoutes);
router.use('/insurance', insuranceRoutes);

// API info endpoint
router.get('/', (req, res) => {
  res.json({
    message: 'Vehicle Validation API v1.0',
    version: '1.0.0',
    endpoints: {
      auth: '/api/auth',
      vehicles: '/api/vehicles',
      documents: '/api/documents',
      insurance: '/api/insurance'
    }
  });
});

module.exports = router;