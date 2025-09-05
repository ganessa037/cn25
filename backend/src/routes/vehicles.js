const express = require('express');
const vehicleController = require('../controllers/vehicleController');
const authMiddleware = require('../middleware/auth');
const validationMiddleware = require('../middleware/validation');

const router = express.Router();

// Apply authentication to all vehicle routes
router.use(authMiddleware);

// Vehicle CRUD routes
router.get('/', vehicleController.getAllVehicles);
router.get('/:id', vehicleController.getVehicleById);
router.post('/', validationMiddleware.validateVehicleData, vehicleController.createVehicle);
router.put('/:id', validationMiddleware.validateVehicleData, vehicleController.updateVehicle);
router.delete('/:id', vehicleController.deleteVehicle);

// Vehicle validation routes
router.post('/validate', validationMiddleware.validateVehicleInput, vehicleController.validateVehicle);
router.post('/batch-validate', vehicleController.batchValidateVehicles);
router.get('/suggestions/:field', vehicleController.getFieldSuggestions);

module.exports = router;