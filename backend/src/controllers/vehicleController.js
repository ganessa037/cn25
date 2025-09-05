const vehicleService = require('../services/vehicleService');
const validationService = require('../services/validationService');
const logger = require('../utils/logger');

const vehicleController = {
  
  // GET /api/vehicles
  getAllVehicles: async (req, res, next) => {
    try {
      const userId = req.user.id;
      const vehicles = await vehicleService.getUserVehicles(userId);
      
      res.json({
        success: true,
        data: vehicles,
        count: vehicles.length
      });
    } catch (error) {
      logger.error('Error fetching vehicles:', error);
      next(error);
    }
  },

  // POST /api/vehicles/validate - Main validation endpoint
  validateVehicle: async (req, res, next) => {
    try {
      const { vehicleData, documents, validationLevel = 'standard' } = req.body;
      
      logger.info('Starting vehicle validation', { 
        userId: req.user.id, 
        validationLevel 
      });

      // Orchestrate the validation process
      const validationResult = await validationService.validateVehicleData({
        vehicleData,
        documents,
        validationLevel,
        userId: req.user.id
      });

      res.json({
        success: true,
        validationId: validationResult.id,
        results: validationResult.results,
        processedAt: validationResult.processedAt
      });

    } catch (error) {
      logger.error('Vehicle validation failed:', error);
      next(error);
    }
  },

  // GET /api/vehicles/:id
  getVehicleById: async (req, res, next) => {
    try {
      const { id } = req.params;
      const userId = req.user.id;
      
      const vehicle = await vehicleService.getVehicleById(id, userId);
      
      if (!vehicle) {
        return res.status(404).json({
          success: false,
          message: 'Vehicle not found'
        });
      }

      res.json({
        success: true,
        data: vehicle
      });
    } catch (error) {
      logger.error('Error fetching vehicle:', error);
      next(error);
    }
  },

  // POST /api/vehicles
  createVehicle: async (req, res, next) => {
    try {
      const vehicleData = req.body;
      const userId = req.user.id;

      const newVehicle = await vehicleService.createVehicle({
        ...vehicleData,
        userId
      });

      res.status(201).json({
        success: true,
        data: newVehicle,
        message: 'Vehicle created successfully'
      });
    } catch (error) {
      logger.error('Error creating vehicle:', error);
      next(error);
    }
  },

  // Additional methods...
  updateVehicle: async (req, res, next) => {
    // Implementation for PUT /api/vehicles/:id
  },

  deleteVehicle: async (req, res, next) => {
    // Implementation for DELETE /api/vehicles/:id
  },

  batchValidateVehicles: async (req, res, next) => {
    // Implementation for POST /api/vehicles/batch-validate
  },

  getFieldSuggestions: async (req, res, next) => {
    // Implementation for GET /api/vehicles/suggestions/:field
  }

};

module.exports = vehicleController;