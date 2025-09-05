const axios = require('axios');
const config = require('../config/config');
const logger = require('../utils/logger');

const ML_SERVICE_URL = config.mlService.url;

const mlService = {
  
  async validateVehicleData({ vehicleData, validationLevel, ocrData }) {
    try {
      const response = await axios.post(`${ML_SERVICE_URL}/ml/validate-vehicle-data`, {
        vehicleData,
        ocrData,
        validationRules: this.getValidationRules(validationLevel),
        region: 'MY' // Malaysia
      }, {
        timeout: 30000, // 30 seconds timeout
        headers: {
          'Content-Type': 'application/json'
        }
      });

      return response.data.validationResults;
    } catch (error) {
      logger.error('ML Service validation failed:', error.message);
      
      if (error.code === 'ECONNREFUSED') {
        throw new Error('ML service is unavailable. Please try again later.');
      }
      
      throw new Error(`ML validation failed: ${error.message}`);
    }
  },

  async extractTextFromImage(imageData, documentType) {
    try {
      const response = await axios.post(`${ML_SERVICE_URL}/ocr/extract-text`, {
        image: imageData,
        documentType,
        preprocessing: {
          denoise: true,
          rotate: true,
          enhanceContrast: true
        }
      }, {
        timeout: 60000, // OCR can take longer
        headers: {
          'Content-Type': 'application/json'
        }
      });

      return response.data;
    } catch (error) {
      logger.error('OCR extraction failed:', error.message);
      throw new Error(`OCR processing failed: ${error.message}`);
    }
  },

  getValidationRules(validationLevel) {
    const rules = {
      basic: ['format'],
      standard: ['format', 'consistency'],
      comprehensive: ['format', 'consistency', 'existence', 'cross_reference']
    };

    return rules[validationLevel] || rules.standard;
  },

  // Health check for ML service
  async healthCheck() {
    try {
      const response = await axios.get(`${ML_SERVICE_URL}/health`, {
        timeout: 5000
      });
      return response.status === 200;
    } catch (error) {
      logger.warn('ML service health check failed:', error.message);
      return false;
    }
  }
};

module.exports = mlService;