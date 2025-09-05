const mlService = require('./mlService');
const ocrService = require('./ocrService');
const logger = require('../utils/logger');

const validationService = {
  
  async validateVehicleData({ vehicleData, documents, validationLevel, userId }) {
    try {
      const validationId = `val_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      logger.info(`Starting validation ${validationId}`, { userId, validationLevel });

      // Step 1: Process documents with OCR 
      let ocrResults = null;
      if (documents && documents.length > 0) {
        ocrResults = await ocrService.processDocuments(documents);
      }

      // Step 2: Call ML service for validation
      const mlValidationResults = await mlService.validateVehicleData({
        vehicleData,
        validationLevel,
        ocrData: ocrResults?.extractedData
      });

      // Step 3: Perform cross-validation
      const crossValidationResults = await this.performCrossValidation(
        vehicleData, 
        ocrResults?.extractedData
      );

      // Step 4: Calculate overall confidence score
      const overallScore = this.calculateOverallScore(
        mlValidationResults,
        crossValidationResults,
        ocrResults?.confidence || 0
      );

      // Step 5: Compile final results
      const results = {
        overallScore,
        fieldValidations: mlValidationResults.fieldValidations,
        ocrResults: ocrResults ? {
          extractedData: ocrResults.extractedData,
          confidence: ocrResults.confidence
        } : null,
        crossValidation: crossValidationResults,
        recommendations: this.generateRecommendations(mlValidationResults)
      };

      // Step 6: Save validation results (optional)
      // await this.saveValidationResults(validationId, results, userId);

      return {
        id: validationId,
        results,
        processedAt: new Date().toISOString()
      };

    } catch (error) {
      logger.error('Validation service error:', error);
      throw error;
    }
  },

  async performCrossValidation(vehicleData, ocrData) {
    // Cross-validate vehicle data with OCR extracted data
    const results = {
      vinMakeMatch: true,
      yearModelMatch: true,
      plateFormatValid: true
    };

    // Implementation of cross-validation logic
    if (ocrData && vehicleData) {
      // Compare VIN with make
      if (ocrData.make && vehicleData.make) {
        results.vinMakeMatch = ocrData.make.toLowerCase().includes(
          vehicleData.make.toLowerCase()
        );
      }
      
      // Compare year and model
      if (ocrData.year && vehicleData.year) {
        results.yearModelMatch = Math.abs(ocrData.year - vehicleData.year) <= 1;
      }
      
      // Validate plate format (Malaysia specific)
      if (vehicleData.plateNumber) {
        results.plateFormatValid = this.validateMalaysianPlateFormat(
          vehicleData.plateNumber
        );
      }
    }

    return results;
  },

  validateMalaysianPlateFormat(plateNumber) {
    // Malaysian plate number formats
    const patterns = [
      /^[A-Z]{1,3}\s?\d{1,4}\s?[A-Z]?$/,  // Standard format
      /^W[A-Z]{2}\s?\d{1,4}$/,            // Wilayah format
      // Add more Malaysian plate patterns as needed
    ];

    return patterns.some(pattern => pattern.test(plateNumber.toUpperCase()));
  },

  calculateOverallScore(mlResults, crossValidation, ocrConfidence) {
    // Weight different validation aspects
    const weights = {
      mlValidation: 0.6,
      crossValidation: 0.3,
      ocrConfidence: 0.1
    };

    const mlScore = mlResults.overallScore || 0;
    const crossScore = Object.values(crossValidation).filter(Boolean).length * 
                      (100 / Object.keys(crossValidation).length);

    return Math.round(
      (mlScore * weights.mlValidation) +
      (crossScore * weights.crossValidation) +
      (ocrConfidence * weights.ocrConfidence)
    );
  },

  generateRecommendations(mlResults) {
    const recommendations = [];
    
    // Generate recommendations based on ML results
    Object.entries(mlResults.fieldValidations || {}).forEach(([field, validation]) => {
      if (!validation.isValid && validation.suggestions) {
        recommendations.push({
          field,
          type: 'correction',
          message: `Consider: ${validation.suggestions.join(', ')}`,
          priority: validation.confidence < 50 ? 'high' : 'medium'
        });
      }
    });

    return recommendations;
  }
};

module.exports = validationService;