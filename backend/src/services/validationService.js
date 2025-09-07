import mlService from './mlService.js';
import * as ocrService from './ocrService.js'; // 与下方 stub 对齐
import logger from '../utils/logger.js';

export async function validateVehicleData({ vehicleData, documents, validationLevel, userId }) {
  try {
    const validationId = `val_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`;
    logger.info(`Starting validation ${validationId}`, { userId, validationLevel });

    // Step 1: OCR
    let ocrResults = null;
    if (documents?.length) {
      ocrResults = await ocrService.processDocuments(documents);
    }

    // Step 2: ML
    const mlValidationResults = await mlService.validateVehicleData({
      vehicleData,
      validationLevel,
      ocrData: ocrResults?.extractedData
    });

    // Step 3: Cross-check
    const crossValidationResults = await performCrossValidation(vehicleData, ocrResults?.extractedData);

    // Step 4: Score
    const overallScore = calculateOverallScore(
      mlValidationResults,
      crossValidationResults,
      ocrResults?.confidence || 0
    );

    // Step 5: Compose
    const results = {
      overallScore,
      fieldValidations: mlValidationResults.fieldValidations,
      ocrResults: ocrResults
        ? { extractedData: ocrResults.extractedData, confidence: ocrResults.confidence }
        : null,
      crossValidation: crossValidationResults,
      recommendations: generateRecommendations(mlValidationResults)
    };

    return { id: validationId, results, processedAt: new Date().toISOString() };
  } catch (error) {
    logger.error('Validation service error:', error);
    throw error;
  }
}

export function performCrossValidation(vehicleData, ocrData) {
  const results = {
    vinMakeMatch: true,
    yearModelMatch: true,
    plateFormatValid: true
  };
  if (ocrData && vehicleData) {
    if (ocrData.make && vehicleData.make) {
      results.vinMakeMatch = ocrData.make.toLowerCase().includes(vehicleData.make.toLowerCase());
    }
    if (ocrData.year && vehicleData.year) {
      results.yearModelMatch = Math.abs(ocrData.year - vehicleData.year) <= 1;
    }
    if (vehicleData.plateNumber) {
      results.plateFormatValid = validateMalaysianPlateFormat(vehicleData.plateNumber);
    }
  }
  return results;
}

export function validateMalaysianPlateFormat(plateNumber) {
  const patterns = [
    /^[A-Z]{1,3}\s?\d{1,4}\s?[A-Z]?$/,
    /^W[A-Z]{2}\s?\d{1,4}$/
  ];
  return patterns.some(p => p.test(String(plateNumber).toUpperCase()));
}

export function calculateOverallScore(mlResults, crossValidation, ocrConfidence) {
  const weights = { mlValidation: 0.6, crossValidation: 0.3, ocrConfidence: 0.1 };
  const mlScore = mlResults.overallScore || 0;
  const crossScore = Object.values(crossValidation).filter(Boolean).length *
    (100 / Object.keys(crossValidation).length);
  return Math.round(mlScore * weights.mlValidation + crossScore * weights.crossValidation + ocrConfidence * weights.ocrConfidence);
}

export function generateRecommendations(mlResults) {
  const rec = [];
  Object.entries(mlResults.fieldValidations || {}).forEach(([field, v]) => {
    if (!v.isValid && v.suggestions) {
      rec.push({
        field,
        type: 'correction',
        message: `Consider: ${v.suggestions.join(', ')}`,
        priority: v.confidence < 50 ? 'high' : 'medium'
      });
    }
  });
  return rec;
}