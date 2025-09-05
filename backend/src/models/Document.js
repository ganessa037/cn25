const { DataTypes } = require('sequelize');
const { sequelize } = require('../config/database');

const Document = sequelize.define('Document', {
  id: {
    type: DataTypes.UUID,
    defaultValue: DataTypes.UUIDV4,
    primaryKey: true
  },
  vehicleId: {
    type: DataTypes.UUID,
    references: {
      model: 'vehicles',
      key: 'id'
    }
  },
  userId: {
    type: DataTypes.UUID,
    allowNull: false,
    references: {
      model: 'users',
      key: 'id'
    }
  },
  fileName: {
    type: DataTypes.STRING,
    allowNull: false
  },
  originalName: {
    type: DataTypes.STRING,
    allowNull: false
  },
  filePath: {
    type: DataTypes.STRING,
    allowNull: false
  },
  fileSize: {
    type: DataTypes.INTEGER
  },
  mimeType: {
    type: DataTypes.STRING
  },
  documentType: {
    type: DataTypes.ENUM('registration', 'insurance_card', 'license_plate_photo', 'identity_card', 'other'),
    allowNull: false
  },
  ocrResults: {
    type: DataTypes.JSONB
  },
  processingStatus: {
    type: DataTypes.ENUM('uploaded', 'processing', 'processed', 'failed'),
    defaultValue: 'uploaded'
  },
  extractedData: {
    type: DataTypes.JSONB
  },
  confidenceScore: {
    type: DataTypes.INTEGER,
    validate: {
      min: 0,
      max: 100
    }
  },
  errorMessage: {
    type: DataTypes.TEXT
  }
}, {
  tableName: 'documents',
  indexes: [
    {
      fields: ['userId']
    },
    {
      fields: ['vehicleId']
    },
    {
      fields: ['documentType']
    }
  ]
});

module.exports = Document;