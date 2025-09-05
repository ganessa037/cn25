const { DataTypes } = require('sequelize');
const { sequelize } = require('../config/database');

const Vehicle = sequelize.define('Vehicle', {
  id: {
    type: DataTypes.UUID,
    defaultValue: DataTypes.UUIDV4,
    primaryKey: true
  },
  userId: {
    type: DataTypes.UUID,
    allowNull: false,
    references: {
      model: 'users',
      key: 'id'
    }
  },
  plateNumber: {
    type: DataTypes.STRING(20),
    allowNull: false,
    validate: {
      notEmpty: {
        msg: 'Plate number is required'
      }
    }
  },
  make: {
    type: DataTypes.STRING(50),
    allowNull: false,
    validate: {
      notEmpty: {
        msg: 'Vehicle make is required'
      }
    }
  },
  model: {
    type: DataTypes.STRING(50),
    allowNull: false,
    validate: {
      notEmpty: {
        msg: 'Vehicle model is required'
      }
    }
  },
  year: {
    type: DataTypes.INTEGER,
    allowNull: false,
    validate: {
      min: {
        args: [1900],
        msg: 'Year must be 1900 or later'
      },
      max: {
        args: [new Date().getFullYear() + 1],
        msg: 'Year cannot be in the future'
      }
    }
  },
  vin: {
    type: DataTypes.STRING(17),
    validate: {
      len: {
        args: [17, 17],
        msg: 'VIN must be exactly 17 characters'
      }
    }
  },
  engineSize: {
    type: DataTypes.STRING(20)
  },
  fuelType: {
    type: DataTypes.ENUM('Petrol', 'Diesel', 'Hybrid', 'Electric', 'CNG'),
    defaultValue: 'Petrol'
  },
  transmission: {
    type: DataTypes.ENUM('Manual', 'Automatic', 'CVT'),
    defaultValue: 'Manual'
  },
  color: {
    type: DataTypes.STRING(30)
  },
  validationStatus: {
    type: DataTypes.ENUM('pending', 'validated', 'failed', 'needs_review'),
    defaultValue: 'pending'
  },
  validationScore: {
    type: DataTypes.INTEGER,
    validate: {
      min: 0,
      max: 100
    }
  },
  isActive: {
    type: DataTypes.BOOLEAN,
    defaultValue: true
  },
  notes: {
    type: DataTypes.TEXT
  }
}, {
  tableName: 'vehicles',
  indexes: [
    {
      fields: ['userId']
    },
    {
      fields: ['plateNumber']
    },
    {
      fields: ['validationStatus']
    }
  ]
});

module.exports = Vehicle;