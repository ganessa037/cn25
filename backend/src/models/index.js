const { sequelize } = require('../config/database');
const User = require('./User');
const Vehicle = require('./Vehicle');
const ValidationResult = require('./ValidationResult');
const Document = require('./Document');

// Define associations
User.hasMany(Vehicle, {
  foreignKey: 'userId',
  as: 'vehicles'
});

Vehicle.belongsTo(User, {
  foreignKey: 'userId',
  as: 'user'
});

User.hasMany(ValidationResult, {
  foreignKey: 'userId',
  as: 'validationResults'
});

ValidationResult.belongsTo(User, {
  foreignKey: 'userId',
  as: 'user'
});

Vehicle.hasMany(ValidationResult, {
  foreignKey: 'vehicleId',
  as: 'validationResults'
});

ValidationResult.belongsTo(Vehicle, {
  foreignKey: 'vehicleId',
  as: 'vehicle'
});

User.hasMany(Document, {
  foreignKey: 'userId',
  as: 'documents'
});

Document.belongsTo(User, {
  foreignKey: 'userId',
  as: 'user'
});

Vehicle.hasMany(Document, {
  foreignKey: 'vehicleId',
  as: 'documents'
});

Document.belongsTo(Vehicle, {
  foreignKey: 'vehicleId',
  as: 'vehicle'
});

module.exports = {
  sequelize,
  User,
  Vehicle,
  ValidationResult,
  Document
};