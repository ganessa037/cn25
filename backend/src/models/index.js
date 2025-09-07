import { sequelize } from "../config/database.js";         // must export { sequelize }
import User from "./User.js";
import Vehicle from "./Vehicle.js";
import ValidationResult from "./ValidationResult.js";
import Document from "./Document.js";

// --- Associations ---
User.hasMany(Vehicle, { foreignKey: "userId", as: "vehicles" });
Vehicle.belongsTo(User, { foreignKey: "userId", as: "user" });

User.hasMany(ValidationResult, { foreignKey: "userId", as: "validationResults" });
ValidationResult.belongsTo(User, { foreignKey: "userId", as: "user" });

Vehicle.hasMany(ValidationResult, { foreignKey: "vehicleId", as: "validationResults" });
ValidationResult.belongsTo(Vehicle, { foreignKey: "vehicleId", as: "vehicle" });

User.hasMany(Document, { foreignKey: "userId", as: "documents" });
Document.belongsTo(User, { foreignKey: "userId", as: "user" });

Vehicle.hasMany(Document, { foreignKey: "vehicleId", as: "documents" });
Document.belongsTo(Vehicle, { foreignKey: "vehicleId", as: "vehicle" });

// --- Exports (named) ---
export { sequelize, User, Vehicle, ValidationResult, Document };