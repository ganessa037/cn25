import { DataTypes } from "sequelize";
import { sequelize } from "../config/database.js";

const ValidationResult = sequelize.define(
  "ValidationResult",
  {
    id: { type: DataTypes.UUID, defaultValue: DataTypes.UUIDV4, primaryKey: true },
    validationId: { type: DataTypes.STRING, unique: true, allowNull: false },
    vehicleId: { type: DataTypes.UUID, references: { model: "vehicles", key: "id" } },
    userId: { type: DataTypes.UUID, allowNull: false, references: { model: "users", key: "id" } },
    validationData: { type: DataTypes.JSONB, allowNull: false },
    confidenceScore: { type: DataTypes.INTEGER, validate: { min: 0, max: 100 } },
    errorsDetected: { type: DataTypes.JSONB },
    correctionsSuggested: { type: DataTypes.JSONB },
    ocrResults: { type: DataTypes.JSONB },
    validationLevel: { type: DataTypes.ENUM("basic", "standard", "comprehensive"), defaultValue: "standard" },
    status: { type: DataTypes.ENUM("processing", "completed", "failed"), defaultValue: "processing" },
    processingTimeMs: { type: DataTypes.INTEGER },
  },
  {
    tableName: "validation_results",
    indexes: [{ fields: ["userId"] }, { fields: ["vehicleId"] }, { unique: true, fields: ["validationId"] }],
  }
);

export default ValidationResult;