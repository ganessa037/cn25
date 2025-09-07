import { DataTypes } from "sequelize";
import { sequelize } from "../config/database.js";

const User = sequelize.define(
  "User",
  {
    id: { type: DataTypes.UUID, defaultValue: DataTypes.UUIDV4, primaryKey: true },
    email: {
      type: DataTypes.STRING,
      allowNull: false,
      unique: { name: "unique_email", msg: "Email already in use" },
      validate: { isEmail: { msg: "Please provide a valid email address" } },
    },
    name: { type: DataTypes.STRING },
    avatarUrl: { type: DataTypes.STRING },
    // createdAt / updatedAt are handled by Sequelize when timestamps: true and underscored: false
  },
  {
    tableName: "User",     // Prisma created table "User" (case-sensitive)
    timestamps: true,
    underscored: false,    // Prisma uses createdAt / updatedAt (camelCase)
    freezeTableName: true, // do not pluralize ("User" exactly)
  }
);

export default User;