import { Sequelize } from "sequelize";
import config from "./config.js";

// Prefer DATABASE_URL if provided; fall back to discrete fields.
const url = process.env.DATABASE_URL;

export const sequelize = url
  ? new Sequelize(url, { logging: false, dialect: "postgres" })
  : new Sequelize(
      config.database.name,
      config.database.username,
      config.database.password,
      {
        host: config.database.host,
        port: config.database.port,
        dialect: config.database.dialect,
        logging: config.database.logging,
        pool: { max: 10, min: 0, acquire: 30000, idle: 10000 },
        timezone: "+08:00",
        define: { timestamps: true, underscored: true, freezeTableName: true },
      }
    );

try { await sequelize.authenticate(); } catch {}

