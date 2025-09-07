import { Sequelize } from "sequelize";
import config from "./config.js";

// Prefer DATABASE_URL if provided; fall back to discrete fields.
const url =
  process.env.DATABASE_URL ||
  `postgresql://${encodeURIComponent(config.database.username)}:${encodeURIComponent(
    config.database.password || ""
  )}@${config.database.host}:${config.database.port}/${config.database.name}`;

// Enable SSL automatically for hosted DBs; disable for localhost
const isLocal =
  url.includes("localhost") || url.includes("127.0.0.1") || url.includes("::1");

export const sequelize = new Sequelize(url, {
  dialect: "postgres",
  logging: false,
  dialectOptions: isLocal
    ? {}
    : {
        ssl: { require: true, rejectUnauthorized: false },
      },
});

