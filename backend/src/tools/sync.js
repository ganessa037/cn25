import "dotenv/config";
import { sequelize, User, Vehicle, ValidationResult, Document } from "../models/index.js";

async function main() {
  try {
    console.log("[sync] connecting…");
    await sequelize.authenticate();
    console.log("[sync] starting…");
    await sequelize.sync({ alter: true }); // creates tables / aligns columns
    console.log("[sync] done");
    process.exit(0);
  } catch (err) {
    console.error("[sync] error:", err);
    process.exit(1);
  }
}
await main();
