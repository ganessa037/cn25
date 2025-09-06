import dotenv from "dotenv";
dotenv.config();

const config = {
  port: process.env.PORT || 3000,
  nodeEnv: process.env.NODE_ENV || "development",

  database: {
    host: process.env.DB_HOST || "localhost",
    port: Number(process.env.DB_PORT || 5432),
    name: process.env.DB_NAME || "vehicle_validation_db",
    username: process.env.DB_USER || "postgres",
    password: process.env.DB_PASSWORD || "",
    dialect: process.env.DB_DIALECT || "postgres",
    logging: process.env.NODE_ENV === "development" ? console.log : false,
  },

  jwt: {
    secret: process.env.JWT_SECRET || "fallback-secret-key",
    expiresIn: process.env.JWT_EXPIRES_IN || "24h",
    refreshExpiresIn: process.env.JWT_REFRESH_EXPIRES_IN || "7d",
  },

  mlService: {
    url: process.env.ML_SERVICE_URL || "http://localhost:8000",
  },

  upload: {
    maxSize: Number(process.env.UPLOAD_MAX_SIZE || 10 * 1024 * 1024), // 10MB
    allowedTypes:
      (process.env.UPLOAD_ALLOWED_TYPES &&
        process.env.UPLOAD_ALLOWED_TYPES.split(",")) || [
        "image/jpeg",
        "image/png",
        "image/jpg",
        "application/pdf",
      ],
  },
};

export default config;
export const { database, jwt, mlService, upload } = config;