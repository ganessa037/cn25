import "dotenv/config";

const config = {
  // App
  nodeEnv: process.env.NODE_ENV || "development",
  port: Number(process.env.PORT || 3000),
  host: process.env.HOST || "127.0.0.1",

  postLoginPath: process.env.FRONTEND_POST_LOGIN_PATH || "/dashboard",

  // Frontend origin used for redirects & CORS
  frontendUrl: process.env.FRONTEND_URL || "http://localhost:5173",

  // JWT
  jwt: {
    secret: process.env.JWT_SECRET || "dev_change_me_now",
  },

  // Google OAuth (must match Google Cloud Console)
  google: {
    clientID: process.env.GOOGLE_CLIENT_ID || "",
    clientSecret: process.env.GOOGLE_CLIENT_SECRET || "",
    callbackURL:
      process.env.GOOGLE_CALLBACK_URL ||
      "http://localhost:3000/api/auth/google/callback",
  },

  // Database (used by sequelize bootstrap)
  database: {
    host: process.env.DB_HOST || "localhost",
    port: Number(process.env.DB_PORT || 5432),
    name: process.env.DB_NAME || "cn25",
    username: process.env.DB_USER || "postgres",
    password: process.env.DB_PASSWORD || "postgres",
    dialect: process.env.DB_DIALECT || "postgres",
    logging: process.env.NODE_ENV === "development" ? false : false,
  },

  // Uploads
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
// Optional named exports if some modules import destructured values
export const { database, jwt, google, upload, frontendUrl } = config;