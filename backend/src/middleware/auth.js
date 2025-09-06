const jwt = require("jsonwebtoken");
const models = require("../models");
const config = require("../config/config");
const logger = require("../utils/logger");

// 等价于你原来的 ESM 版本：验证 Bearer token、查用户、挂到 req.user
const authMiddleware = async (req, res, next) => {
  try {
    const authHeader = req.header("Authorization");
    if (!authHeader?.startsWith("Bearer ")) {
      return res.status(401).json({
        success: false,
        error: { code: "NO_TOKEN", message: "Access denied. No token provided." },
      });
    }
    const token = authHeader.slice(7);
    const decoded = jwt.verify(token, config.jwt.secret);
    const user = await models.User.findByPk(decoded.userId, {
      attributes: { exclude: ["password"] },
    });

    if (!user) {
      return res.status(401).json({
        success: false,
        error: { code: "USER_NOT_FOUND", message: "Token is valid but user not found." },
      });
    }
    if (!user.isActive) {
      return res.status(401).json({
        success: false,
        error: { code: "ACCOUNT_INACTIVE", message: "Account is inactive." },
      });
    }

    req.user = user;
    next();
  } catch (error) {
    logger.error("Auth middleware error:", error);
    if (error.name === "JsonWebTokenError") {
      return res.status(401).json({ success: false, error: { code: "INVALID_TOKEN", message: "Invalid token." } });
    }
    if (error.name === "TokenExpiredError") {
      return res.status(401).json({ success: false, error: { code: "TOKEN_EXPIRED", message: "Token has expired." } });
    }
    res.status(500).json({ success: false, error: { code: "AUTH_ERROR", message: "Authentication error." } });
  }
};

const optionalAuth = async (req, _res, next) => {
  try {
    const authHeader = req.header("Authorization");
    if (!authHeader?.startsWith("Bearer ")) return next();
    const token = authHeader.slice(7);
    const decoded = jwt.verify(token, config.jwt.secret);
    const user = await models.User.findByPk(decoded.userId, { attributes: { exclude: ["password"] } });
    if (user?.isActive) req.user = user;
    next();
  } catch {
    next();
  }
};

module.exports = { authMiddleware, optionalAuth };