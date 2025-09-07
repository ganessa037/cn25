import jwt from 'jsonwebtoken';
import models from '../models/index.js';       
import config from '../config/config.js';
import logger from '../utils/logger.js';

export async function authMiddleware(req, res, next) {
  try {
    const authHeader = req.header('Authorization');
    if (!authHeader?.startsWith('Bearer ')) {
      return res.status(401).json({ success: false, error: { code: 'NO_TOKEN', message: 'Access denied. No token provided.' } });
    }
    const token = authHeader.slice(7);
    const decoded = jwt.verify(token, config.jwt.secret);
    const user = await models.User.findByPk(decoded.userId, { attributes: { exclude: ['password'] } });

    if (!user)   return res.status(401).json({ success: false, error: { code: 'USER_NOT_FOUND', message: 'Token is valid but user not found.' } });
    if (!user.isActive) return res.status(401).json({ success: false, error: { code: 'ACCOUNT_INACTIVE', message: 'Account is inactive.' } });

    req.user = user;
    next();
  } catch (error) {
    logger.error('Auth middleware error:', error);
    if (error.name === 'JsonWebTokenError') return res.status(401).json({ success: false, error: { code: 'INVALID_TOKEN', message: 'Invalid token.' } });
    if (error.name === 'TokenExpiredError') return res.status(401).json({ success: false, error: { code: 'TOKEN_EXPIRED', message: 'Token has expired.' } });
    res.status(500).json({ success: false, error: { code: 'AUTH_ERROR', message: 'Authentication error.' } });
  }
}

export async function optionalAuth(req, _res, next) {
  try {
    const authHeader = req.header('Authorization');
    if (!authHeader?.startsWith('Bearer ')) return next();
    const token = authHeader.slice(7);
    const decoded = jwt.verify(token, config.jwt.secret);
    const user = await models.User.findByPk(decoded.userId, { attributes: { exclude: ['password'] } });
    if (user?.isActive) req.user = user;
    next();
  } catch {
    next();
  }
}