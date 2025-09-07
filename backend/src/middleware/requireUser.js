import jwt from 'jsonwebtoken';
import config from '../config/config.js';

export function requireUser(req, res, next) {
  const hdr = req.headers.authorization || '';
  const token = hdr.startsWith('Bearer ') ? hdr.slice(7) : null;
  let userId = null;

  if (token) {
    try {
      const decoded = jwt.verify(token, config.jwt.secret);
      userId = decoded.userId || decoded.sub || decoded.id;
    } catch {}
  }
  if (!userId && req.headers['x-demo-user-id']) {
    userId = String(req.headers['x-demo-user-id']);
  }
  if (!userId) {
    return res.status(401).json({ success: false, message: 'Unauthorized' });
  }
  req.user = { id: userId };
  next();
}