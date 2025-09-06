import logger from '../utils/logger.js';

function normalize(error) {
  let status = error.statusCode || 500;
  let message = error.message || 'Internal Server Error';
  let code = error.code || 'INTERNAL_ERROR';

  // Common cases (extend as you need)
  switch (error.name) {
    case 'ValidationError':
    case 'SequelizeValidationError':
      status = 400; message = 'Validation failed'; code = 'VALIDATION_ERROR'; break;
    case 'SequelizeUniqueConstraintError':
      status = 409; message = 'Duplicate resource'; code = 'DUPLICATE'; break;
    case 'UnauthorizedError':
    case 'JsonWebTokenError':
      status = 401; message = 'Unauthorized access'; code = 'UNAUTHORIZED'; break;
    case 'TokenExpiredError':
      status = 401; message = 'Token expired'; code = 'TOKEN_EXPIRED'; break;
  }
  if (error.code === 'ECONNREFUSED') {
    status = 503; message = 'External service unavailable'; code = 'UPSTREAM_UNAVAILABLE';
  }
  return { status, message, code };
}

export default function errorHandler(err, req, res, _next) {
  logger.error('Error occurred', { message: err.message, stack: err.stack });
  const status = err.statusCode || 500;
  res.status(status).json({
    success: false,
    error: { message: err.message || 'Internal Server Error' }
  });
}

  const { status, message, code } = normalize(error);
  const isDevelopment = process.env.NODE_ENV !== 'production';

  res.status(status).json({
    success: false,
    error: {
      code,
      message,
      ...(isDevelopment && { stack: error.stack }),
      timestamp: new Date().toISOString(),
    },
  });
