// backend/server.js
const app = require('./src/app');
const config = require('./src/config/config');
const logger = require('./src/utils/logger');

const PORT = config.port || 3001;

app.listen(PORT, () => {
  logger.info(`Backend server running on port ${PORT}`);
  logger.info(`API Documentation: http://localhost:${PORT}/api-docs`);
});