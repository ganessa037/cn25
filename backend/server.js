import app from './app.js';
import config from './src/config/config.js';
import logger from './src/utils/logger.js';

const PORT = config.port || process.env.PORT || 3000;
const HOST = config.host || process.env.HOST || '127.0.0.1';

app.listen(PORT, HOST, () => {
  logger.info(`API listening on http://${HOST}:${PORT}`);
});