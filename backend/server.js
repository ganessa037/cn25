import app from './app.js';
import 'dotenv/config';

const PORT = process.env.PORT || 3000;
app.listen(PORT, '127.0.0.1', () => {
  console.log(`API listening on http://127.0.0.1:${PORT}`);
});