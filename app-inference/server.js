require('dotenv').config();
const express = require('express');
const cors    = require('cors');
const path    = require('path');
const apiRouter = require('./src/routes/api');

const app  = express();
const PORT = process.env.PORT || 3000;

// ── Middleware ──────────────────────────────────────────────
app.use(cors());
app.use(express.json());

// ── Static files (HTML, CSS, JS frontend) ──────────────────
app.use(express.static(path.join(__dirname, 'public')));

// ── API Routes ──────────────────────────────────────────────
app.use('/api', apiRouter);

// ── Fallback → selalu serve index.html (SPA-ready) ─────────
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// ── Start Server ────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n🟢 EWS Dashboard berjalan di http://localhost:${PORT}`);
  console.log(`   Mode    : ${process.env.NODE_ENV || 'development'}`);
  console.log(`   API     : http://localhost:${PORT}/api\n`);
});