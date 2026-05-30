const express    = require('express');
const router     = express.Router();
const mockData   = require('../data/mockData');

// GET /api/status — health check
router.get('/status', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    version: '1.0.0'
  });
});

// GET /api/predict?scenario=4&node=N-01
// Endpoint prediksi model (memanggil FastAPI predict service, fallback ke mock)
router.get('/predict', async (req, res) => {
  const scenario = parseInt(req.query.scenario) || 4;
  const node_id  = req.query.node || 'N-01';
  const model_name = req.query.model_name || 'RandomForest';

  try {
    const payload = {
      scenario,
      model_name,
      node_id,
      temp_c: parseFloat(req.query.temp_c) || 0.0,
      hum_pct: parseFloat(req.query.hum_pct) || 0.0,
      wind_kph: parseFloat(req.query.wind_kph) || 0.0,
      hour: parseInt(req.query.hour) || 0,
      minute: parseInt(req.query.minute) || 0,
      minute_of_day: parseInt(req.query.minute_of_day) || 0,
      h2s_diff: parseFloat(req.query.h2s_diff) || 0.0,
      so2_diff: parseFloat(req.query.so2_diff) || 0.0,
      gas_ratio_so2_h2s: parseFloat(req.query.gas_ratio_so2_h2s) || 0.0,
      so2_ugm_lag1: parseFloat(req.query.so2_ugm_lag1) || 0.0,
      so2_ugm_lag2: parseFloat(req.query.so2_ugm_lag2) || 0.0,
      h2s_ugm_lag1: parseFloat(req.query.h2s_ugm_lag1) || 0.0,
      h2s_ugm_lag2: parseFloat(req.query.h2s_ugm_lag2) || 0.0,
      lat: parseFloat(req.query.lat) || 0.0,
      lon: parseFloat(req.query.lon) || 0.0,
      elev: parseFloat(req.query.elev) || 0.0
    };

    // Panggil FastAPI backend
    const response = await fetch('http://127.0.0.1:8000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`FastAPI error: ${errorText}`);
    }

    const data = await response.json();
    const meta = mockData.SCENARIO_META[scenario] || mockData.SCENARIO_META[4];

    res.json({
      timestamp: new Date().toISOString(),
      scenario,
      node: node_id,
      model: model_name,
      mode: meta.mode,
      features: { temp_c: payload.temp_c, hum_pct: payload.hum_pct, wind_kph: payload.wind_kph },
      predictions: {
        so2_ugm3: data.so2_pred,
        h2s_ugm3: data.h2s_pred
      },
      alert_level: (data.so2_pred >= mockData.THRESHOLDS.so2.danger || data.h2s_pred >= mockData.THRESHOLDS.h2s.danger) ? 'BAHAYA' :
                   (data.so2_pred >= mockData.THRESHOLDS.so2.warning || data.h2s_pred >= mockData.THRESHOLDS.h2s.warning) ? 'WASPADA' : 'AMAN',
      rmse: { so2: meta.rmse_so2, h2s: meta.rmse_h2s },
      model_used: data.model_used
    });

  } catch (error) {
    console.warn(`[Proxy Warning] FastAPI offline/error: ${error.message}. Fallback ke data simulasi.`);
    // Fallback ke data simulasi
    const result = mockData.generatePrediction(scenario, node_id);
    // Masukkan info fallback agar di dashboard kelihatan
    result.model = `${model_name} (S)`;
    res.json(result);
  }
});

// GET /api/history?scenario=4&node=N-01&limit=30
router.get('/history', (req, res) => {
  const scenario = parseInt(req.query.scenario) || 4;
  const node     = req.query.node || 'N-01';
  const limit    = parseInt(req.query.limit) || 30;

  const history = mockData.getHistory(scenario, node, limit);
  res.json(history);
});

// GET /api/scenarios — daftar semua skenario
router.get('/scenarios', (req, res) => {
  res.json(mockData.SCENARIO_META);
});

// GET /api/nodes — daftar node sensor
router.get('/nodes', (req, res) => {
  res.json(mockData.NODES);
});

module.exports = router;