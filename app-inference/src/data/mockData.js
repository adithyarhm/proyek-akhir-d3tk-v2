// Simulasi data — nanti ganti dengan output model joblib sesungguhnya

const SCENARIO_META = {
  1: { name:'Baseline Per-Node',  mode:'per_node', modelName:'RandomForest',   rmse_so2:28.4, rmse_h2s:18.2 },
  2: { name:'Enhanced Per-Node',  mode:'per_node', modelName:'XGBoost',        rmse_so2:22.1, rmse_h2s:14.8 },
  3: { name:'Baseline Global',    mode:'global',   modelName:'GradientBoost',  rmse_so2:31.2, rmse_h2s:21.5 },
  4: { name:'Enhanced Global',    mode:'global',   modelName:'XGBoost',        rmse_so2:19.8, rmse_h2s:12.3 }
};

const NODES = [
  { id:'N-01', name:'Node 01', location:'Kawah Utama',  lat:-7.1647, lon:107.3892 },
  { id:'N-02', name:'Node 02', location:'Tepi Barat',   lat:-7.1658, lon:107.3871 },
  { id:'N-03', name:'Node 03', location:'Jalur Wisata', lat:-7.1635, lon:107.3905 },
  { id:'N-04', name:'Node 04', location:'Pos Pantau',   lat:-7.1670, lon:107.3880 }
];

const THRESHOLDS = {
  so2: { warning: 250, danger: 500 },
  h2s: { warning: 70,  danger: 150 }
};

function generatePrediction(scenario, nodeId) {
  const meta    = SCENARIO_META[scenario] || SCENARIO_META[4];
  const t       = Date.now();
  const phase   = (t / 8000) % (2 * Math.PI);

  const temp_c  = 18 + 4 * Math.sin(phase * 0.7) + (Math.random() - 0.5) * 2;
  const hum_pct = 72 + 10 * Math.cos(phase * 0.5) + (Math.random() - 0.5) * 5;
  const wind_kph= 8  + 4 * Math.sin(phase * 1.2) + (Math.random() - 0.5) * 3;

  const so2_base = 180 + 120 * Math.abs(Math.sin(phase * 0.3));
  const h2s_base = 45  + 60  * Math.abs(Math.sin(phase * 0.4));

  const so2 = Math.max(0, so2_base + (Math.random() - 0.5) * meta.rmse_so2 * 2);
  const h2s = Math.max(0, h2s_base + (Math.random() - 0.5) * meta.rmse_h2s * 2);

  const alertLevel =
    so2 >= THRESHOLDS.so2.danger || h2s >= THRESHOLDS.h2s.danger ? 'BAHAYA' :
    so2 >= THRESHOLDS.so2.warning|| h2s >= THRESHOLDS.h2s.warning ? 'WASPADA' : 'AMAN';

  return {
    timestamp   : new Date().toISOString(),
    scenario,
    node        : nodeId,
    model       : meta.modelName,
    mode        : meta.mode,
    features    : { temp_c: +temp_c.toFixed(1), hum_pct: +hum_pct.toFixed(1), wind_kph: +wind_kph.toFixed(1) },
    predictions : { so2_ugm3: +so2.toFixed(2), h2s_ugm3: +h2s.toFixed(2) },
    alert_level : alertLevel,
    rmse        : { so2: meta.rmse_so2, h2s: meta.rmse_h2s }
  };
}

// Buat array histori dummy
function getHistory(scenario, nodeId, limit = 30) {
  return Array.from({ length: limit }, (_, i) => {
    const offset = (limit - i) * 2000;
    return generatePrediction(scenario, nodeId);
  });
}

module.exports = { SCENARIO_META, NODES, THRESHOLDS, generatePrediction, getHistory };