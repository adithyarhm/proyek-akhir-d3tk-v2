// ── SIMULATION STATE ──────────────────────────────────────────────────────────
const SCENARIO_META = {
  1:{name:'Baseline Per-Node',mode:'per_node',modelName:'RandomForest',features:['hum_pct','temp_c','wind_kph'],nFeatures:3,rmse_so2:28.4,rmse_h2s:18.2},
  2:{name:'Enhanced Per-Node',mode:'per_node',modelName:'XGBoost',features:['hum_pct','temp_c','wind_kph','hour','minute','so2_lag1','h2s_lag1'],nFeatures:13,rmse_so2:19.7,rmse_h2s:12.6},
  3:{name:'Global Model (Baseline)',mode:'global',modelName:'GradientBoosting',features:['hum_pct','temp_c','wind_kph'],nFeatures:6,rmse_so2:31.2,rmse_h2s:20.8},
  4:{name:'Global Model + Temporal',mode:'global',modelName:'CatBoost',features:['hum_pct','temp_c','wind_kph','hour','minute','so2_lag1','h2s_lag1'],nFeatures:16,rmse_so2:16.3,rmse_h2s:10.4},
};

const NODES = [
  {id:'N-01',lat:-7.1234,lon:107.4021,elev:2090,label:'Crater Rim A'},
  {id:'N-02',lat:-7.1241,lon:107.4035,elev:2085,label:'Crater Rim B'},
  {id:'N-03',lat:-7.1228,lon:107.4010,elev:2095,label:'West Slope'},
  {id:'N-04',lat:-7.1250,lon:107.4050,elev:2075,label:'East Lookout'},
];

const SO2_DANGER=500, SO2_WARN=250, H2S_DANGER=150, H2S_WARN=70;
const MAX_POINTS=30;

let currentScenario=1, simRunning=false, simInterval=null, simStep=0, simSpeed=1000;
let chartSO2, chartH2S, gaugeSO2, gaugeH2S;
let so2Pred=[], h2sPred=[], so2Truth=[], h2sTruth=[], timeLabels=[];
let prevSO2=0, prevH2S=0, prevTemp=0, prevHum=0, prevWind=0;
let activeNodeIdx=0;

// ── HELPERS ──────────────────────────────────────────────────────────────────
function rand(min,max){return Math.random()*(max-min)+min}
function randNorm(mean,std){let u=1-Math.random(),v=1-Math.random();return mean+std*Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)}
function clamp(v,lo,hi){return Math.max(lo,Math.min(hi,v))}
function fmt(v,d=1){return typeof v==='number'?v.toFixed(d):'--'}
function nowStr(){return new Date().toLocaleTimeString('id-ID',{hour:'2-digit',minute:'2-digit',second:'2-digit'})}

function simGasPattern(step){
  const t=step/MAX_POINTS;
  const base_so2=150+80*Math.sin(t*Math.PI*2)+30*Math.sin(t*Math.PI*7);
  const base_h2s=60+30*Math.sin(t*Math.PI*2+1)+10*Math.sin(t*Math.PI*5);
  const meta=SCENARIO_META[currentScenario];
  const noise_so2=randNorm(0,meta.rmse_so2*0.6);
  const noise_h2s=randNorm(0,meta.rmse_h2s*0.6);
  const spike=step>0&&step%17===0?rand(150,400):0;
  const so2_gt=clamp(base_so2+spike+rand(-20,20),5,900);
  const h2s_gt=clamp(base_h2s+spike*0.3+rand(-8,8),1,300);
  const so2_pr=clamp(so2_gt+noise_so2,0,950);
  const h2s_pr=clamp(h2s_gt+noise_h2s,0,320);
  const temp=clamp(randNorm(18,2),10,28);
  const hum=clamp(randNorm(82,6),50,100);
  const wind=clamp(randNorm(12,4),0,40);
  return{so2_gt,so2_pr,h2s_gt,h2s_pr,temp,hum,wind};
}

// ── INIT CHARTS ───────────────────────────────────────────────────────────────
function makeLineChart(id,label1,label2,c1,c2){
  const ctx=document.getElementById(id).getContext('2d');
  return new Chart(ctx,{
    type:'line',
    data:{
      labels:[],
      datasets:[
        {label:'Prediksi',data:[],borderColor:c1,backgroundColor:c1+'22',borderWidth:2,pointRadius:2,tension:0.4,fill:true},
        {label:'Ground Truth',data:[],borderColor:c1+'88',borderDash:[5,3],borderWidth:1.5,pointRadius:0,tension:0.4,fill:false}
      ]
    },
    options:{
      responsive:true,maintainAspectRatio:true,
      animation:{duration:400},
      plugins:{legend:{labels:{color:'#8892a8',font:{size:11,family:'Inter'},boxWidth:12}},tooltip:{bodyFont:{family:'JetBrains Mono',size:11}}},
      scales:{
        x:{ticks:{color:'#8892a8',font:{size:9,family:'JetBrains Mono'},maxTicksLimit:8},grid:{color:'rgba(255,255,255,0.04)'}},
        y:{ticks:{color:'#8892a8',font:{size:9,family:'JetBrains Mono'}},grid:{color:'rgba(255,255,255,0.04)'}}
      }
    }
  });
}

function makeGauge(id,max,color){
  const ctx=document.getElementById(id).getContext('2d');
  return new Chart(ctx,{
    type:'doughnut',
    data:{datasets:[{data:[0,max],backgroundColor:[color,getComputedStyle(document.documentElement).getPropertyValue('--color-surface-dynamic').trim()||'#252a38'],borderWidth:0,circumference:180,rotation:270}]},
    options:{responsive:false,cutout:'72%',animation:{duration:400},plugins:{legend:{display:false},tooltip:{enabled:false}}}
  });
}

// ── THEME ─────────────────────────────────────────────────────────────────────
let isDark=true;
function toggleTheme(){
  isDark=!isDark;
  document.documentElement.setAttribute('data-theme',isDark?'dark':'light');
  document.getElementById('themeToggle').innerHTML=isDark
    ?'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>'
    :'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>';
}

// ── SCENARIO ─────────────────────────────────────────────────────────────────
function switchScenario(sc){
  currentScenario=sc;
  document.querySelectorAll('.scenario-btn').forEach(b=>{
    b.classList.toggle('active',parseInt(b.dataset.sc)===sc);
  });
  updateModelInfo();
  addLog(`Beralih ke Skenario ${sc}: ${SCENARIO_META[sc].name}`,'ok');
}

function updateModelInfo(){
  const m=SCENARIO_META[currentScenario];
  document.getElementById('mName').textContent=m.modelName;
  document.getElementById('mScenario').textContent=`S${currentScenario} · ${m.name}`;
  document.getElementById('mMode').textContent=m.mode;
  document.getElementById('mFeatures').textContent=`${m.nFeatures} fitur`;
  document.getElementById('mRMSE_SO2').textContent=m.rmse_so2.toFixed(1)+' µg/m³';
  document.getElementById('mRMSE_H2S').textContent=m.rmse_h2s.toFixed(1)+' µg/m³';
  document.getElementById('kpiRMSE').textContent=m.rmse_so2.toFixed(1);
  document.getElementById('trendRMSE').textContent=`${m.modelName} · pemenang skenario ini`;
  document.getElementById('trendRMSE').className='kpi-trend neutral';
}

// ── NODES ─────────────────────────────────────────────────────────────────────
function renderNodes(){
  const el=document.getElementById('nodeList');
  el.innerHTML=NODES.map((n,i)=>`
    <div class="node-item ${i===activeNodeIdx?'active':''}" onclick="selectNode(${i})">
      <div>
        <div class="node-name">${n.id}</div>
        <div class="node-coords">${n.lat.toFixed(4)}, ${n.lon.toFixed(4)}</div>
      </div>
      <span class="node-status-dot" style="background:var(--color-success)"></span>
    </div>`).join('');
}
function selectNode(i){
  activeNodeIdx=i;
  renderNodes();
  addLog(`Node aktif: ${NODES[i].id} (${NODES[i].label})`,'ok');
}

// ── SIMULATION STEP ──────────────────────────────────────────────────────────
async function simTick(){
  const d=simGasPattern(simStep++);
  const ts=nowStr();

  // Hitung derived features dan temporal untuk ML
  const node = NODES[activeNodeIdx];
  const now = new Date();
  const hour = now.getHours();
  const minute = now.getMinutes();
  const minute_of_day = hour * 60 + minute;

  const prev_so2 = so2Truth.length > 0 ? so2Truth[so2Truth.length - 1] : d.so2_gt;
  const prev_prev_so2 = so2Truth.length > 1 ? so2Truth[so2Truth.length - 2] : prev_so2;
  const prev_h2s = h2sTruth.length > 0 ? h2sTruth[h2sTruth.length - 1] : d.h2s_gt;
  const prev_prev_h2s = h2sTruth.length > 1 ? h2sTruth[h2sTruth.length - 2] : prev_h2s;

  const so2_diff = d.so2_gt - prev_so2;
  const h2s_diff = d.h2s_gt - prev_h2s;
  const gas_ratio = d.so2_gt / (d.h2s_gt + 1e-6);

  const model_name = SCENARIO_META[currentScenario].modelName;

  let so2_pr = d.so2_pr;
  let h2s_pr = d.h2s_pr;
  let model_display = model_name;

  try {
    const url = `/api/predict?` + new URLSearchParams({
      scenario: currentScenario,
      model_name: model_name,
      node: node.id,
      temp_c: d.temp.toFixed(2),
      hum_pct: d.hum.toFixed(2),
      wind_kph: d.wind.toFixed(2),
      hour: hour,
      minute: minute,
      minute_of_day: minute_of_day,
      so2_diff: so2_diff.toFixed(2),
      h2s_diff: h2s_diff.toFixed(2),
      gas_ratio_so2_h2s: gas_ratio.toFixed(4),
      so2_ugm_lag1: prev_so2.toFixed(2),
      so2_ugm_lag2: prev_prev_so2.toFixed(2),
      h2s_ugm_lag1: prev_h2s.toFixed(2),
      h2s_ugm_lag2: prev_prev_h2s.toFixed(2),
      lat: node.lat,
      lon: node.lon,
      elev: node.elev
    });

    const res = await fetch(url);
    if (res.ok) {
      const apiData = await res.json();
      so2_pr = apiData.predictions.so2_ugm3;
      h2s_pr = apiData.predictions.h2s_ugm3;
      model_display = apiData.model;
      
      // Update model name label in dashboard header to indicate it's using the live model
      document.getElementById('mName').textContent = `${model_display} (LIVE)`;
    } else {
      document.getElementById('mName').textContent = `${model_name} (SIM)`;
    }
  } catch (err) {
    console.error("Prediction fetch failed, using fallback:", err);
    document.getElementById('mName').textContent = `${model_name} (SIM)`;
  }

  // Append chart data
  if(timeLabels.length>=MAX_POINTS){timeLabels.shift();so2Pred.shift();so2Truth.shift();h2sPred.shift();h2sTruth.shift()}
  timeLabels.push(ts);so2Pred.push(+so2_pr.toFixed(1));so2Truth.push(+d.so2_gt.toFixed(1));
  h2sPred.push(+h2s_pr.toFixed(1));h2sTruth.push(+d.h2s_gt.toFixed(1));

  // Update line charts
  chartSO2.data.labels=timeLabels;chartSO2.data.datasets[0].data=so2Pred;chartSO2.data.datasets[1].data=so2Truth;chartSO2.update('none');
  chartH2S.data.labels=timeLabels;chartH2S.data.datasets[0].data=h2sPred;chartH2S.data.datasets[1].data=h2sTruth;chartH2S.update('none');

  // Update gauges
  const so2Max=900,h2sMax=300;
  gaugeSO2.data.datasets[0].data=[so2_pr,so2Max-so2_pr];gaugeSO2.update('none');
  gaugeH2S.data.datasets[0].data=[h2s_pr,h2sMax-h2s_pr];gaugeH2S.update('none');
  document.getElementById('gaugeValSO2').textContent=fmt(so2_pr);
  document.getElementById('gaugeValH2S').textContent=fmt(h2s_pr);

  // KPI
  const so2Trend=so2_pr-prevSO2;const h2sTrend=h2s_pr-prevH2S;
  setKPI('kpiSO2',fmt(so2_pr),'trendSO2',so2Trend,'µg/m³');
  setKPI('kpiH2S',fmt(h2s_pr),'trendH2S',h2sTrend,'µg/m³');
  setKPI('kpiTemp',fmt(d.temp),'trendTemp',d.temp-prevTemp,'°C');
  setKPI('kpiHum',fmt(d.hum,0),'trendHum',d.hum-prevHum,'%');
  setKPI('kpiWind',fmt(d.wind),'trendWind',d.wind-prevWind,'km/h');
  prevSO2=so2_pr;prevH2S=h2s_pr;prevTemp=d.temp;prevHum=d.hum;prevWind=d.wind;

  // Weather bars
  updateWeatherBars(d.temp,d.hum,d.wind);

  // Status & alert
  const so2Status=so2_pr>=SO2_DANGER?'danger':so2_pr>=SO2_WARN?'warning':'safe';
  const h2sStatus=h2s_pr>=H2S_DANGER?'danger':h2s_pr>=H2S_WARN?'warning':'safe';
  const overallStatus=['safe','warning','danger'].indexOf(so2Status)>['safe','warning','danger'].indexOf(h2sStatus)?so2Status:h2sStatus;
  updateStatus(overallStatus);

  // EWS badge
  setBadge('gaugeBadgeSO2',so2Status,so2_pr>=SO2_DANGER?'BAHAYA':so2_pr>=SO2_WARN?'WASPADA':'AMAN');
  setBadge('gaugeBadgeH2S',h2sStatus,h2s_pr>=H2S_DANGER?'BAHAYA':h2s_pr>=H2S_WARN?'WASPADA':'AMAN');

  // Anomaly banner
  const banner=document.getElementById('anomalyBanner');
  if(overallStatus!=='safe'){
    const gas=so2Status!=='safe'?'SO₂':'H₂S';
    const val=so2Status!=='safe'?fmt(so2_pr):fmt(h2s_pr);
    document.getElementById('anomalyText').innerHTML=`⚠ ${gas} terdeteksi <span>${val} µg/m³</span> — melampaui ambang batas ${overallStatus==='danger'?'BAHAYA':'WASPADA'}`;
    banner.className='anomaly-banner show';
  } else {
    banner.className='anomaly-banner';
  }

  // Prediction table
  const tbodyEl=document.getElementById('predTableBody');
  const statusLabel=overallStatus==='danger'?'bahaya':overallStatus==='warning'?'waspada':'aman';
  const newRow=`<tr><td>${ts}</td><td>${node.id}</td><td>${fmt(so2_pr)}</td><td>${fmt(h2s_pr)}</td><td><span class="badge ${statusLabel}">${statusLabel.toUpperCase()}</span></td></tr>`;
  tbodyEl.innerHTML=newRow+tbodyEl.innerHTML;
  if(tbodyEl.rows.length>10)tbodyEl.deleteRow(tbodyEl.rows.length-1);

  // Log occasional messages
  if(simStep%5===0||overallStatus!=='safe'){
    const lvl=overallStatus==='danger'?'err':overallStatus==='warning'?'warn':'ok';
    addLog(`[${node.id}] SO₂=${fmt(so2_pr)} H₂S=${fmt(h2s_pr)} ${overallStatus.toUpperCase()}`,lvl);
  }
}

function setKPI(valId,val,trendId,delta,unit){
  document.getElementById(valId).textContent=val;
  const tel=document.getElementById(trendId);
  const dir=delta>0.1?'▲':delta<-0.1?'▼':'→';
  const cls=delta>0.1?'up':delta<-0.1?'down':'neutral';
  tel.textContent=`${dir} ${Math.abs(delta).toFixed(1)} ${unit}`;
  tel.className=`kpi-trend ${cls}`;
}

function setBadge(id,status,label){
  const el=document.getElementById(id);
  el.textContent=label;
  el.className='gauge-status '+status;
  el.style.background=status==='danger'?'var(--color-error-dim)':status==='warning'?'var(--color-warning-dim)':'var(--color-success-dim)';
  el.style.color=status==='danger'?'var(--color-error)':status==='warning'?'var(--color-warning)':'var(--color-success)';
}

function updateStatus(status){
  const pill=document.getElementById('topStatusPill');
  const txt=document.getElementById('topStatusText');
  pill.className='status-pill '+status;
  txt.textContent=status==='danger'?'BAHAYA':status==='warning'?'WASPADA':'AMAN';
}

function updateWeatherBars(temp,hum,wind){
  const items=[
    {name:'Suhu (temp_c)',val:temp,max:40,unit:'°C',color:'var(--color-temp)'},
    {name:'Kelembapan (hum_pct)',val:hum,max:100,unit:'%',color:'var(--color-hum)'},
    {name:'Angin (wind_kph)',val:wind,max:60,unit:'km/h',color:'var(--color-wind)'},
  ];
  const el=document.getElementById('weatherMetrics');
  el.innerHTML=items.map(it=>`
    <div class="metric-item">
      <div class="metric-head">
        <span class="metric-name">${it.name}</span>
        <span class="metric-val" style="color:${it.color}">${it.val.toFixed(1)} ${it.unit}</span>
      </div>
      <div class="metric-bar">
        <div class="metric-fill" style="width:${(it.val/it.max*100).toFixed(1)}%;--bar-color:${it.color}"></div>
      </div>
    </div>`).join('');
}

// ── LOG ───────────────────────────────────────────────────────────────────────
let logCount=0;
function addLog(msg,lvl=''){
  const el=document.getElementById('sysLog');
  const entry=document.createElement('div');entry.className='log-entry';
  entry.innerHTML=`<span class="log-time">${nowStr()}</span><span class="log-msg ${lvl}">${msg}</span>`;
  el.prepend(entry);
  if(el.children.length>40)el.removeChild(el.lastChild);
}

// ── SIM CONTROLS ─────────────────────────────────────────────────────────────
function toggleSim(){
  simRunning=!simRunning;
  const btn=document.getElementById('btnStart');
  if(simRunning){
    simInterval=setInterval(simTick,simSpeed);
    btn.textContent='⏸ Pause Simulasi';
    addLog('Simulasi dimulai — model aktif: '+SCENARIO_META[currentScenario].modelName,'ok');
  }else{
    clearInterval(simInterval);
    btn.textContent='▶ Lanjut Simulasi';
    addLog('Simulasi dijeda','warn');
  }
}
function resetSim(){
  clearInterval(simInterval);simRunning=false;simStep=0;
  so2Pred=[];h2sPred=[];so2Truth=[];h2sTruth=[];timeLabels=[];
  chartSO2.data.labels=[];chartSO2.data.datasets[0].data=[];chartSO2.data.datasets[1].data=[];chartSO2.update();
  chartH2S.data.labels=[];chartH2S.data.datasets[0].data=[];chartH2S.data.datasets[1].data=[];chartH2S.update();
  gaugeSO2.data.datasets[0].data=[0,900];gaugeSO2.update();
  gaugeH2S.data.datasets[0].data=[0,300];gaugeH2S.update();
  document.getElementById('predTableBody').innerHTML='';
  document.getElementById('btnStart').textContent='▶ Mulai Simulasi';
  document.getElementById('anomalyBanner').className='anomaly-banner';
  updateStatus('safe');
  addLog('Simulasi di-reset','ok');
  ['kpiSO2','kpiH2S','kpiTemp','kpiHum','kpiWind'].forEach(id=>document.getElementById(id).textContent='--');
  updateWeatherBars(18,82,12);
}
function changeSpeed(){
  simSpeed=parseInt(document.getElementById('speedSel').value);
  if(simRunning){clearInterval(simInterval);simInterval=setInterval(simTick,simSpeed);}
}

// ── CLOCK ─────────────────────────────────────────────────────────────────────
function updateClock(){document.getElementById('clockDisplay').textContent=new Date().toLocaleTimeString('id-ID')}
setInterval(updateClock,1000);updateClock();

// ── BOOT ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded',()=>{
  renderNodes();
  updateModelInfo();
  updateWeatherBars(18,82,12);

  chartSO2=makeLineChart('chartSO2','Prediksi','GT','#38bdf8','#a78bfa');
  chartH2S=makeLineChart('chartH2S','Prediksi','GT','#a78bfa','#38bdf8');
  gaugeSO2=makeGauge('gaugeSO2',900,'#38bdf8');
  gaugeH2S=makeGauge('gaugeH2S',300,'#a78bfa');

  addLog('EWS Dashboard aktif — menunggu simulasi dimulai','ok');
  addLog('Model tersedia: RF, XGBoost, GradientBoosting, CatBoost');
  addLog('Ambang SO₂: WASPADA=250, BAHAYA=500 µg/m³','warn');
  addLog('Ambang H₂S: WASPADA=70,  BAHAYA=150 µg/m³','warn');
});