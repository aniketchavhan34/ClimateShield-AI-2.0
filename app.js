/* ClimateShield AI 2.0 — dashboard logic (Step 5)

   What this file does
     1. Calls the Flask API with fetch():  /api/health, /api/risk-map, /api/predict,
        /api/vulnerabilities, /api/simulate
     2. Draws the Leaflet map, the gauge, the Chart.js bar chart and every text panel
     3. Handles loading states, errors and empty data

   NOTHING is hard-coded as a result. Every score, level, importance value and scenario
   estimate shown on the page comes from the API response.

   DATA STATUS: PROTOTYPE / DEMO - the model was trained on synthetic flood labels.
*/
(function (root) {
  'use strict';

  /* =====================================================================
     1. CONSTANTS
     ===================================================================== */
  var API = {
    health: '/api/health',
    riskMap: '/api/risk-map',
    predict: '/api/predict',
    vulnerabilities: '/api/vulnerabilities',
    simulate: '/api/simulate',
    transparency: '/api/transparency'
  };
  var REQUEST_TIMEOUT_MS = 20000;

  var FEATURE_LABELS = {
    rainfall_mm: 'Rainfall (24 h)',
    elevation_m: 'Elevation',
    elevation_relative_m: 'Low-spot depth (vs neighbours)',
    drainage_condition: 'Drainage condition',
    impervious_surface_pct: 'Impervious surface'
  };
  var DRAINAGE_NAMES = ['very poor', 'poor', 'moderate', 'good', 'very good'];
  var LEVEL_COLORS = { LOW: '#22c55e', MEDIUM: '#facc15', HIGH: '#ef4444' };
  var LEVELS = ['LOW', 'MEDIUM', 'HIGH'];
  var FALLBACK_SCENARIO_LABEL = 'Hypothetical Scenario Estimate \u2014 Not a Guaranteed Real-World Prediction.';

  // Simple, visible rules used ONLY to describe a cell in the prototype explanation.
  // They are not part of the model.
  var RULES = {
    drainagePoorMax: 1,        // drainage level 0-1 = poor or very poor
    drainageGoodMin: 3,        // drainage level 3-4 = good or very good
    imperviousMargin: 5,       // percentage points above/below the area average
    lowSpotMeters: -1.0,       // at least 1 m lower than neighbours
    highSpotMeters: 1.0        // at least 1 m higher than neighbours
  };

  /* =====================================================================
     2. PURE HELPER FUNCTIONS (no page access - these are unit-tested in Node)
     ===================================================================== */
  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, function (ch) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch];
    });
  }

  function isNum(value) {
    return value !== null && value !== undefined && value !== '' && isFinite(Number(value));
  }

  function fmt(value, digits) {
    if (!isNum(value)) return '\u2014';
    return Number(value).toFixed(digits === undefined ? 1 : digits);
  }

  function fmtSigned(value, digits) {
    if (!isNum(value)) return '\u2014';
    var n = Number(value);
    var text = Math.abs(n).toFixed(digits === undefined ? 1 : digits);
    if (n > 0) return '+' + text;
    if (n < 0) return '\u2212' + text;      // proper minus sign
    return text;
  }

  function round1(value) { return Math.round(value * 10) / 10; }

  function levelKey(level) {
    var l = String(level || '').toUpperCase();
    return l === 'LOW' ? 'low' : l === 'MEDIUM' ? 'medium' : l === 'HIGH' ? 'high' : 'na';
  }

  function drainageText(value) {
    var v = Number(value);
    var name = DRAINAGE_NAMES[v] || 'unknown';
    return name + ' (' + v + ' of 4)';
  }

  // Rectangle corners of a grid cell from its centre point and the cell size (degrees)
  function cellBounds(cell, sizeDeg) {
    var half = sizeDeg / 2;
    return [[cell.latitude - half, cell.longitude - half], [cell.latitude + half, cell.longitude + half]];
  }

  function computeAreaStats(cells) {
    var total = 0;
    cells.forEach(function (c) { total += Number(c.impervious_surface_pct); });
    return {
      meanImpervious: cells.length ? total / cells.length : null,
      elevations: cells.map(function (c) { return Number(c.elevation_m); }),
      count: cells.length
    };
  }

  function highestCells(cells, n) {
    return cells.slice().sort(function (a, b) {
      return (b.risk_score - a.risk_score) || (a.grid_id < b.grid_id ? -1 : 1);
    }).slice(0, n);
  }

  // [{key, label, value}] sorted from most to least important (values come from the model)
  function topFactors(importance) {
    if (!importance) return [];
    return Object.keys(importance).map(function (key) {
      return { key: key, label: FEATURE_LABELS[key] || key, value: Number(importance[key]) };
    }).sort(function (a, b) { return b.value - a.value; });
  }

  // Compare ONE cell with the area using the visible RULES above.
  function conditionsForCell(cell, stats) {
    var adverse = [], favourable = [];
    var d = Number(cell.drainage_condition);
    if (d <= RULES.drainagePoorMax) adverse.push('Drainage condition is ' + drainageText(d));
    else if (d >= RULES.drainageGoodMin) favourable.push('Drainage condition is ' + drainageText(d));

    var imp = Number(cell.impervious_surface_pct);
    if (stats && stats.meanImpervious !== null) {
      var mean = stats.meanImpervious;
      if (imp > mean + RULES.imperviousMargin) {
        adverse.push('Impervious (paved/built) surface is ' + fmt(imp) + '%, above the area average of ' + fmt(mean) + '%');
      } else if (imp < mean - RULES.imperviousMargin) {
        favourable.push('Impervious (paved/built) surface is ' + fmt(imp) + '%, below the area average of ' + fmt(mean) + '%');
      }
    }

    var rel = Number(cell.elevation_relative_m);
    if (rel <= RULES.lowSpotMeters) adverse.push('The cell sits about ' + fmt(Math.abs(rel)) + ' m lower than its neighbouring cells (a local low spot)');
    else if (rel >= RULES.highSpotMeters) favourable.push('The cell sits about ' + fmt(rel) + ' m higher than its neighbouring cells');

    return { adverse: adverse, favourable: favourable };
  }

  /* ---- Step 6: describing every input of the selected cell ---- */
  var EXPLANATION_LABEL = 'Model-based prototype explanation';
  var DEMO_RAIN_RANGE = { min: 10, max: 220 };   // range of the demo training data (data/generate_demo_data.py)
  var LOW_WEIGHT = 0.05;                         // model importance below this = "little weight in the model"
  var FLAG_TEXT = { higher: 'higher vulnerability', lower: 'lower vulnerability', neutral: 'neutral', unused: 'not used' };
  var CATEGORY_ORDER = ['road', 'school', 'hospital', 'residential', 'infrastructure', 'low_lying'];
  var CATEGORY_LABELS = {
    road: 'Roads', school: 'Schools', hospital: 'Hospitals', residential: 'Residential areas',
    infrastructure: 'Public infrastructure', low_lying: 'Low-lying areas'
  };
  var GENERAL_ACTIONS = {
    HIGH: ['Monitor heavy rainfall conditions.', 'Inspect and maintain drainage pathways.',
           'Clear potential drainage blockages.', 'Increase water-retention and green areas where feasible.'],
    MEDIUM: ['Monitor rainfall conditions.', 'Review local drainage conditions.', 'Maintain clear stormwater pathways.'],
    LOW: ['Continue routine drainage maintenance.', 'Maintain green/open areas.', 'Continue monitoring during heavy rainfall.']
  };

  function rainfallBand(mm) {
    var frac = (mm - DEMO_RAIN_RANGE.min) / (DEMO_RAIN_RANGE.max - DEMO_RAIN_RANGE.min);
    var span = DEMO_RAIN_RANGE.min + '\u2013' + DEMO_RAIN_RANGE.max + ' mm demo range';
    if (frac >= 2 / 3) return { flag: 'higher', text: 'upper third of the ' + span };
    if (frac < 1 / 3) return { flag: 'lower', text: 'lower third of the ' + span };
    return { flag: 'neutral', text: 'middle third of the ' + span };
  }

  // Where does this cell's elevation sit among the other cells of the area?
  function elevationPosition(cell, stats) {
    var all = (stats && stats.elevations) || [];
    if (all.length < 4) return { flag: 'neutral', text: 'too few cells to compare' };
    var e = Number(cell.elevation_m), others = all.length - 1;
    var below = all.filter(function (v) { return v < e; }).length;
    var above = all.filter(function (v) { return v > e; }).length;
    if (below <= 0.25 * others) return { flag: 'higher', text: 'lower than ' + above + ' of the other ' + others + ' cells' };
    if (below >= 0.75 * others) return { flag: 'lower', text: 'higher than ' + below + ' of the other ' + others + ' cells' };
    return { flag: 'neutral', text: 'in the middle of the area (higher than ' + below + ' of the other ' + others + ' cells)' };
  }

  // One entry per model input (plus historical flood frequency, which is NOT used). The "flag" comes
  // from simple visible rules comparing the cell with the rest of the area. "weight" is the model's
  // own feature importance.
  function describeInputs(cell, stats, importance) {
    var imp = importance || {};
    function weight(key) { return isNum(imp[key]) ? Number(imp[key]) : null; }
    var rain = rainfallBand(Number(cell.rainfall_mm));
    var elev = elevationPosition(cell, stats);
    var rel = Number(cell.elevation_relative_m);
    var d = Number(cell.drainage_condition);
    var pct = Number(cell.impervious_surface_pct);
    var mean = stats ? stats.meanImpervious : null;

    var relFlag = rel <= RULES.lowSpotMeters ? 'higher' : rel >= RULES.highSpotMeters ? 'lower' : 'neutral';
    var relText = relFlag === 'higher' ? 'a local low spot (at least 1 m below neighbouring cells)'
      : relFlag === 'lower' ? 'at least 1 m higher than neighbouring cells' : 'roughly level with neighbouring cells';
    var drainFlag = d <= RULES.drainagePoorMax ? 'higher' : d >= RULES.drainageGoodMin ? 'lower' : 'neutral';
    var drainText = drainFlag === 'higher' ? 'poor or very poor drainage'
      : drainFlag === 'lower' ? 'good or very good drainage' : 'moderate drainage';
    var impFlag = 'neutral', impText = 'no area average available';
    if (mean !== null && mean !== undefined) {
      impFlag = pct > mean + RULES.imperviousMargin ? 'higher' : pct < mean - RULES.imperviousMargin ? 'lower' : 'neutral';
      impText = (impFlag === 'higher' ? 'above' : impFlag === 'lower' ? 'below' : 'close to') + ' the area average of ' + fmt(mean) + '%';
    }
    return [
      { key: 'rainfall_mm', label: 'Rainfall (24 h)', value: fmt(cell.rainfall_mm, 0) + ' mm', context: rain.text, flag: rain.flag, weight: weight('rainfall_mm') },
      { key: 'elevation_m', label: 'Elevation', value: fmt(cell.elevation_m) + ' m', context: elev.text, flag: elev.flag, weight: weight('elevation_m') },
      { key: 'elevation_relative_m', label: 'Low-spot depth (vs neighbours)', value: fmtSigned(rel) + ' m', context: relText, flag: relFlag, weight: weight('elevation_relative_m') },
      { key: 'drainage_condition', label: 'Drainage condition', value: drainageText(d), context: drainText, flag: drainFlag, weight: weight('drainage_condition') },
      { key: 'impervious_surface_pct', label: 'Impervious surface', value: fmt(pct) + '%', context: impText, flag: impFlag, weight: weight('impervious_surface_pct') },
      { key: 'historical_flood_frequency', label: 'Historical flood frequency', value: 'not used',
        context: 'no verified flood history exists in this prototype', flag: 'unused', weight: null }
    ];
  }

  // "Model-based prototype explanation": the model's own numbers + simple visible rules.
  // It does NOT claim to know real-world causes, and it never says "because".
  function buildExplanation(ctx) {
    var cell = ctx.cell, p = ctx.prediction;
    var inputs = describeInputs(cell, ctx.stats, p.feature_importance);
    var higher = inputs.filter(function (i) { return i.flag === 'higher'; });
    var lower = inputs.filter(function (i) { return i.flag === 'lower'; });
    function names(list) {
      return list.map(function (i) { return i.label.toLowerCase() + ' (' + i.value + ')'; }).join(', ');
    }

    var summary = [];
    summary.push(higher.length
      ? 'Conditions that indicate higher vulnerability here: ' + names(higher) + '.'
      : 'No input of this cell indicates higher vulnerability under the simple prototype rules.');
    if (lower.length) summary.push('Conditions that indicate lower vulnerability: ' + names(lower) + '.');
    var minor = higher.filter(function (i) { return i.weight !== null && i.weight < LOW_WEIGHT; });
    if (minor.length) {
      summary.push('Of the higher-vulnerability conditions, ' + minor.map(function (i) {
        return i.label.toLowerCase() + ' (model weight ' + fmt(i.weight, 3) + ')';
      }).join(', ') + ' carry little weight in the model overall, so they have limited influence on the score.');
    }

    var factors = topFactors(p.feature_importance).slice(0, 3);
    var modelNote = factors.length
      ? 'The model\u2019s overall weights are largest for ' + factors.map(function (f) {
          return f.label.toLowerCase() + ' (' + fmt(f.value, 3) + ')';
        }).join(', ') + '. These weights are model-wide, not specific to this cell.'
      : 'The model did not provide feature-importance values.';

    return {
      label: EXPLANATION_LABEL,
      headline: 'Under a ' + fmt(cell.rainfall_mm, 0) + ' mm rainfall scenario, the model gives cell ' + cell.grid_id +
        ' a ' + p.risk_level + ' model-estimated score of ' + fmt(p.risk_score) + '.',
      inputs: inputs,
      summary: summary,
      modelNote: modelNote,
      caveat: EXPLANATION_LABEL + ': it combines this model\u2019s feature importance with simple, visible rules that compare ' +
        'the cell with the rest of the area. It describes how inputs and model weights line up, not real-world causes, and it is ' +
        'not a per-prediction method such as SHAP. Inputs and training labels are synthetic/demo data.'
    };
  }

  // One card per model factor (real importance values), plus a card for the factor we do NOT use.
  function buildFactorCards(cell, prediction, stats) {
    var factors = topFactors(prediction.feature_importance);
    var inputs = describeInputs(cell, stats, prediction.feature_importance);
    var total = factors.reduce(function (sum, f) { return sum + f.value; }, 0) || 1;
    var max = factors.length ? factors[0].value || 1 : 1;
    var cards = factors.map(function (f, i) {
      var input = inputs.filter(function (x) { return x.key === f.key; })[0];
      return { rank: i + 1, key: f.key, label: f.label, importance: f.value, share: 100 * f.value / total,
               barPct: 100 * f.value / max, valueText: input ? input.value : '\u2014', flag: input ? input.flag : 'neutral' };
    });
    cards.push({ unused: true, key: 'historical_flood_frequency', label: 'Historical flood frequency',
                 note: 'Not used: no verified flood history exists in this prototype.' });
    return cards;
  }

  // General prototype recommendations from the risk level, plus items specific to this cell's conditions.
  function buildRecommendations(level, cell, stats) {
    var l = String(level).toUpperCase();
    var cond = conditionsForCell(cell, stats);
    var citizens = [], authorities = [], specific = [];

    if (l === 'HIGH') {
      citizens.push('Avoid unnecessary travel through this area during severe rainfall, and avoid flooded roads.');
      citizens.push('Follow official weather warnings and keep emergency contacts and basic supplies ready.');
      authorities.push('Monitor vulnerable roads and low spots; consider pre-positioning response teams and traffic diversions if flooding is reported.');
    } else if (l === 'MEDIUM') {
      citizens.push('Take extra care on low-lying roads during heavy rain and allow extra travel time.');
      citizens.push('Follow official weather warnings for your area.');
      authorities.push('Keep monitoring rainfall conditions and be ready to escalate if conditions worsen.');
    } else if (l === 'LOW') {
      citizens.push('No special action is indicated by this scenario; keep following official forecasts.');
      authorities.push('Continue routine monitoring.');
    }

    if (cond.adverse.some(function (t) { return t.indexOf('Drainage') === 0; })) {
      specific.push('Clear blocked drainage paths and prioritise drain maintenance here (drainage is ' + drainageText(cell.drainage_condition) + ').');
    }
    if (cond.adverse.some(function (t) { return t.indexOf('Impervious') === 0; })) {
      specific.push('Consider water-retention and green or permeable areas to slow runoff (impervious surface is ' + fmt(cell.impervious_surface_pct) + '%).');
    }
    if (cond.adverse.some(function (t) { return t.indexOf('lower than') !== -1; })) {
      specific.push('This cell is lower than its surroundings, so water may collect: avoid storing valuables in basements or low-lying spots during heavy rain.');
    }
    return { level: l, actions: GENERAL_ACTIONS[l] || [], citizens: citizens, authorities: authorities, specific: specific };
  }

  // ---- vulnerable places ----
  function groupLocations(list) {
    var groups = {};
    CATEGORY_ORDER.forEach(function (k) { groups[k] = []; });
    list.forEach(function (loc) { (groups[loc.type] = groups[loc.type] || []).push(loc); });
    return groups;
  }

  function vulnerabilitySummary(list) {
    var verified = list.filter(function (l) { return l.data_status === 'verified'; }).length;
    var demo = list.length - verified;
    var banner = null;
    if (verified === 0) banner = 'Vulnerability data is currently demo/prototype data.';
    else if (demo > 0) banner = demo + ' of ' + list.length + ' listed places are demo data.';
    return { total: list.length, verified: verified, demo: demo, banner: banner };
  }

  // Grid cells that sit at least 1 m below their neighbours (derived from the grid's elevation data)
  function lowLyingCells(cells, n) {
    var low = cells.filter(function (c) { return Number(c.elevation_relative_m) <= RULES.lowSpotMeters; })
      .sort(function (a, b) { return (a.elevation_relative_m - b.elevation_relative_m) || (a.grid_id < b.grid_id ? -1 : 1); });
    return { total: low.length, top: low.slice(0, n) };
  }

  // Sentence for the What-If result: describes the MODEL's output only.
  function describeChange(baseScore, scenarioScore) {
    var diff = round1(Number(scenarioScore) - Number(baseScore));
    var pair = '(' + fmt(baseScore) + ' \u2192 ' + fmt(scenarioScore) + ')';
    if (diff < 0) return { direction: 'lower', diff: diff, text: 'Under this hypothetical scenario, the model estimates a lower risk score ' + pair + '.' };
    if (diff > 0) return { direction: 'higher', diff: diff, text: 'Under this hypothetical scenario, the model estimates a higher risk score ' + pair + '.' };
    return { direction: 'same', diff: 0, text: 'Under this hypothetical scenario, the model estimates no change in the risk score ' + pair + '.' };
  }

  // Turn the slider positions into the numbers sent to /api/simulate
  function scenarioFromControls(cell, controls) {
    var drainage = Math.min(4, Number(cell.drainage_condition) + Number(controls.drainageSteps || 0));
    var impervious = round1(Math.max(0, Number(cell.impervious_surface_pct) - Number(controls.imperviousCut || 0)));
    return { rainfall_mm: Number(controls.rainfallMm), drainage_condition: drainage, impervious_surface_pct: impervious };
  }

  function predictPayload(cell) {
    return {
      rainfall_mm: cell.rainfall_mm,
      elevation_m: cell.elevation_m,
      elevation_relative_m: cell.elevation_relative_m,
      drainage_condition: cell.drainage_condition,
      impervious_surface_pct: cell.impervious_surface_pct
    };
  }

  function popupHtml(cell) {
    var k = levelKey(cell.risk_level);
    return '<div class="cs-pop">' +
      '<h4>Grid cell ' + escapeHtml(cell.grid_id) + ' <span class="level-pill level-' + k + '">' + escapeHtml(cell.risk_level) + '</span></h4>' +
      '<dl>' +
      '<dt>Model-estimated score</dt><dd>' + fmt(cell.risk_score) + ' / 100</dd>' +
      '<dt>Rainfall scenario</dt><dd>' + fmt(cell.rainfall_mm, 0) + ' mm</dd>' +
      '<dt>Elevation</dt><dd>' + fmt(cell.elevation_m) + ' m</dd>' +
      '<dt>vs neighbours</dt><dd>' + fmtSigned(cell.elevation_relative_m) + ' m</dd>' +
      '<dt>Drainage</dt><dd>' + escapeHtml(drainageText(cell.drainage_condition)) + '</dd>' +
      '<dt>Impervious surface</dt><dd>' + fmt(cell.impervious_surface_pct) + '%</dd>' +
      '</dl>' +
      '<p>Demo / synthetic inputs. Clicking a cell selects it for analysis.</p></div>';
  }

  /* ---- exports for the Node unit tests (does nothing in the browser) ---- */
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      escapeHtml: escapeHtml, fmt: fmt, fmtSigned: fmtSigned, round1: round1, levelKey: levelKey,
      drainageText: drainageText, cellBounds: cellBounds, computeAreaStats: computeAreaStats,
      highestCells: highestCells, topFactors: topFactors, conditionsForCell: conditionsForCell,
      buildExplanation: buildExplanation, buildRecommendations: buildRecommendations,
      describeChange: describeChange, scenarioFromControls: scenarioFromControls, predictPayload: predictPayload,
      popupHtml: popupHtml, rainfallBand: rainfallBand, elevationPosition: elevationPosition,
      describeInputs: describeInputs, buildFactorCards: buildFactorCards, groupLocations: groupLocations,
      vulnerabilitySummary: vulnerabilitySummary, lowLyingCells: lowLyingCells, EXPLANATION_LABEL: EXPLANATION_LABEL
    };
  }
  if (typeof document === 'undefined') return;

  /* =====================================================================
     3. PAGE CODE (needs the browser)
     ===================================================================== */
  var $ = function (id) { return document.getElementById(id); };

  var state = {
    rainfall: 100,
    analyzedRainfall: null,
    mapData: null,
    cells: [],
    byId: {},
    stats: null,
    selectedId: null,
    prediction: null,
    analyzeToken: 0,
    predictToken: 0,
    retry: null,
    transparency: null,
    vuln: null,
    mapEngine: null,           // 'leaflet' or 'fallback'
    map: null, gridLayer: null, vulnLayer: null, rects: {}, chart: null
  };

  /* ---------------------------------------------------------------------
     3a. fetch() wrapper: timeout, JSON parsing, readable errors
     --------------------------------------------------------------------- */
  function ApiFailure(message, kind, status) {
    this.name = 'ApiFailure'; this.message = message; this.kind = kind; this.status = status;
  }
  ApiFailure.prototype = Object.create(Error.prototype);

  function request(path, options) {
    options = options || {};
    var controller = new AbortController();
    var timer = setTimeout(function () { controller.abort(); }, REQUEST_TIMEOUT_MS);
    var headers = { Accept: 'application/json' };
    if (options.body) headers['Content-Type'] = 'application/json';

    return fetch(path, { method: options.method || 'GET', body: options.body, headers: headers, signal: controller.signal })
      .catch(function (err) {
        if (err && err.name === 'AbortError') throw new ApiFailure('The server took too long to respond.', 'timeout');
        throw new ApiFailure('Cannot reach the ClimateShield backend. Is "python app.py" still running?', 'network');
      })
      .then(function (response) {
        clearTimeout(timer);
        return response.json().then(function (data) { return { response: response, data: data }; },
                                    function () { return { response: response, data: null }; });
      })
      .then(function (r) {
        if (!r.response.ok) {
          throw new ApiFailure((r.data && r.data.error) || ('The server returned HTTP ' + r.response.status + '.'), 'http', r.response.status);
        }
        if (r.data === null) throw new ApiFailure('The server sent a response that is not valid JSON.', 'parse');
        return r.data;
      })
      .catch(function (err) { clearTimeout(timer); throw err; });
  }

  /* ---------------------------------------------------------------------
     3b. small UI helpers
     --------------------------------------------------------------------- */
  function showGlobalError(message, retryFn) {
    state.retry = retryFn || null;
    $('global-alert-text').textContent = message;
    $('global-alert-retry').hidden = !retryFn;
    $('global-alert').hidden = false;
  }
  function hideGlobalError() { $('global-alert').hidden = true; state.retry = null; }

  function setBusy(button, busy, idleText) {
    button.disabled = busy;
    button.classList.toggle('loading', busy);
    button.querySelector('.btn-label').textContent = busy ? 'Working' : idleText;
  }

  function skeleton(lines) {
    var html = '<div class="skeleton tall"></div>';
    for (var i = 0; i < (lines || 2); i++) html += '<div class="skeleton"></div>';
    return html;
  }

  function inlineError(message, retryId) {
    return '<div class="inline-error">' + escapeHtml(message) +
      (retryId ? ' <button type="button" class="btn-ghost" id="' + retryId + '">Retry</button>' : '') + '</div>';
  }

  function animateNumber(el, to, ms) {
    var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    var from = isNum(el.dataset.value) ? Number(el.dataset.value) : 0;
    el.dataset.value = to;
    if (reduce || !window.requestAnimationFrame) { el.textContent = fmt(to); return; }
    var start = null;
    function step(ts) {
      if (start === null) start = ts;
      var t = Math.min(1, (ts - start) / ms);
      el.textContent = fmt(from + (to - from) * (1 - Math.pow(1 - t, 3)));
      if (t < 1) requestAnimationFrame(step); else el.textContent = fmt(to);
    }
    requestAnimationFrame(step);
  }

  /* ---------------------------------------------------------------------
     3c. health check
     --------------------------------------------------------------------- */
  function checkHealth() {
    var box = $('api-status');
    return request(API.health).then(function (h) {
      var ok = h.status === 'running' && h.model_loaded;
      box.className = 'status ' + (ok ? 'ok' : 'warn');
      box.querySelector('span').textContent = ok ? 'API online \u00b7 model loaded' : 'API degraded \u00b7 ' + (h.problem || 'model not loaded');
      $('footer-status').textContent = 'Data status: ' + (h.data_status || 'not provided by the API');
    }).catch(function (err) {
      box.className = 'status bad';
      box.querySelector('span').textContent = 'Backend unreachable';
      $('footer-status').textContent = 'Data status: unavailable (' + err.message + ')';
    });
  }

  /* ---------------------------------------------------------------------
     3d. MAP  (Leaflet if it loaded, otherwise a simple clickable grid)
     --------------------------------------------------------------------- */
  function legendHtml() {
    return '<strong>Model-estimated risk</strong>' + LEVELS.map(function (l) {
      var range = l === 'LOW' ? '0\u201339' : l === 'MEDIUM' ? '40\u201369' : '70\u2013100';
      return '<div><span class="legend-sw" style="background:' + LEVEL_COLORS[l] + '"></span>' + l + ' \u00b7 ' + range + '</div>';
    }).join('') + '<small>Prototype thresholds, not official.</small>';
  }

  function initLeaflet(grid) {
    var b = grid.bounds;
    var bounds = L.latLngBounds([b.lat_min, b.lon_min], [b.lat_max, b.lon_max]);
    state.map = L.map('map', { zoomControl: true });
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO', subdomains: 'abcd', maxZoom: 19
    }).addTo(state.map);
    state.gridLayer = L.layerGroup().addTo(state.map);
    state.vulnLayer = L.layerGroup().addTo(state.map);
    // dashed outline of the prototype study area (the page text says it is not the official boundary)
    L.rectangle(bounds, { color: '#22d3ee', weight: 1.5, dashArray: '6 6', fill: false, interactive: false }).addTo(state.map);
    state.map.fitBounds(bounds, { padding: [16, 16] });

    var legend = L.control({ position: 'bottomright' });
    legend.onAdd = function () {
      var div = L.DomUtil.create('div', 'map-legend');
      div.innerHTML = legendHtml();
      return div;
    };
    legend.addTo(state.map);
    L.control.layers(null, { 'Risk grid': state.gridLayer, 'Vulnerable places': state.vulnLayer }, { collapsed: true }).addTo(state.map);
    setTimeout(function () { state.map.invalidateSize(); }, 250);
  }

  function chooseMapEngine(grid) {
    if (typeof L !== 'undefined') {
      try { initLeaflet(grid); state.mapEngine = 'leaflet'; return; } catch (e) { console.error('Leaflet failed to start:', e); }
    }
    state.mapEngine = 'fallback';
  }

  function drawLeafletGrid(data) {
    state.gridLayer.clearLayers();
    state.rects = {};
    data.cells.forEach(function (cell) {
      var color = LEVEL_COLORS[cell.risk_level] || '#64748b';
      var rect = L.rectangle(cellBounds(cell, data.grid.cell_size_deg), {
        color: color, weight: 1, fillColor: color, fillOpacity: 0.55
      });
      rect.bindPopup(popupHtml(cell), { maxWidth: 280 });
      rect.bindTooltip(cell.grid_id + ' \u00b7 ' + fmt(cell.risk_score), { sticky: true });
      rect.on('click', function () { selectCell(cell.grid_id); });
      rect.addTo(state.gridLayer);
      state.rects[cell.grid_id] = rect;
    });
  }

  function drawFallbackGrid(data) {
    var map = $('map');
    var rows = data.grid.rows, cols = data.grid.cols;
    map.innerHTML = '';
    map.classList.add('is-fallback');
    var legend = document.createElement('div');
    legend.className = 'map-legend fb-legend';
    legend.innerHTML = legendHtml();
    map.appendChild(legend);

    var grid = document.createElement('div');
    grid.className = 'fallback-grid';
    grid.style.gridTemplateColumns = 'repeat(' + cols + ', 1fr)';
    grid.style.gridTemplateRows = 'repeat(' + rows + ', 1fr)';
    var byPos = {};
    data.cells.forEach(function (c) { byPos[c.grid_row + ',' + c.grid_col] = c; });
    for (var r = rows - 1; r >= 0; r--) {            // north (high row number) at the top
      for (var c = 0; c < cols; c++) {
        var cell = byPos[r + ',' + c];
        if (!cell) { grid.appendChild(document.createElement('span')); continue; }
        var btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'fb-cell';
        btn.id = 'fb-' + cell.grid_id;
        btn.style.background = LEVEL_COLORS[cell.risk_level] || '#64748b';
        btn.style.opacity = '0.85';
        btn.textContent = fmt(cell.risk_score, 0);
        btn.title = 'Cell ' + cell.grid_id + ' \u00b7 ' + cell.risk_level + ' \u00b7 ' + fmt(cell.risk_score);
        btn.setAttribute('aria-label', 'Grid cell ' + cell.grid_id + ', ' + cell.risk_level + ' risk, score ' + fmt(cell.risk_score));
        (function (id) { btn.addEventListener('click', function () { selectCell(id); }); })(cell.grid_id);
        grid.appendChild(btn);
      }
    }
    map.appendChild(grid);
    var banner = document.createElement('div');
    banner.className = 'fb-banner';
    banner.textContent = 'Interactive map library could not load (internet is needed for Leaflet). Showing the grid as a simple clickable view.';
    map.appendChild(banner);
  }

  function renderMap(data) {
    if (state.mapEngine === null) chooseMapEngine(data.grid);
    if (state.mapEngine === 'leaflet') drawLeafletGrid(data); else drawFallbackGrid(data);
    $('map-mode').textContent = state.mapEngine === 'leaflet' ? 'Leaflet \u00b7 interactive' : 'Fallback grid view';
    highlightSelected();
  }

  function highlightSelected() {
    if (state.mapEngine === 'leaflet') {
      Object.keys(state.rects).forEach(function (id) {
        var cell = state.byId[id];
        var color = LEVEL_COLORS[cell.risk_level] || '#64748b';
        state.rects[id].setStyle(id === state.selectedId
          ? { color: '#ffffff', weight: 3, fillOpacity: 0.7 }
          : { color: color, weight: 1, fillOpacity: 0.55 });
      });
      if (state.rects[state.selectedId] && state.rects[state.selectedId].bringToFront) state.rects[state.selectedId].bringToFront();
    } else if (state.mapEngine === 'fallback') {
      document.querySelectorAll('.fb-cell').forEach(function (el) {
        el.classList.toggle('selected', el.id === 'fb-' + state.selectedId);
      });
    }
  }

  function focusCell(id) {
    var cell = state.byId[id];
    if (!cell) return;
    selectCell(id);
    if (state.mapEngine === 'leaflet' && state.map) {
      try {
        state.map.setView([cell.latitude, cell.longitude], Math.max(state.map.getZoom(), 14));
        state.rects[id].openPopup();
      } catch (e) { /* the map is optional decoration here */ }
    }
    var mapEl = $('map');
    if (mapEl.scrollIntoView) mapEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  function drawVulnerabilityMarkers(locations) {
    if (state.mapEngine !== 'leaflet' || !state.vulnLayer) return;
    state.vulnLayer.clearLayers();
    locations.forEach(function (loc) {
      L.circleMarker([loc.latitude, loc.longitude], { radius: 7, color: '#22d3ee', weight: 2, fillColor: '#06101f', fillOpacity: 0.9 })
        .bindPopup('<div class="cs-pop"><h4>' + escapeHtml(loc.name) + '</h4><dl><dt>Type</dt><dd>' + escapeHtml(loc.type) +
          '</dd><dt>Data</dt><dd>' + escapeHtml(loc.data_status) + '</dd><dt>Source</dt><dd>' + escapeHtml(loc.source) +
          '</dd></dl><p>Being listed does not mean this place is unsafe.</p></div>')
        .addTo(state.vulnLayer);
    });
  }

  /* ---------------------------------------------------------------------
     3e. OVERALL CARD  (data from GET /api/risk-map)
     --------------------------------------------------------------------- */
  function renderOverall(data) {
    var o = data.overall;
    var k = levelKey(o.risk_level);
    var gauge = $('gauge');
    gauge.className = 'gauge level-' + k + '-g';
    $('gauge-arc').style.strokeDashoffset = String(326.73 * (1 - Math.max(0, Math.min(100, o.risk_score)) / 100));
    animateNumber($('overall-score'), o.risk_score, 800);
    var pill = $('overall-level');
    pill.textContent = o.risk_level;
    pill.className = 'level-pill level-' + k;
    $('overall-method').textContent = o.method + ' Based on ' + data.cells.length + ' grid cells.';

    var total = data.cells.length || 1;
    $('level-bars').innerHTML = LEVELS.map(function (l) {
      var n = (data.level_counts && data.level_counts[l]) || 0;
      return '<div class="lb-row"><span>' + l + '</span><div class="lb-track"><div class="lb-fill" style="width:' +
        (100 * n / total).toFixed(1) + '%;background:' + LEVEL_COLORS[l] + '"></div></div><b>' + n + '</b></div>';
    }).join('');

    var s = data.scenario;
    $('overall-scenario').textContent = 'Rainfall scenario: ' + fmt(s.rainfall_mm, 0) + ' mm \u00b7 ' + s.rainfall_source + '. ' + s.note;
    $('footer-thresholds').textContent = data.thresholds_note || '';
  }

  function renderOverallLoading() {
    $('overall-method').textContent = 'Asking the model\u2026';
    $('overall-level').textContent = '\u2026';
    $('overall-level').className = 'level-pill level-na';
  }

  function renderOverallError(err) {
    $('overall-score').textContent = '\u2014';
    $('overall-score').dataset.value = '';
    $('gauge').className = 'gauge';
    $('gauge-arc').style.strokeDashoffset = '326.73';
    $('overall-level').textContent = 'N/A';
    $('overall-level').className = 'level-pill level-na';
    $('overall-method').textContent = 'Could not load the overall score: ' + err.message;
    $('level-bars').innerHTML = '';
  }

  /* ---------------------------------------------------------------------
     3f. TOP CELLS + VULNERABLE PLACES
     --------------------------------------------------------------------- */
  function renderTopCells() {
    var box = $('topcells');
    var top = highestCells(state.cells, 5);
    if (!top.length) { box.innerHTML = '<p class="empty">No grid cells to show.</p>'; return; }
    box.innerHTML = top.map(function (c, i) {
      return '<button type="button" class="top-item" data-id="' + escapeHtml(c.grid_id) + '">' +
        '<span class="top-left"><span class="rank">' + (i + 1) + '</span><b>' + escapeHtml(c.grid_id) + '</b>' +
        '<span class="meta">' + fmt(c.latitude, 4) + ', ' + fmt(c.longitude, 4) + '</span></span>' +
        '<span><b>' + fmt(c.risk_score) + '</b> <span class="level-pill level-' + levelKey(c.risk_level) + '">' + escapeHtml(c.risk_level) + '</span></span></button>';
    }).join('');
    box.querySelectorAll('.top-item').forEach(function (el) {
      el.addEventListener('click', function () { focusCell(el.dataset.id); });
    });
  }

  // Grid cells that are at least 1 m below their neighbours (derived from the grid's elevation data)
  function renderLowCells() {
    var box = $('lowcells');
    var low = lowLyingCells(state.cells, 5);
    if (!low.total) {
      box.innerHTML = '<p class="empty">No grid cell is at least 1 m lower than its neighbours.</p>';
      return;
    }
    box.innerHTML = '<p class="tiny muted">' + low.total + ' of ' + state.cells.length + ' grid cells sit at least 1 m below their neighbours. ' +
      'Lowest five shown. Whether the terrain is demo or real is stated in Data &amp; Model Transparency.</p>' +
      low.top.map(function (c) {
        return '<button type="button" class="top-item" data-id="' + escapeHtml(c.grid_id) + '">' +
          '<span class="top-left"><b>' + escapeHtml(c.grid_id) + '</b><span class="meta">elevation ' + fmt(c.elevation_m) + ' m</span></span>' +
          '<span><b>' + fmtSigned(c.elevation_relative_m) + ' m</b> <span class="meta">vs neighbours</span></span></button>';
      }).join('');
    box.querySelectorAll('.top-item').forEach(function (el) {
      el.addEventListener('click', function () { focusCell(el.dataset.id); });
    });
  }

  function renderVulnCategories(list) {
    var groups = groupLocations(list);
    $('vuln-cats').innerHTML = CATEGORY_ORDER.map(function (k) {
      var n = groups[k].length;
      return '<span class="cat-chip' + (n ? ' has' : '') + '">' + escapeHtml(CATEGORY_LABELS[k]) + ' <b>' + n + '</b></span>';
    }).join('');
  }

  function loadVulnerabilities() {
    var box = $('vuln-body');
    box.innerHTML = skeleton(1);
    return request(API.vulnerabilities).then(function (data) {
      var list = data.locations || [];
      var sum = vulnerabilitySummary(list);
      renderVulnCategories(list);
      $('vuln-chip').textContent = list.length ? list.length + ' place' + (list.length === 1 ? '' : 's') + ' \u00b7 prototype' : 'Prototype \u00b7 no dataset';
      box.innerHTML = '';
      if (sum.banner) {
        var banner = document.createElement('div');
        banner.className = 'banner-demo';
        banner.textContent = sum.banner;
        box.appendChild(banner);
      }
      if (!list.length) {
        var note = document.createElement('p');
        note.className = 'empty';
        note.textContent = (data.note || '') + ' Schools, hospitals, roads and similar places will appear here only when a real, ' +
          'source-labelled dataset (or clearly marked demo rows) is added. Nothing is invented.';
        box.appendChild(note);
      } else {
        var groups = groupLocations(list);
        Object.keys(groups).forEach(function (type) {
          if (!groups[type].length) return;
          var group = document.createElement('div');
          group.className = 'loc-group';
          var h = document.createElement('h4');
          h.textContent = CATEGORY_LABELS[type] || type;
          group.appendChild(h);
          groups[type].forEach(function (loc) {
            var row = document.createElement('div');
            row.className = 'loc-item';
            var left = document.createElement('b');
            left.textContent = loc.name;                      // textContent: names from the data can never run as code
            var right = document.createElement('span');
            right.className = 'chip ' + (loc.data_status === 'verified' ? 'chip-cyan' : 'chip-amber');
            right.textContent = loc.data_status + ' \u00b7 ' + loc.source;
            row.appendChild(left); row.appendChild(right);
            group.appendChild(row);
          });
          box.appendChild(group);
        });
      }
      drawVulnerabilityMarkers(list);
      state.vuln = list;
    }).catch(function (err) {
      $('vuln-cats').innerHTML = '';
      box.innerHTML = inlineError('Could not load vulnerable places: ' + err.message, 'vuln-retry');
      var btn = $('vuln-retry');
      if (btn) btn.addEventListener('click', loadVulnerabilities);
    });
  }

  /* ---------------------------------------------------------------------
     3g. SELECTED CELL  (data from POST /api/predict)
     --------------------------------------------------------------------- */
  function renderCellLoading(id) {
    $('cell-id').textContent = id;
    $('cell-body').innerHTML = skeleton(3);
  }

  function renderCellCard(cell, p) {
    var inp = p.input_used;
    $('cell-id').textContent = cell.grid_id;
    $('cell-body').innerHTML =
      '<div class="cell-top"><div class="cell-score">' + fmt(p.risk_score) + '</div><div>' +
      '<span class="level-pill level-' + levelKey(p.risk_level) + '">' + escapeHtml(p.risk_level) + '</span>' +
      '<p class="tiny muted">' + escapeHtml(p.score_label) + '</p></div></div>' +
      '<div class="metrics">' +
      metric('Rainfall scenario', fmt(inp.rainfall_mm, 0) + ' mm') +
      metric('Elevation', fmt(inp.elevation_m) + ' m') +
      metric('Vs neighbours', fmtSigned(inp.elevation_relative_m) + ' m') +
      metric('Drainage', drainageText(inp.drainage_condition)) +
      metric('Impervious surface', fmt(inp.impervious_surface_pct) + '%') +
      metric('Model', p.model_status) +
      '</div>' +
      '<p class="tiny muted">From POST /api/predict. Historical flood frequency is not used: no verified flood history exists in this prototype.</p>';
  }
  function metric(label, value) {
    return '<div class="metric"><span>' + escapeHtml(label) + '</span><b>' + escapeHtml(value) + '</b></div>';
  }

  function renderCellError(err, id) {
    $('cell-body').innerHTML = inlineError('Could not analyse cell ' + id + ': ' + err.message, 'cell-retry');
    var btn = $('cell-retry');
    if (btn) btn.addEventListener('click', function () { selectCell(id); });
  }

  /* ---------------------------------------------------------------------
     3h. RISK FACTORS (Chart.js, with a plain-HTML fallback)
     --------------------------------------------------------------------- */
  function factorCardHtml(c) {
    if (c.unused) {
      return '<div class="fcard unused"><div class="fcard-top"><span class="fcard-rank">\u2013</span>' + escapeHtml(c.label) +
        '</div><p class="tiny muted" style="margin:0">' + escapeHtml(c.note) + '</p></div>';
    }
    return '<div class="fcard"><div class="fcard-top"><span class="fcard-rank">' + c.rank + '</span>' + escapeHtml(c.label) + '</div>' +
      '<div class="fcard-bar"><div class="fcard-fill" style="width:' + c.barPct.toFixed(1) + '%"></div></div>' +
      '<div class="fcard-nums"><span>importance <b>' + fmt(c.importance, 3) + '</b></span><span>' + fmt(c.share, 0) + '% of total</span></div>' +
      '<div class="fcard-cell"><span>This cell: <b>' + escapeHtml(c.valueText) + '</b></span>' +
      '<span class="flag flag-' + c.flag + '">' + escapeHtml(FLAG_TEXT[c.flag] || '') + '</span></div></div>';
  }

  function renderFactors(cell, p) {
    var factors = topFactors(p.feature_importance);
    var box = $('factors-box'), cardsBox = $('factor-cards');
    if (state.chart) { state.chart.destroy(); state.chart = null; }

    if (!factors.length) {
      box.hidden = true;
      cardsBox.innerHTML = '<p class="empty">The model did not return feature-importance values.</p>';
      $('factors-note').textContent = '';
      return;
    }

    box.hidden = true;                                   // shown only if Chart.js loads and draws
    if (typeof Chart !== 'undefined') {
      try {
        box.hidden = false;
        Chart.defaults.color = '#94a7c7';
        Chart.defaults.font.family = '"Inter","Segoe UI",system-ui,sans-serif';
        var colors = factors.map(function (f, i) { return i === 0 ? '#34d399' : i === 1 ? '#22d3ee' : 'rgba(34,211,238,0.55)'; });
        state.chart = new Chart($('factors-chart'), {
          type: 'bar',
          data: { labels: factors.map(function (f) { return f.label; }),
                  datasets: [{ data: factors.map(function (f) { return f.value; }), backgroundColor: colors, borderRadius: 6, borderSkipped: false }] },
          options: {
            indexAxis: 'y', responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false },
                       tooltip: { callbacks: { label: function (c) { return ' importance ' + Number(c.parsed.x).toFixed(3); } } } },
            scales: { x: { beginAtZero: true, grid: { color: 'rgba(148,170,205,0.12)' } },
                      y: { grid: { display: false }, ticks: { color: '#eaf2ff' } } }
          }
        });
      } catch (e) {
        console.error('Chart.js failed:', e);
        box.hidden = true;                               // the factor cards below still show every value
      }
    }
    cardsBox.innerHTML = buildFactorCards(cell, p, state.stats).map(factorCardHtml).join('');
    $('factors-note').textContent = (p.feature_importance_note || '') + ' Values are read from the trained model. ' +
      '"This cell" flags come from simple prototype rules, not from the model.';
  }

  /* ---------------------------------------------------------------------
     3i. EXPLANATION + RECOMMENDATIONS
     --------------------------------------------------------------------- */
  function listHtml(items, cls) {
    return '<ul>' + items.map(function (t) { return '<li class="' + (cls || '') + '">' + escapeHtml(t) + '</li>'; }).join('') + '</ul>';
  }

  var FLAG_LI = { higher: 'adverse', lower: 'favourable', neutral: 'neutral', unused: 'unused' };

  function renderExplanation(cell, p) {
    var e = buildExplanation({ cell: cell, prediction: p, stats: state.stats });
    var html = '<p><strong>' + escapeHtml(e.headline) + '</strong></p><h4>Inputs considered</h4><ul>' +
      e.inputs.map(function (i) {
        return '<li class="' + FLAG_LI[i.flag] + '"><b>' + escapeHtml(i.label) + '</b> ' + escapeHtml(i.value) + ' \u2014 ' + escapeHtml(i.context) +
          ' <span class="flag flag-' + i.flag + '">' + escapeHtml(FLAG_TEXT[i.flag]) + '</span>' +
          (i.weight !== null ? '<span class="wt">model weight ' + fmt(i.weight, 3) + '</span>' : '') + '</li>';
      }).join('') + '</ul>' +
      e.summary.map(function (t) { return '<p>' + escapeHtml(t) + '</p>'; }).join('') +
      '<p>' + escapeHtml(e.modelNote) + '</p><p class="caveat">' + escapeHtml(e.caveat) + '</p>';
    $('explain-body').innerHTML = html;
  }

  function renderRecommendations(cell, p) {
    var r = buildRecommendations(p.risk_level, cell, state.stats);
    var html = '<div class="reco"><p class="small">Cell <b>' + escapeHtml(cell.grid_id) + '</b> \u00b7 ' + escapeHtml(p.risk_level) + ' in this scenario</p>';
    if (r.actions.length) {
      html += '<h4 class="actions-title">Preventive actions \u00b7 ' + escapeHtml(r.level) + ' risk</h4>' + listHtml(r.actions);
    } else {
      html += '<p class="empty">No recommendation is defined for the risk level "' + escapeHtml(p.risk_level) + '".</p>';
    }
    html += '<div class="reco-cols"><div><h4>For citizens</h4>' + listHtml(r.citizens) +
      '</div><div><h4>For authorities</h4>' + listHtml(r.authorities) + '</div></div>';
    if (r.specific.length) html += '<h4>Specific to this cell</h4>' + listHtml(r.specific);
    $('reco-body').innerHTML = html + '</div>';
  }

  function clearDerivedPanels() {
    if (state.chart) { state.chart.destroy(); state.chart = null; }
    $('factors-box').hidden = true;
    $('factor-cards').innerHTML = '<p class="empty">Risk factors are shown after a successful prediction.</p>';
    $('factors-note').textContent = '';
    $('explain-body').innerHTML = '<p class="empty">No explanation available because the prediction failed.</p>';
    $('reco-body').innerHTML = '<p class="empty">No recommendations available because the prediction failed.</p>';
  }

  /* ---------------------------------------------------------------------
     3j. SELECT A CELL  ->  POST /api/predict
     --------------------------------------------------------------------- */
  function selectCell(id) {
    var cell = state.byId[id];
    if (!cell) return Promise.resolve();
    state.selectedId = id;
    state.prediction = null;
    highlightSelected();
    renderCellLoading(id);
    resetSimulator(cell);

    var token = ++state.predictToken;
    return request(API.predict, { method: 'POST', body: JSON.stringify(predictPayload(cell)) }).then(function (p) {
      if (token !== state.predictToken) return;           // the user already clicked another cell
      state.prediction = p;
      renderCellCard(cell, p);
      renderFactors(cell, p);
      renderExplanation(cell, p);
      renderRecommendations(cell, p);
    }).catch(function (err) {
      if (token !== state.predictToken) return;
      renderCellError(err, id);
      clearDerivedPanels();
    });
  }

  /* ---------------------------------------------------------------------
     3k. ANALYZE RISK  ->  GET /api/risk-map
     --------------------------------------------------------------------- */
  function analyze() {
    var token = ++state.analyzeToken;
    var button = $('analyze');
    setBusy(button, true, 'Analyze Risk');
    hideGlobalError();
    renderOverallLoading();
    var rainfall = state.rainfall;

    return request(API.riskMap + '?rainfall_mm=' + encodeURIComponent(rainfall)).then(function (data) {
      if (token !== state.analyzeToken) return;
      if (!data.cells || !data.cells.length) throw new ApiFailure('The map data contained no grid cells.', 'empty');
      state.mapData = data;
      state.cells = data.cells;
      state.byId = {};
      data.cells.forEach(function (c) { state.byId[c.grid_id] = c; });
      state.stats = computeAreaStats(data.cells);
      state.analyzedRainfall = data.scenario.rainfall_mm;

      renderOverall(data);
      renderMap(data);
      renderTopCells();
      renderLowCells();
      if (state.vuln) drawVulnerabilityMarkers(state.vuln);

      // keep the user's selected cell if it still exists, otherwise pick the highest-scoring one
      var keep = state.selectedId && state.byId[state.selectedId];
      return selectCell(keep ? state.selectedId : highestCells(data.cells, 1)[0].grid_id);
    }).catch(function (err) {
      if (token !== state.analyzeToken) return;
      renderOverallError(err);
      showGlobalError('Could not load the risk map: ' + err.message, analyze);
    }).then(function () {
      if (token === state.analyzeToken) setBusy(button, false, 'Analyze Risk');
    });
  }

  /* ---------------------------------------------------------------------
     3l. WHAT-IF SIMULATOR  ->  POST /api/simulate
     --------------------------------------------------------------------- */
  function simControls() {
    return {
      drainageSteps: Number($('sim-drainage').value),
      imperviousCut: Number($('sim-green').value),
      rainfallMm: Number($('sim-rain').value)
    };
  }

  function updateSimOutputs() {
    var cell = state.byId[state.selectedId];
    if (!cell) return;
    var c = simControls();
    var drainNew = Math.min(4, Number(cell.drainage_condition) + c.drainageSteps);
    $('sim-drainage-out').textContent = c.drainageSteps === 0 ? 'no change'
      : '+' + (drainNew - cell.drainage_condition) + ' level' + ((drainNew - cell.drainage_condition) === 1 ? '' : 's') + ' (' + cell.drainage_condition + ' \u2192 ' + drainNew + ')';
    $('sim-green-out').textContent = c.imperviousCut === 0 ? 'no change'
      : '\u2212' + c.imperviousCut + ' pts (' + fmt(cell.impervious_surface_pct) + '% \u2192 ' + fmt(round1(Math.max(0, cell.impervious_surface_pct - c.imperviousCut))) + '%)';
    $('sim-rain-out').textContent = fmt(c.rainfallMm, 0) + ' mm' + (c.rainfallMm === state.analyzedRainfall ? ' (same as baseline)' : '');
  }

  function resetSimulator(cell) {
    var drain = $('sim-drainage'), green = $('sim-green'), rain = $('sim-rain');
    var maxDrain = 4 - Number(cell.drainage_condition);
    var maxGreen = Math.min(40, Math.floor(Number(cell.impervious_surface_pct) / 5) * 5);
    drain.max = String(maxDrain); drain.value = '0'; drain.disabled = maxDrain <= 0;
    green.max = String(maxGreen); green.value = '0'; green.disabled = maxGreen <= 0;
    rain.value = String(state.analyzedRainfall === null ? state.rainfall : state.analyzedRainfall); rain.disabled = false;
    $('sim-run').disabled = false; $('sim-reset').disabled = false;
    $('sim-cell').textContent = 'Baseline: cell ' + cell.grid_id + ' \u00b7 drainage ' + drainageText(cell.drainage_condition) +
      ' \u00b7 impervious ' + fmt(cell.impervious_surface_pct) + '% \u00b7 rainfall ' + fmt(cell.rainfall_mm, 0) + ' mm.' +
      (maxDrain <= 0 ? ' Drainage is already at the top level.' : '') + (maxGreen <= 0 ? ' Impervious surface is already very low.' : '');
    $('sim-result').innerHTML = '<p class="empty">Adjust a lever and press <strong>Run scenario</strong>. The scores come from the Flask API / saved model.</p>';
    updateSimOutputs();
  }

  function runScenario() {
    var cell = state.byId[state.selectedId];
    if (!cell) return;
    var button = $('sim-run');
    var scenario = scenarioFromControls(cell, simControls());
    var body = { grid_id: cell.grid_id, baseline: { rainfall_mm: cell.rainfall_mm }, scenario: scenario };
    setBusy(button, true, 'Run scenario');
    $('sim-result').innerHTML = skeleton(2);

    request(API.simulate, { method: 'POST', body: JSON.stringify(body) }).then(function (r) {
      renderSimResult(r);
    }).catch(function (err) {
      $('sim-result').innerHTML = inlineError('The scenario could not be calculated: ' + err.message, 'sim-retry');
      var btn = $('sim-retry');
      if (btn) btn.addEventListener('click', runScenario);
    }).then(function () { setBusy(button, false, 'Run scenario'); });
  }

  function renderSimResult(r) {
    var change = describeChange(r.baseline_risk_score, r.scenario_risk_score);
    var changed = Object.keys(r.changed_inputs || {}).map(function (key) {
      var c = r.changed_inputs[key];
      return '<li class="chip">' + escapeHtml(FEATURE_LABELS[key] || key) + ': ' + fmt(c.from, key === 'drainage_condition' || key === 'rainfall_mm' ? 0 : 1) +
        ' \u2192 ' + fmt(c.to, key === 'drainage_condition' || key === 'rainfall_mm' ? 0 : 1) + '</li>';
    }).join('');
    $('sim-result').innerHTML =
      '<div class="sim-compare">' +
        '<div class="sim-box"><small>Baseline risk</small><div class="big">' + fmt(r.baseline_risk_score) + '</div>' +
          '<span class="level-pill level-' + levelKey(r.baseline_risk_level) + '">' + escapeHtml(r.baseline_risk_level) + '</span></div>' +
        '<div class="sim-arrow" aria-hidden="true">\u2192</div>' +
        '<div class="sim-box scenario"><small>Scenario risk (hypothetical)</small><div class="big">' + fmt(r.scenario_risk_score) + '</div>' +
          '<span class="level-pill level-' + levelKey(r.scenario_risk_level) + '">' + escapeHtml(r.scenario_risk_level) + '</span></div>' +
      '</div>' +
      '<p class="sim-change">Change in model-estimated score: <b>' + fmtSigned(r.difference) + ' points</b><br>' + escapeHtml(change.text) + '</p>' +
      (changed ? '<ul class="chg-list">' + changed + '</ul>' : '') +
      (r.warning ? '<p class="tiny muted">' + escapeHtml(r.warning) + '</p>' : '') +
      '<p class="sim-label">' + escapeHtml(r.label || FALLBACK_SCENARIO_LABEL) + '</p>' +
      '<p class="tiny muted">' + escapeHtml(r.disclaimer || '') + '</p>';
  }

  /* ---------------------------------------------------------------------
     3m. DATA & MODEL TRANSPARENCY  (GET /api/transparency)
     --------------------------------------------------------------------- */
  function itemList(items) {
    return '<ul>' + items.map(function (i) {
      return '<li><b>' + escapeHtml(i.item) + '.</b> <span>' + escapeHtml(i.detail || i.reason || '') + '</span></li>';
    }).join('') + '</ul>';
  }

  function renderTransparency(t) {
    var real = t.real_data && t.real_data.length
      ? itemList(t.real_data)
      : '<p><b>None in this build.</b></p><p>' + escapeHtml(t.real_data_note || '') + '</p>';

    var demo = itemList(t.demo_data || []);
    if (t.excluded && t.excluded.length) {
      demo += '<p class="tiny"><b>Deliberately not used:</b></p>' + itemList(t.excluded.map(function (x) {
        return { item: x.item, detail: x.reason };
      }));
    }

    var m = t.model, model;
    if (m) {
      model = '<p><b>' + escapeHtml(m.model || 'Model') + '</b> (Random Forest), trained ' + escapeHtml(String(m.trained_at || 'at an unknown time')) + '.</p>' +
        '<p>Trained on ' + escapeHtml(String(m.training_rows)) + ' rows and tested on ' + escapeHtml(String(m.testing_rows)) +
        ' rows of the demo data, using ' + escapeHtml(String((m.features || []).length)) + ' input features.</p>';
    } else {
      model = '<p><b>Model details are not available</b> (model/model_metadata.json was not found).</p>';
    }
    model += '<p>' + escapeHtml(t.model_note || '') + '</p><p>' + escapeHtml(t.thresholds_note || '') + '</p>';

    var hypo = '<p>' + escapeHtml(t.hypothetical_note || '') + '</p><p>Shown in the What-If Simulator under the label \u201c' +
      escapeHtml(FALLBACK_SCENARIO_LABEL) + '\u201d</p>';

    var html =
      '<div class="tile tile-real"><span class="tag tag-real">REAL DATA</span>' + real + '</div>' +
      '<div class="tile tile-demo"><span class="tag tag-demo">DEMO / SYNTHETIC DATA</span>' + demo + '</div>' +
      '<div class="tile tile-model"><span class="tag tag-model">MODEL PREDICTION</span>' + model + '</div>' +
      '<div class="tile tile-hypo"><span class="tag tag-hypo">HYPOTHETICAL SCENARIO</span>' + hypo + '</div>';

    if (m && m.test_metrics) {
      var tm = m.test_metrics, cv = m.cross_validation || {};
      var rows = [['Accuracy', 'accuracy'], ['Precision', 'precision'], ['Recall', 'recall'], ['F1 score', 'f1'], ['ROC-AUC', 'roc_auc']];
      html += '<details class="eval"><summary>Model self-check on the synthetic labels (not real-world accuracy)</summary><table>' +
        '<tr><th>Metric</th><th>Held-out test set</th><th>Cross-validation (training cells)</th></tr>' +
        rows.map(function (r) {
          var c = cv[r[1]];
          return '<tr><td>' + r[0] + '</td><td>' + fmt(tm[r[1]], 3) + '</td><td>' +
            (c ? fmt(c.mean, 3) + ' \u00b1 ' + fmt(c.std, 3) : '\u2014') + '</td></tr>';
        }).join('') + '</table><p class="tiny muted">' + escapeHtml(m.disclaimer || '') +
        ' These numbers show how well the model reproduces the demo rule that created the labels.</p></details>';
    }
    $('transp-body').innerHTML = html;
  }

  function loadTransparency() {
    var box = $('transp-body');
    box.innerHTML = skeleton(2);
    return request(API.transparency).then(function (t) {
      state.transparency = t;
      renderTransparency(t);
    }).catch(function (err) {
      box.innerHTML = inlineError('Could not load the transparency details: ' + err.message, 'transp-retry');
      var btn = $('transp-retry');
      if (btn) btn.addEventListener('click', loadTransparency);
    });
  }

  /* ---------------------------------------------------------------------
     3n. EVENTS + START
     --------------------------------------------------------------------- */
  function setRainfall(mm) {
    state.rainfall = Number(mm);
    $('rainfall').value = String(mm);
    $('rainfall-out').textContent = mm + ' mm';
    document.querySelectorAll('#rain-presets button').forEach(function (b) {
      b.classList.toggle('active', Number(b.dataset.mm) === state.rainfall);
    });
    if (state.analyzedRainfall !== null && state.analyzedRainfall !== state.rainfall) {
      $('overall-scenario').textContent = 'Rainfall changed to ' + mm + ' mm \u2014 press Analyze Risk to update the map and scores.';
    }
  }

  function bindEvents() {
    $('rainfall').addEventListener('input', function (e) { setRainfall(Number(e.target.value)); });
    document.querySelectorAll('#rain-presets button').forEach(function (b) {
      b.addEventListener('click', function () { setRainfall(Number(b.dataset.mm)); });
    });
    $('analyze').addEventListener('click', analyze);
    $('global-alert-retry').addEventListener('click', function () { if (state.retry) state.retry(); });
    ['sim-drainage', 'sim-green', 'sim-rain'].forEach(function (id) { $(id).addEventListener('input', updateSimOutputs); });
    $('sim-run').addEventListener('click', runScenario);
    $('sim-reset').addEventListener('click', function () { var c = state.byId[state.selectedId]; if (c) resetSimulator(c); });
    window.addEventListener('resize', function () { if (state.map) state.map.invalidateSize(); });
  }

  function init() {
    bindEvents();
    setRainfall(Number($('rainfall').value));
    checkHealth();
    loadTransparency();
    // load the map first, then the vulnerable places (their markers need the map to exist)
    analyze().then(loadVulnerabilities);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})(typeof window !== 'undefined' ? window : globalThis);
