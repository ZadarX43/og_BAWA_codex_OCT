(function () {
  const DATA_ROOT = "./public/data";
  const PREMIUM_TOKEN_STORAGE_KEY = "og_premium_token";
  const app = document.getElementById("app");
  const page = document.body.dataset.page || "home";
  const query = new URLSearchParams(window.location.search);
  const premiumDemoMode = query.get("demo") === "1";
  const debugMode = query.get("debug") === "1";
  const accountIntent = query.get("intent") || "";
  const checkoutState = query.get("checkout") || "";
  const runtimeConfig = window.OG_CONFIG || {};
  const workerApiBase = String(runtimeConfig.WORKER_API_BASE || "").replace(/\/+$/, "");
  const checkoutPlaceholderHref = "./account.html?intent=checkout";

  const state = {
    summary: null,
    publicPredictions: [],
    premiumPredictions: [],
    securePremiumPredictions: [],
    weeklyResults: null,
    runtime: {
      workerApiBase,
      premiumToken: null,
      premiumFetchError: "",
      premiumSourceLabel: "",
      premiumGeneratedAt: "",
      premiumSubscriberCustomerId: "",
      checkoutMessage: "",
      accountMessage: "",
    },
  };

  const fetchJson = async (path) => {
    const response = await fetch(path, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Failed to load ${path}`);
    }
    return response.json();
  };

  const fetchOptionalJson = async (path) => {
    const response = await fetch(path, { cache: "no-store" });
    if (!response.ok) {
      return null;
    }
    return response.json();
  };

  const escapeHtml = (value) =>
    String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");

  const readStoredPremiumToken = () => {
    try {
      return window.localStorage.getItem(PREMIUM_TOKEN_STORAGE_KEY) || "";
    } catch {
      return "";
    }
  };

  const writeStoredPremiumToken = (token) => {
    try {
      if (token) {
        window.localStorage.setItem(PREMIUM_TOKEN_STORAGE_KEY, token);
      } else {
        window.localStorage.removeItem(PREMIUM_TOKEN_STORAGE_KEY);
      }
    } catch {
      return;
    }
  };

  const workerConfigured = () => Boolean(state.runtime.workerApiBase);
  const premiumTokenPresent = () => Boolean(state.runtime.premiumToken);

  const workerApiUrl = (path) => {
    if (!workerConfigured()) {
      return "";
    }
    return new URL(path, `${state.runtime.workerApiBase}/`).toString();
  };

  const fetchWorkerJson = async (path, options = {}) => {
    const headers = new Headers(options.headers || {});
    headers.set("accept", "application/json");
    if (options.body && !headers.has("content-type")) {
      headers.set("content-type", "application/json");
    }
    if (options.withToken && state.runtime.premiumToken) {
      headers.set("authorization", `Bearer ${state.runtime.premiumToken}`);
    }
    const response = await fetch(workerApiUrl(path), {
      method: options.method || "GET",
      headers,
      body: options.body ? JSON.stringify(options.body) : undefined,
    });
    let payload = null;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }
    return { response, payload };
  };

  const statPanel = (label, value, note = "") => `
    <article class="panel">
      <span class="muted">${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      ${note ? `<span>${escapeHtml(note)}</span>` : ""}
    </article>
  `;

  const renderNotice = (message, tone = "default") =>
    message ? `<div class="notice notice-${escapeHtml(tone)}">${escapeHtml(message)}</div>` : "";

  const formatProbability = (value) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
      return "N/A";
    }
    return `${Math.round(numeric * 100)}%`;
  };

  const edgeLabel = (row) => row.value_edge_display || row.value_edge || "N/A";
  const confidenceLabel = (row) => row.display_confidence || row.model_prob_display || formatProbability(row.model_prob);
  const tierClass = (tier) => (String(tier || "").toUpperCase() === "STANDARD" ? "standard" : "elite");

  const proofTile = (label, value, note = "") => `
    <article class="proof-tile">
      <span class="metric-label">${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      ${note ? `<span class="muted">${escapeHtml(note)}</span>` : ""}
    </article>
  `;

  const compactPercent = (value) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
      return "Pending";
    }
    return `${(numeric * 100).toFixed(2)}%`;
  };

  const resultsStatusTone = (value) => {
    if (value == null || !Number.isFinite(Number(value))) {
      return "neutral";
    }
    if (Number(value) >= 0.8) {
      return "good";
    }
    if (Number(value) >= 0.55) {
      return "warn";
    }
    return "bad";
  };

  const buildProofProfile = (weekly) => {
    const total = Math.max(Number(weekly.total_picks) || 0, 1);
    const points = [{ label: "Window open", y: 0.14 }];
    let running = 0;

    (weekly.by_market || []).forEach((item) => {
      running += Number(item.total_picks) || 0;
      const coverage = running / total;
      const settledShare = (Number(item.settled_picks) || 0) / total;
      const hit = Number.isFinite(Number(item.hit_rate)) ? Number(item.hit_rate) : 0.55;
      const value = Math.min(0.9, 0.16 + coverage * 0.42 + settledShare * 0.22 + hit * 0.18);
      points.push({ label: item.market, y: value });
    });

    const overall = Number.isFinite(Number(weekly.overall_hit_rate)) ? Number(weekly.overall_hit_rate) : 0.5;
    const finalValue = Math.min(
      0.94,
      0.2 + ((Number(weekly.settled_picks) || 0) / total) * 0.3 + ((Number(weekly.pending_picks) || 0) / total) * 0.06 + overall * 0.34
    );
    points.push({ label: "Published proof", y: finalValue });
    return points;
  };

  const renderProofCurve = (weekly) => {
    const points = buildProofProfile(weekly);
    const width = 860;
    const height = 300;
    const xStep = width / Math.max(points.length - 1, 1);
    const coords = points.map((point, index) => ({
      x: index * xStep,
      y: height - point.y * height,
      label: point.label,
    }));

    const linePath = coords.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`).join(" ");
    const areaPath = `${linePath} L ${width} ${height} L 0 ${height} Z`;
    const xLabels = coords
      .map(
        (point) => `
          <span>${escapeHtml(point.label)}</span>
        `
      )
      .join("");

    return `
      <article class="proof-graph-panel">
        <div class="proof-graph-head">
          <div>
            <p class="hero-kicker">Proof profile</p>
            <h2>Current graded window curve</h2>
            <p class="section-copy">
              Derived from the current published window using cumulative settled share and hit-rate checkpoints.
              It is a proof profile for this release window, not a fabricated back-history chart.
            </p>
          </div>
          <div class="chart-legend">
            <span><i class="legend-dot legend-dot-live"></i> AI proof profile</span>
            <span><i class="legend-dot"></i> Window baseline</span>
          </div>
        </div>
        <div class="proof-chart-shell">
          <svg viewBox="0 0 ${width} ${height}" class="proof-chart" aria-hidden="true" preserveAspectRatio="none">
            <defs>
              <linearGradient id="proofGlow" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#31c7ff"></stop>
                <stop offset="100%" stop-color="#4edea3"></stop>
              </linearGradient>
              <linearGradient id="proofFill" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stop-color="rgba(78, 222, 163, 0.35)"></stop>
                <stop offset="100%" stop-color="rgba(78, 222, 163, 0.03)"></stop>
              </linearGradient>
            </defs>
            <line x1="0" y1="${height - height * 0.26}" x2="${width}" y2="${height - height * 0.26}" class="chart-baseline"></line>
            <path d="${areaPath}" fill="url(#proofFill)"></path>
            <path d="${linePath}" class="chart-line"></path>
            ${coords
              .map(
                (point) => `
                  <circle cx="${point.x.toFixed(2)}" cy="${point.y.toFixed(2)}" r="5" class="chart-point"></circle>
                `
              )
              .join("")}
          </svg>
          <div class="chart-x-labels">${xLabels}</div>
        </div>
      </article>
    `;
  };

  const predictionCard = (row, locked) => {
    const shortlist = Array.isArray(row.correct_score_shortlist) ? row.correct_score_shortlist : [];
    return `
      <article class="card prediction-card">
        <div class="prediction-top">
          <div class="teams">
            <span class="muted">${escapeHtml(row.league)} • ${escapeHtml(row.kickoff_time)}</span>
            <strong>${escapeHtml(row.home_team)} vs ${escapeHtml(row.away_team)}</strong>
          </div>
          <div class="pill-row">
            <span class="market-badge">${escapeHtml(row.market)}</span>
            <span class="confidence-badge ${tierClass(row.confidence_tier)}">${escapeHtml(row.confidence_tier)}</span>
          </div>
        </div>
        <div class="signal-grid">
          <div class="signal-cell">
            <span class="signal-label">Pick</span>
            <span class="signal-value">${escapeHtml(row.pick)}</span>
          </div>
          <div class="signal-cell">
            <span class="signal-label">Odds</span>
            <span class="signal-value">${escapeHtml(row.bookie_od ?? "N/A")}</span>
          </div>
          <div class="signal-cell">
            <span class="signal-label">${locked ? "Confidence" : "Model"}</span>
            <span class="signal-value">${escapeHtml(locked ? confidenceLabel(row) : formatProbability(row.model_prob))}</span>
          </div>
          <div class="signal-cell">
            <span class="signal-label">Edge</span>
            <span class="signal-value">${escapeHtml(locked ? row.value_edge_display || "Members only" : edgeLabel(row))}</span>
          </div>
        </div>
        <div class="prediction-footer">
          ${
            locked
              ? `<span class="premium-lock">Locked in free view</span>`
              : `<span class="value-badge">Value edge ${escapeHtml(edgeLabel(row))}</span>`
          }
          <span class="pill">${escapeHtml(confidenceLabel(row))}</span>
        </div>
        <p class="muted">${escapeHtml(
          row.short_reason || row.human_reason || "Cleared value-edge threshold vs market price."
        )}</p>
        ${
          !locked && shortlist.length
            ? `<div class="detail-row"><span class="muted">Correct-score support</span><span>${shortlist
                .map((item) => `${escapeHtml(item.scoreline)} (${escapeHtml(item.probability)})`)
                .join(" · ")}</span></div>`
            : ""
        }
      </article>
    `;
  };

  const boardTable = (rows, premium) => {
    if (!rows.length) {
      return `<div class="empty-state">No predictions are available for this board yet.</div>`;
    }
    return `
      <div class="table-shell">
        <table>
          <thead>
            <tr>
              <th>Fixture</th>
              <th>Market</th>
              <th>Pick</th>
              <th>Tier</th>
              <th>${premium ? "Model" : "Confidence"}</th>
              <th>Odds</th>
              <th>${premium ? "Reason" : "Edge"}</th>
            </tr>
          </thead>
          <tbody>
            ${rows
              .map(
                (row) => `
                  <tr>
                    <td>
                      <strong>${escapeHtml(row.home_team)} vs ${escapeHtml(row.away_team)}</strong><br />
                      <span class="muted">${escapeHtml(row.league)} • ${escapeHtml(row.kickoff_time)}</span>
                    </td>
                    <td>${escapeHtml(row.market)}</td>
                    <td>${escapeHtml(row.pick)}</td>
                    <td><span class="confidence-badge">${escapeHtml(row.confidence_tier)}</span></td>
                    <td>${escapeHtml(
                      premium ? row.model_prob ?? "N/A" : row.display_confidence || row.model_prob_display || "N/A"
                    )}</td>
                    <td>${escapeHtml(row.bookie_od ?? "N/A")}</td>
                    <td>${escapeHtml(
                      premium ? row.human_reason || "Qualified premium play." : row.value_edge_display || "N/A"
                    )}</td>
                  </tr>
                `
              )
              .join("")}
          </tbody>
        </table>
      </div>
    `;
  };

  const lockedPreviewCards = () => {
    const seedRows = state.publicPredictions.slice(0, 3);
    const rows = seedRows.length
      ? seedRows
      : [
          {
            league: "Premium board",
            kickoff_time: "Subscriber view",
            home_team: "Locked fixture",
            away_team: "Upgrade required",
            market: "ELITE",
            confidence_tier: "Premium",
          },
          {
            league: "Value edge board",
            kickoff_time: "Subscriber view",
            home_team: "Locked fixture",
            away_team: "Upgrade required",
            market: "STANDARD",
            confidence_tier: "Premium",
          },
        ];

    return rows
      .map(
        (row, index) => `
          <article class="card locked-card">
            <div class="prediction-top">
              <div class="teams">
                <span class="muted">${escapeHtml(row.league)} • ${escapeHtml(row.kickoff_time)}</span>
                <strong>${escapeHtml(row.home_team)} vs ${escapeHtml(row.away_team)}</strong>
              </div>
              <div class="pill-row">
                <span class="market-badge">${escapeHtml(row.market || "Premium")}</span>
                <span class="confidence-badge ${tierClass(row.confidence_tier)}">${escapeHtml(row.confidence_tier || "Locked")}</span>
              </div>
            </div>
            <div class="locked-copy">
              <span class="premium-lock">Locked while subscribed</span>
              <h3>Premium card ${index + 1}</h3>
              <p>
                Unlock full pick detail, sharper edge context, shortlist support, and the complete premium board.
              </p>
            </div>
            <div class="blur-stack" aria-hidden="true">
              <span class="blur-line blur-line-short"></span>
              <span class="blur-line"></span>
              <span class="blur-line blur-line-tiny"></span>
            </div>
          </article>
        `
      )
      .join("");
  };

  const workerStatusCopy = () => {
    if (!workerConfigured()) {
      return "Worker not configured";
    }
    return premiumTokenPresent() ? "Token detected" : "Token required";
  };

  const checkoutCta = () =>
    `<a class="button" data-action="worker-checkout" href="./account.html?intent=checkout">${
      workerConfigured() ? "Unlock founding membership" : "Open checkout placeholder"
    }</a>`;

  const homeView = () => `
    <section class="hero">
      <div class="hero-main">
        <p class="hero-kicker">Prediction intelligence system</p>
        <h1>Institutional-grade football prediction intelligence.</h1>
        <div class="pill-row">
          <span class="stat-chip">Precision</span>
          <span class="stat-chip">Control</span>
          <span class="stat-chip">Inevitability</span>
        </div>
        <p>
          Odds Genius identifies bookmaker mispricing and only deploys signals when the edge survives structural,
          volatility, and stability checks. This is not a betting brand or a flashy tipster shell. It is a
          prediction intelligence system built to detect when the market is wrong.
        </p>
        <div class="hero-actions">
          <a class="button" href="./predictions.html">View live board</a>
          <a class="ghost-button" href="./results.html">See proof</a>
          <a class="ghost-button" href="./premium.html">Unlock premium</a>
        </div>
        <div class="section-head home-proof-head">
          <div>
            <h2>ELITE / PREMIUM system performance</h2>
            <p class="section-copy">Consolidated 3-year walk-forward and backtested proof across the core stack.</p>
          </div>
        </div>
        <div class="proof-strip">
          ${proofTile("Value Edge Premium System", "83.3% hit rate", "15,203 rows • +53.9% ROI")}
          ${proofTile("FTR", "82%", "29 leagues • 3,000 fixtures")}
          ${proofTile("OU2.5 calibrated", "95.35%", "3,828 fixtures")}
          ${proofTile("BTTS calibrated", "93.55%", "3,382 fixtures")}
          ${proofTile("Correct Score", "41% direct", "81% top-3 shortlist hit rate")}
        </div>
      </div>
      <aside class="hero-side">
        <article class="sample-board deployment-stack">
          <div class="sample-board-head">
            <div>
              <span class="metric-label">Weekly production engine</span>
              <strong>65+ picks every week, year-round</strong>
            </div>
            <span class="pill">Lowest 87% • Avg 92%</span>
          </div>
          <div class="home-ladder">
            <article class="ladder-card ladder-card-elite">
              <span class="metric-label">ELITE / PREMIUM</span>
              <strong>High-ROI selective deployment</strong>
              <p class="muted">Value edge premium system, correct score support, acca formatting, and flagship filtered output.</p>
            </article>
            <article class="ladder-card ladder-card-standard">
              <span class="metric-label">STANDARD</span>
              <strong>65+ deployable weekly picks</strong>
              <p class="muted">Year-round output across FTR, OU2.5, BTTS, and adjacent production-safe markets.</p>
            </article>
            <article class="ladder-card ladder-card-soon">
              <span class="metric-label">COMING SOON</span>
              <strong>Player events</strong>
              <p class="muted">Shots, tackles, fouls, and bookings added as the next intelligence layer.</p>
            </article>
          </div>
        </article>
      </aside>
    </section>

    <section class="section split">
      <div>
        <div class="section-head">
          <div>
            <h2>Live public board</h2>
            <p class="section-copy">
              A compact look at the current free board. Premium members unlock the full deployable set, richer
              reasons, shortlist support, and Worker-protected access.
            </p>
          </div>
          <a class="ghost-button" href="./predictions.html">Open full board</a>
        </div>
        <div class="card-grid">
          ${state.publicPredictions.slice(0, 3).map((row) => predictionCard(row, true)).join("")}
        </div>
      </div>
      <article class="panel">
        <h3>Why this wins</h3>
        <ul class="method-list">
          <li>Most betting products predict every match. Odds Genius does the opposite.</li>
          <li>It scans broadly, filters aggressively, and only deploys when independent systems agree that the price is wrong and the signal is stable enough to act on.</li>
          <li>The edge is not just prediction accuracy. The edge is knowing when not to bet.</li>
        </ul>
      </article>
    </section>

    <section class="section split">
      <article class="panel">
        <h3>The decision layer is the moat.</h3>
        <ul class="method-list">
          <li>Core models estimate probability.</li>
          <li>Poisson goal mass checks match structure.</li>
          <li>Value edge compares model price against bookmaker price.</li>
          <li>Volatility and fragility filters suppress dangerous signals.</li>
          <li>When the systems align, the pick deploys. When they conflict, it stays out.</li>
        </ul>
      </article>
      <article class="panel">
        <h3>Why it stands apart</h3>
        <ul class="method-list">
          <li>Typical public tipster products struggle to sustain even 50–60% with lower volume and weaker proof discipline.</li>
          <li>Odds Genius combines a flagship selective layer with a weekly production engine delivering 65+ picks at scale.</li>
          <li>Consolidated proof includes FTR, OU2.5, BTTS, premium value edge, acca formatting, and correct score support.</li>
          <li>If the model doesn't beat the price, it doesn't deploy.</li>
        </ul>
      </article>
    </section>
  `;

  const predictionsView = () => `
    <section class="section">
      <div class="hero-main board-layout">
        <div class="board-toolbar">
          <div class="board-hero-copy">
            <p class="hero-kicker">Predictions Board</p>
            <h1>Live public picks.</h1>
          <p class="section-copy">
            Fast-scan board from the latest validated deploy. Premium carries the full signal density.
          </p>
          <p class="section-copy">All picks are derived from model probability vs bookmaker implied probability.</p>
        </div>
          <div class="pill-row">
            <span class="pill pill-elite">${state.summary.public_predictions_count} free picks</span>
            <span class="pill">${escapeHtml(sourceWindowLabel())}</span>
          </div>
        </div>
        <div class="proof-strip">
          ${proofTile("Source rows", state.summary.source_rows_read)}
          ${proofTile("Public cards", state.summary.public_predictions_count)}
          ${proofTile("Premium cards", state.summary.premium_predictions_count)}
          ${proofTile("Proof page", "Weekly results", "Settled outcomes published separately")}
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>Current board</h2>
          <p class="section-copy">
            Denser live cards with clearer hierarchy across market, pick, odds, confidence, and edge.
          </p>
        </div>
        <a class="ghost-button" href="./premium.html">See premium unlock</a>
      </div>
      <div class="card-grid">
        ${state.publicPredictions.map((row) => predictionCard(row, true)).join("")}
      </div>
    </section>
  `;

  const premiumView = () => {
    if (premiumDemoMode && state.premiumPredictions.length) {
      return `
        <section class="section split">
          <article class="hero-main">
            <p class="hero-kicker">Premium Demo Mode</p>
            <h1>Internal premium preview.</h1>
            <p>
              Demo mode is enabled for internal product review, so the exported premium board is rendered below.
              Customer-facing access should continue to rely on the Worker route.
            </p>
            <div class="cta-row">
              <a class="button" href="./pricing.html">See founding plan</a>
              <a class="ghost-button" href="./premium.html">Return to locked view</a>
            </div>
          </article>
          <aside class="hero-side">
            <div class="metric">
              <span class="metric-label">Rendered premium cards</span>
              <span class="metric-value">${state.summary.premium_predictions_count}</span>
            </div>
            <div class="metric">
              <span class="metric-label">Mode</span>
              <span class="metric-value">Demo</span>
            </div>
          </aside>
        </section>
        <section class="section">
          <div class="section-head">
            <div>
              <h2>Premium board preview</h2>
              <p class="section-copy">
                Internal-only rendering of the premium board for product review.
              </p>
            </div>
          </div>
          ${boardTable(state.premiumPredictions, true)}
        </section>
      `;
    }

    const secureBoardReady = state.securePremiumPredictions.length > 0;
    const workerMessage = state.runtime.premiumFetchError || (!premiumTokenPresent() ? "No premium token found." : "");

    return `
      <section class="section split">
        <article class="hero-main">
          <p class="hero-kicker">Premium Board</p>
          <h1>${secureBoardReady ? "Protected premium access is live." : "The strongest board stays locked by default."}</h1>
          <p>
            ${
              secureBoardReady
                ? "This board is being served through the Worker after token verification and subscriber-state checks."
                : "Premium unlocks the complete deployable board — not just more picks, but the deeper pricing intelligence behind them. Built for bettors who care about value, stability, and proof."
            }
          </p>
          <div class="cta-row">
            <a class="button" data-action="worker-checkout" href="./account.html?intent=checkout">Unlock founding membership — £20/month</a>
            <a class="ghost-button" href="./pricing.html">See pricing</a>
          </div>
        </article>
        <aside class="hero-side">
          <div class="metric">
            <span class="metric-label">Access state</span>
            <span class="metric-value">${escapeHtml(workerStatusCopy())}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Premium cards</span>
            <span class="metric-value">${secureBoardReady ? state.securePremiumPredictions.length : state.summary.premium_predictions_count}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Status</span>
            <span class="metric-value">${secureBoardReady ? "Unlocked" : "Locked"}</span>
          </div>
        </aside>
      </section>

      ${
        secureBoardReady
          ? `
            <section class="section">
              <div class="stats-grid">
                ${statPanel("Protected cards", state.securePremiumPredictions.length)}
                ${statPanel("Subscriber", state.runtime.premiumSubscriberCustomerId || "Verified")}
                ${statPanel("Worker source", state.runtime.premiumSourceLabel || "Configured")}
                ${statPanel("Generated", state.runtime.premiumGeneratedAt ? state.runtime.premiumGeneratedAt.slice(0, 10) : "Unknown")}
              </div>
            </section>
            <section class="section">
              ${boardTable(state.securePremiumPredictions, true)}
            </section>
          `
          : `
            <section class="section">
              ${renderNotice(
                workerMessage ||
                  (workerConfigured()
                    ? "Verified premium access is required before the full board is shown."
                    : "Worker premium access is not configured yet."),
                workerConfigured() ? "warning" : "default"
              )}
            </section>
            <section class="section">
              <div class="locked-grid">
                ${lockedPreviewCards()}
              </div>
            </section>
          `
      }

      <section class="section split">
        <article class="panel">
          <h3>What you actually get</h3>
          <ul class="feature-list">
            <li>Full deployable board.</li>
            <li>ELITE and STANDARD picks.</li>
            <li>Value edge vs bookmaker implied probability.</li>
            <li>Correct score shortlist support.</li>
            <li>Acca safety signals.</li>
            <li>Stability and fragility context.</li>
            <li>Worker-protected subscriber access.</li>
          </ul>
        </article>
        <article class="panel">
          <h3>Premium value tier performance</h3>
          <ul class="feature-list">
            <li>15,203 signals.</li>
            <li>83.31% hit rate.</li>
            <li>+8,194.99 units profit.</li>
            <li>+53.90% ROI.</li>
            <li>3-year walk-forward tested.</li>
          </ul>
        </article>
      </section>
      <section class="section">
        <div class="notice">Public shows the signal. Premium shows where the edge actually lives. Early pricing. Locked while subscribed.</div>
      </section>
    `;
  };

  const resultsView = () => {
    const weekly = state.weeklyResults;
    if (!weekly) {
      return `
        <section class="section split">
          <article class="hero-main">
            <p class="hero-kicker">Results & Proof</p>
            <h1>Proof layer waiting on settled grading.</h1>
            <p>
              This page will show settled proof once weekly results are available. The structure is live, but the
              current window has not been published here yet.
            </p>
          </article>
          <aside class="hero-side">
            <div class="metric">
              <span class="metric-label">Current source rows</span>
              <span class="metric-value">${state.summary.source_rows_read}</span>
            </div>
            <div class="metric">
              <span class="metric-label">Publish timestamp</span>
              <span class="metric-value">${escapeHtml(state.summary.generated_at.slice(0, 10))}</span>
            </div>
          </aside>
        </section>
        <section class="section">
          <div class="notice">
            Run grade_weekend_results.py after settled outcomes are available to publish weekly proof.
          </div>
        </section>
      `;
    }

    const marketCards = weekly.by_market
      .map(
        (item) => `
          <article class="panel market-proof-card market-proof-card--${resultsStatusTone(item.hit_rate)}">
            <span class="muted">${escapeHtml(item.market)}</span>
            <strong>${escapeHtml(item.hit_rate == null ? "Pending" : `${Math.round(item.hit_rate * 100)}%`)}</strong>
            <span>${escapeHtml(`${item.settled_picks}/${item.total_picks} settled`)}</span>
          </article>
        `
      )
      .join("");

    const tierCards = weekly.by_tier
      .map(
        (item) => `
          <article class="panel tier-panel tier-${String(item.tier || "").toLowerCase()}">
            <span class="muted">${escapeHtml(item.tier)}</span>
            <strong>${escapeHtml(item.hit_rate == null ? "Pending" : `${Math.round(item.hit_rate * 100)}%`)}</strong>
            <span>${escapeHtml(`${item.settled_picks}/${item.total_picks} settled`)}</span>
          </article>
        `
      )
      .join("");

    const featuredList = (rows, emptyLabel) =>
      rows.length
        ? rows
            .map(
              (row) => `
                <li>
                  <strong>${escapeHtml(row.home_team)} vs ${escapeHtml(row.away_team)}</strong><br />
                  <span class="muted">${escapeHtml(row.league)} • ${escapeHtml(row.market)} • ${escapeHtml(row.pick)} • ${escapeHtml(row.confidence_tier)}</span>
                </li>
              `
            )
            .join("")
        : `<li>${escapeHtml(emptyLabel)}</li>`;

    return `
      <section class="section split">
        <article class="hero-main">
          <p class="hero-kicker">Weekly Proof</p>
          <h1>Settled board proof.</h1>
          <p>
            Public-safe weekly proof generated from scored deploy outputs. Use this page to evaluate settled
            performance, not hype.
          </p>
          <p class="section-copy">Results are published independently from predictions.</p>
        </article>
        <aside class="hero-side">
          <div class="metric">
            <span class="metric-label">Window</span>
            <span class="metric-value">${escapeHtml(`${weekly.period_start} → ${weekly.period_end}`)}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Overall hit rate</span>
            <span class="metric-value">${escapeHtml(weekly.overall_hit_rate == null ? "Pending" : compactPercent(weekly.overall_hit_rate))}</span>
            <span class="muted">Tracked across ${weekly.total_picks} picks (${weekly.settled_picks} settled so far)</span>
          </div>
          <div class="metric">
            <span class="metric-label">Settled picks</span>
            <span class="metric-value">${escapeHtml(`${weekly.settled_picks}/${weekly.total_picks}`)}</span>
          </div>
        </aside>
      </section>

      <section class="section">
        <div class="results-highlight results-highlight--four">
          ${statPanel("Total picks", weekly.total_picks, `${weekly.period_start} → ${weekly.period_end}`)}
          ${statPanel("Settled picks", weekly.settled_picks, `${weekly.pending_picks} still live`)}
          ${statPanel("Pending picks", weekly.pending_picks, "Awaiting graded resolution")}
          ${statPanel("Hit rate", weekly.overall_hit_rate == null ? "Pending" : compactPercent(weekly.overall_hit_rate), weekly.generated_at.slice(0, 10))}
        </div>
      </section>

      <section class="section">
        ${renderProofCurve(weekly)}
      </section>

      <section class="section">
        <div class="section-head">
          <div>
            <h2>By market</h2>
            <p class="section-copy">Public-safe hit-rate summary for deployable markets in the graded window.</p>
          </div>
        </div>
        <div class="stats-grid">${marketCards}</div>
      </section>

      <section class="section">
        <div class="section-head">
          <div>
            <h2>By tier</h2>
            <p class="section-copy">Deployable board performance split by live confidence tier.</p>
          </div>
        </div>
        <div class="stats-grid">${tierCards}</div>
      </section>

      <section class="section split">
        <article class="panel featured-proof-panel">
          <h3>Featured wins</h3>
          <ul class="feature-list">${featuredList(weekly.featured_wins, "No settled wins surfaced yet.")}</ul>
        </article>
        <article class="panel">
          <h3>Featured misses</h3>
          <ul class="feature-list">${featuredList(weekly.featured_misses, "No settled misses surfaced yet.")}</ul>
        </article>
      </section>

      <section class="section">
        <div class="notice">
          ${(weekly.notes || []).map((note) => escapeHtml(note)).join("<br />") || "No additional notes."}
        </div>
      </section>
    `;
  };

  const pricingView = () => `
      <section class="section split">
        <article class="hero-main">
          <p class="hero-kicker">Pricing</p>
          <h1>Choose your intelligence level.</h1>
          <p>
            Start free. Secure founder pricing before the main Pro ladder expands.
          </p>
          <div class="pill-row">
            <span class="stat-chip">Worker-protected access</span>
            <span class="stat-chip">First 250 founders</span>
          </div>
          <div class="cta-row">
            ${checkoutCta().replace("Unlock founding membership", "Secure founder access")}
            <a class="ghost-button" href="./premium.html">Preview locked premium</a>
          </div>
        </article>
      <aside class="hero-side">
        <div class="metric">
          <span class="metric-label">Plan</span>
          <span class="metric-value">OG Founder</span>
        </div>
        <div class="metric">
          <span class="metric-label">Founder pricing</span>
          <span class="metric-value">£20/mo for life while active</span>
        </div>
        <div class="metric">
          <span class="metric-label">Cohort cap</span>
          <span class="metric-value">250 users</span>
        </div>
      </aside>
    </section>

    <section class="section">
      ${renderNotice(state.runtime.checkoutMessage, state.runtime.checkoutMessage ? "warning" : "default")}
      <div class="pricing-grid">
        <article class="card pricing-card pricing-card-free">
          <span class="pricing-tag">Free</span>
          <div class="pricing-price">£0</div>
          <p class="pricing-subcopy">Public preview of the board, proof, and methodology layer.</p>
          <ul class="feature-list">
            <li>Limited public board.</li>
            <li>Rounded confidence.</li>
            <li>Rounded edge display.</li>
            <li>Public proof access.</li>
            <li>No shortlist or premium context.</li>
          </ul>
        </article>
        <article class="card pricing-card featured pricing-card-premium">
          <span class="pricing-ribbon">First 250 users</span>
          <span class="pricing-tag">OG Founder</span>
          <div class="pricing-price">£20<span class="pricing-price-note">/month</span></div>
          <p class="pricing-subcopy">Fixed founder pricing while active. Full deployable board access, founder-only upside, and early access to selected advanced systems.</p>
          <ul class="feature-list">
            <li>Full deployable board.</li>
            <li>ELITE and STANDARD signals.</li>
            <li>Full value-edge layer.</li>
            <li>Correct score shortlist support.</li>
            <li>Acca safety indicators.</li>
            <li>Protected Worker-backed access.</li>
            <li>Early access to new markets.</li>
            <li>Early access to selected future systems.</li>
          </ul>
          <div class="notice founder-guardrail">
            £20/month for life while active. Non-transferable. Founder pricing ends after the first 250 users.
          </div>
          <div class="cta-row">
            ${checkoutCta().replace("Unlock founding membership", "Secure founder access")}
          </div>
        </article>
      </div>
      <div class="pricing-band">
        <article class="pricing-band-card">
          <span class="metric-label">Founder advantage</span>
          <strong>Grandfathered pricing while active</strong>
        </article>
        <article class="pricing-band-card">
          <span class="metric-label">Expansion path</span>
          <strong>Selected future systems and new markets unlocked earlier</strong>
        </article>
      </div>
      <div class="section-head pricing-matrix-head">
        <div>
          <h2>Technical matrix comparison</h2>
          <p class="section-copy">Useful for quickly seeing what the free board proves versus what founder access actually unlocks.</p>
        </div>
      </div>
      <div class="table-shell pricing-matrix-shell">
        <table class="pricing-matrix">
          <thead>
            <tr>
              <th>Feature detail</th>
              <th>Free tier</th>
              <th>OG Founder</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Board scope</td>
              <td>Limited public board</td>
              <td>Full deployable board</td>
            </tr>
            <tr>
              <td>Signal depth</td>
              <td>Rounded confidence and rounded edge display</td>
              <td>Full value-edge layer and premium explanation</td>
            </tr>
            <tr>
              <td>Shortlist support</td>
              <td>No shortlist or premium context</td>
              <td>Correct score shortlist support</td>
            </tr>
            <tr>
              <td>Slip support</td>
              <td>Public board only</td>
              <td>Acca safety indicators</td>
            </tr>
            <tr>
              <td>Protection layer</td>
              <td>Static public access</td>
              <td>Protected access through live Worker entitlement</td>
            </tr>
            <tr>
              <td>Founder upside</td>
              <td>None</td>
              <td>Early markets and selected advanced-system access</td>
            </tr>
          </tbody>
        </table>
      </div>
      <section class="pricing-visual-note">
        <article class="pricing-visual-card">
          <span class="metric-label">Founder access</span>
          <h2>Founder pricing now. Pro ladder later.</h2>
          <p class="section-copy">
            OG Founder sits above the free board and below the future Pro ladder. It gives the first cohort more than a normal entry tier without promising unlimited access to every future product line.
          </p>
        </article>
      </section>
      <section class="section">
        <article class="panel">
          <h3>Advanced plans coming later</h3>
          <p class="muted">
            Pro tiers will introduce deeper diagnostics, richer controls, and expanded workflow tooling.
          </p>
        </article>
      </section>
      <p class="footer-note">
        ${
          workerConfigured()
            ? "The founder CTA now routes to the live Worker checkout flow."
            : "Static-only mode cannot provide secure subscriber enforcement."
        }
      </p>
      <p class="footer-note">If you are serious about betting, you do not need more picks. You need better pricing.</p>
    </section>
  `;

  const methodologyView = () => `
    <section class="section split">
      <article class="hero-main">
        <p class="hero-kicker">Methodology</p>
        <h1>Built on generated outputs, not frontend guesswork.</h1>
        <p>
          Odds Genius does not re-run model logic in the browser. The website displays approved exports from the
          live deployment engine.
        </p>
      </article>
      <aside class="hero-side">
        <div class="metric">
          <span class="metric-label">Public schema fields</span>
          <span class="metric-value">${state.summary.public_fields.length}</span>
        </div>
        <div class="metric">
          <span class="metric-label">Premium schema fields</span>
          <span class="metric-value">${state.summary.premium_fields.length}</span>
        </div>
      </aside>
    </section>
    <section class="section">
      <div class="card-grid">
        <article class="panel">
          <h3>Pipeline boundary</h3>
          <p class="muted">Built on generated outputs, not frontend guesswork. Ingest, enrichment, routing, and slip logic remain in Python.</p>
        </article>
        <article class="panel">
          <h3>Decision stack</h3>
          <p class="muted">The system combines independent modelling approaches and only publishes signals that pass probability, structure, value, and stability checks.</p>
        </article>
        <article class="panel">
          <h3>Final line</h3>
          <p class="muted">Odds Genius is not built to predict every game. It is built to identify when the market is wrong.</p>
        </article>
      </div>
    </section>
  `;

  const accountView = () => {
    const checkoutNotice =
      checkoutState === "success"
        ? "Checkout returned successfully. Your membership unlocks premium board access."
        : checkoutState === "cancelled"
          ? "Checkout was cancelled. No premium access was granted."
          : "";

    return `
      <section class="section split">
        <article class="hero-main">
          <p class="hero-kicker">Account</p>
          <h1>Your membership unlocks premium board access.</h1>
          <p>
            Manage subscription and premium access from here. Subscription management is coming soon.
          </p>
          <div class="cta-row">
            <a class="button" href="./pricing.html">See founding plan</a>
            <a class="ghost-button" href="./premium.html">Open premium page</a>
          </div>
        </article>
        <aside class="hero-side">
          <div class="metric">
            <span class="metric-label">Worker</span>
            <span class="metric-value">${workerConfigured() ? "Configured" : "Placeholder"}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Stored token</span>
            <span class="metric-value">${premiumTokenPresent() ? "Present" : "Missing"}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Membership</span>
            <span class="metric-value">Premium-ready</span>
          </div>
        </aside>
      </section>

      <section class="section">
        ${renderNotice(checkoutNotice, checkoutState === "success" ? "success" : "warning")}
        ${debugMode ? renderNotice(state.runtime.accountMessage, state.runtime.accountMessage ? "warning" : "default") : ""}
      </section>

      ${
        debugMode
          ? `
            <section class="section split">
              <article class="panel">
                <h3>Developer/test token flow</h3>
                <p class="muted">
                  Internal only. This exists for controlled Worker verification before magic-link or authenticated
                  session issuance is finished.
                </p>
                <form id="premium-token-form" class="stack-form">
                  <label class="field-label" for="premium-token-input">Premium token</label>
                  <textarea id="premium-token-input" name="premium_token" class="text-input" rows="6" placeholder="Paste developer/test premium token here">${escapeHtml(
                    state.runtime.premiumToken || ""
                  )}</textarea>
                  <div class="cta-row">
                    <button class="button" type="submit">Save token</button>
                    <button class="ghost-button" type="button" data-action="clear-premium-token">Clear token</button>
                  </div>
                </form>
              </article>
              <article class="panel">
                <h3>${accountIntent === "checkout" ? "Checkout handoff" : "Subscription management placeholder"}</h3>
                <ul class="feature-list">
                  <li>Worker checkout route is used when WORKER_API_BASE is configured.</li>
                  <li>Webhook-backed subscriber state remains the authority for premium access.</li>
                  <li>Customer Portal and real sign-in still need production wiring later.</li>
                </ul>
              </article>
            </section>
          `
          : `
            <section class="section">
              <div class="notice">Manage subscription coming soon.</div>
            </section>
          `
      }
    `;
  };

  const sourceWindowLabel = () => {
    const source = state.summary?.selected_source_csv || "";
    const parts = source.split("/");
    return parts.length > 2 ? parts.slice(-3, -1).join(" / ") : source;
  };

  const render = () => {
    const views = {
      home: homeView,
      predictions: predictionsView,
      premium: premiumView,
      results: resultsView,
      pricing: pricingView,
      methodology: methodologyView,
      account: accountView,
    };
    const view = views[page] || homeView;
    app.innerHTML = view();
  };

  const renderError = (error) => {
    app.innerHTML = `
      <section class="section">
        <div class="empty-state">
          <strong>Data bridge unavailable.</strong>
          <p>${escapeHtml(error.message || "Unable to load published JSON.")}</p>
          <p class="muted">
            If you opened these files directly from the filesystem, use a small static server instead so fetch
            requests can resolve <code>frontend/public/data/*.json</code>.
          </p>
        </div>
      </section>
    `;
  };

  const loadProtectedPremiumPredictions = async () => {
    state.runtime.premiumFetchError = "";
    state.runtime.premiumGeneratedAt = "";
    state.runtime.premiumSubscriberCustomerId = "";
    state.runtime.premiumSourceLabel = "";
    state.securePremiumPredictions = [];

    if (premiumDemoMode || page !== "premium") {
      return;
    }
    if (!workerConfigured() || !premiumTokenPresent()) {
      return;
    }

    const { response, payload } = await fetchWorkerJson("/api/premium/predictions", {
      method: "GET",
      withToken: true,
    });

    if (!response.ok || !payload?.ok) {
      state.runtime.premiumFetchError = payload?.message || "Protected premium route did not return access.";
      return;
    }

    state.securePremiumPredictions = Array.isArray(payload.predictions) ? payload.predictions : [];
    state.runtime.premiumGeneratedAt = typeof payload.generated_at === "string" ? payload.generated_at : "";
    state.runtime.premiumSubscriberCustomerId =
      typeof payload.subscriber_customer_id === "string" ? payload.subscriber_customer_id : "";
    state.runtime.premiumSourceLabel = workerApiUrl("/api/premium/predictions");
  };

  const startWorkerCheckout = async () => {
    if (!workerConfigured()) {
      window.location.href = checkoutPlaceholderHref;
      return;
    }

    state.runtime.checkoutMessage = "Requesting Stripe Checkout from Worker...";
    render();

    try {
      const { response, payload } = await fetchWorkerJson("/api/stripe/checkout", {
        method: "POST",
        body: {
          reference: "frontend_pricing",
        },
      });

      if (!response.ok || !payload?.url) {
        throw new Error(payload?.message || "Worker checkout route did not return a Stripe URL.");
      }

      window.location.href = payload.url;
    } catch (error) {
      state.runtime.checkoutMessage = error.message || "Unable to start Worker checkout.";
      render();
    }
  };

  const handleTokenSave = async (event) => {
    event.preventDefault();
    const formData = new FormData(event.target);
    const token = String(formData.get("premium_token") || "").trim();
    writeStoredPremiumToken(token);
    state.runtime.premiumToken = token;
    state.runtime.accountMessage = token
      ? "Developer/test token saved locally. Open the premium page to test Worker-backed access."
      : "Premium token cleared.";
    if (page === "premium") {
      await loadProtectedPremiumPredictions();
    }
    render();
  };

  app.addEventListener("click", async (event) => {
    const checkoutTarget = event.target.closest("[data-action='worker-checkout']");
    if (checkoutTarget) {
      event.preventDefault();
      await startWorkerCheckout();
      return;
    }

    const clearTokenTarget = event.target.closest("[data-action='clear-premium-token']");
    if (clearTokenTarget) {
      event.preventDefault();
      writeStoredPremiumToken("");
      state.runtime.premiumToken = "";
      state.securePremiumPredictions = [];
      state.runtime.accountMessage = "Stored premium token cleared.";
      state.runtime.premiumFetchError = "";
      render();
    }
  });

  app.addEventListener("submit", async (event) => {
    if (event.target.id === "premium-token-form") {
      await handleTokenSave(event);
    }
  });

  const boot = async () => {
    app.innerHTML = `<div class="loading">Loading published board…</div>`;
    state.runtime.premiumToken = readStoredPremiumToken();

    try {
      const [summary, publicPredictions, premiumPredictions] = await Promise.all([
        fetchJson(`${DATA_ROOT}/publish_summary.json`),
        fetchJson(`${DATA_ROOT}/public_predictions.json`),
        premiumDemoMode ? fetchOptionalJson(`${DATA_ROOT}/premium_predictions.json`) : Promise.resolve([]),
      ]);
      const weeklyResults = await fetchOptionalJson(`${DATA_ROOT}/weekly_results.json`);
      state.summary = summary;
      state.publicPredictions = publicPredictions;
      state.premiumPredictions = Array.isArray(premiumPredictions) ? premiumPredictions : [];
      state.weeklyResults = weeklyResults;
      await loadProtectedPremiumPredictions();
      render();
    } catch (error) {
      renderError(error);
    }
  };

  boot();
})();
