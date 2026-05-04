(function () {
  const DATA_ROOT = "./public/data";
  const PREMIUM_TOKEN_STORAGE_KEY = "og_premium_token";
  const app = document.getElementById("app");
  const page = document.body.dataset.page || "home";
  const query = new URLSearchParams(window.location.search);
  const premiumDemoMode = query.get("demo") === "1";
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
        <p class="muted">${escapeHtml(row.short_reason || row.human_reason || "Model-backed selection.")}</p>
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
        <p class="hero-kicker">Odds Genius Live Board</p>
        <h1>Evidence-first football picks for bettors who want signal, not noise.</h1>
        <p>
          Model-backed selections, weekly proof, and a premium board that stays protected behind the Worker.
          Built to surface real edge, not tipster theatre.
        </p>
        <div class="hero-actions">
          <a class="button" href="./predictions.html">View live picks</a>
          <a class="ghost-button" href="./results.html">See proof</a>
        </div>
        <div class="proof-strip">
          ${proofTile("Live board", `${state.summary.public_predictions_count} free picks`, "Current public export")}
          ${proofTile("Premium board", `${state.summary.premium_predictions_count} cards`, "Protected Worker route ready")}
          ${proofTile(
            "Settled hit rate",
            state.weeklyResults?.overall_hit_rate == null ? "Pending" : `${Math.round(state.weeklyResults.overall_hit_rate * 100)}%`,
            state.weeklyResults ? `${state.weeklyResults.settled_picks}/${state.weeklyResults.total_picks} settled` : "Results layer ready"
          )}
          ${proofTile("Last publish", escapeHtml(state.summary.generated_at.slice(0, 10)), escapeHtml(sourceWindowLabel()))}
        </div>
      </div>
      <aside class="hero-side">
        <article class="sample-board">
          <div class="sample-board-head">
            <div>
              <span class="metric-label">Board snapshot</span>
              <strong>What a live card looks like</strong>
            </div>
            <span class="pill">Free preview</span>
          </div>
          <div class="sample-board-grid">
            ${state.publicPredictions.slice(0, 2).map((row) => `
              <article class="sample-row">
                <div class="sample-row-meta">
                  <strong>${escapeHtml(row.home_team)} vs ${escapeHtml(row.away_team)}</strong>
                  <span class="muted">${escapeHtml(row.league)} • ${escapeHtml(row.kickoff_time)}</span>
                  <div class="pill-row">
                    <span class="market-badge">${escapeHtml(row.market)}</span>
                    <span class="confidence-badge ${tierClass(row.confidence_tier)}">${escapeHtml(row.confidence_tier)}</span>
                  </div>
                </div>
                <div class="sample-row-side">
                  <span class="stat-chip">${escapeHtml(row.pick)}</span>
                  <span class="stat-chip">${escapeHtml(row.bookie_od ?? "N/A")}</span>
                  <span class="stat-chip">${escapeHtml(row.value_edge_display || "Edge ready")}</span>
                </div>
              </article>
            `).join("")}
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
        <h3>Why this converts</h3>
        <ul class="method-list">
          <li>Walk-forward proof is published separately from the picks page.</li>
          <li>The public board is useful, but the premium board is where the full signal density lives.</li>
          <li>Protected Worker delivery keeps premium access tied to live subscriber entitlement.</li>
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
              Fast-scan public cards built from the latest safe export. Confidence is rounded in public, while the
              premium board carries the deeper edge view.
            </p>
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
                : "Unlock the full deployable board, deeper value context, shortlist support, and richer explanations."
            }
          </p>
          <div class="cta-row">
            ${checkoutCta()}
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
          <h3>What unlocks</h3>
          <ul class="feature-list">
            <li>Full deployable board with ELITE and STANDARD cards.</li>
            <li>Model probability, bookie implied probability, and value edge.</li>
            <li>Human-readable premium reasons and shortlist context.</li>
            <li>Worker-protected access tied to live subscription state.</li>
          </ul>
        </article>
        <article class="panel">
          <h3>Internal review</h3>
          <ul class="feature-list">
            <li><a href="./premium.html?demo=1">Demo mode</a> remains available for internal preview only.</li>
            <li>No token means the page stays locked by default.</li>
            <li>Upgrade CTA routes to pricing and Worker checkout flow.</li>
          </ul>
        </article>
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
          <article class="panel">
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
          <article class="panel">
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
        </article>
        <aside class="hero-side">
          <div class="metric">
            <span class="metric-label">Window</span>
            <span class="metric-value">${escapeHtml(`${weekly.period_start} → ${weekly.period_end}`)}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Overall hit rate</span>
            <span class="metric-value">${escapeHtml(
              weekly.overall_hit_rate == null ? "Pending" : `${Math.round(weekly.overall_hit_rate * 100)}%`
            )}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Settled picks</span>
            <span class="metric-value">${escapeHtml(`${weekly.settled_picks}/${weekly.total_picks}`)}</span>
          </div>
        </aside>
      </section>

      <section class="section">
        <div class="results-highlight">
          ${statPanel("Total picks", weekly.total_picks)}
          ${statPanel("Settled picks", weekly.settled_picks)}
          ${statPanel("Pending picks", weekly.pending_picks)}
          ${statPanel(
            "Hit rate",
            weekly.overall_hit_rate == null ? "Pending" : `${Math.round(weekly.overall_hit_rate * 100)}%`,
            weekly.generated_at.slice(0, 10)
          )}
        </div>
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
        <article class="panel">
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
          <h1>Free board or founding member.</h1>
          <p>
            Start with the public board for free, or unlock the full premium board for the founding member price
            while the product is still in its early paid phase.
          </p>
          <div class="cta-row">
            ${checkoutCta()}
            <a class="ghost-button" href="./premium.html">Preview locked premium</a>
          </div>
        </article>
      <aside class="hero-side">
        <div class="metric">
          <span class="metric-label">Plan</span>
          <span class="metric-value">£20/mo</span>
        </div>
        <div class="metric">
          <span class="metric-label">Access</span>
          <span class="metric-value">Locked while subscribed</span>
        </div>
        <div class="metric">
          <span class="metric-label">Worker</span>
          <span class="metric-value">${workerConfigured() ? "Configured" : "Placeholder"}</span>
        </div>
      </aside>
    </section>

    <section class="section">
      ${renderNotice(state.runtime.checkoutMessage, state.runtime.checkoutMessage ? "warning" : "default")}
      <div class="pricing-grid">
        <article class="card pricing-card">
          <span class="pricing-tag">Free</span>
          <div class="pricing-price">£0</div>
          <p class="pricing-subcopy">A useful preview of the current board, intentionally limited.</p>
          <ul class="feature-list">
            <li>Top public board only.</li>
            <li>Rounded confidence and rounded edge display.</li>
            <li>Generic explanation layer.</li>
            <li>No premium shortlist or slip-role detail.</li>
          </ul>
        </article>
        <article class="card pricing-card featured">
          <span class="pricing-tag">Founding Member</span>
          <div class="pricing-price">£20<span class="pricing-price-note">/month</span></div>
          <p class="pricing-subcopy">Locked while subscribed. Built for the first paying cohort.</p>
          <ul class="feature-list">
            <li>Full deployable board with ELITE and STANDARD picks.</li>
            <li>Deeper edge context, richer explanations, and shortlist support.</li>
            <li>Protected Worker-backed delivery after verified access.</li>
            <li>Early founding member pricing while the product expands.</li>
          </ul>
          <div class="cta-row">
            ${checkoutCta()}
          </div>
        </article>
      </div>
      <p class="footer-note">
        ${
          workerConfigured()
            ? "The upgrade CTA now routes to the live Worker checkout flow."
            : "Static-only mode cannot provide secure subscriber enforcement."
        }
      </p>
    </section>
  `;

  const methodologyView = () => `
    <section class="section split">
      <article class="hero-main">
        <p class="hero-kicker">Methodology</p>
        <h1>Built on generated outputs, not frontend guesswork.</h1>
        <p>
          The website layer consumes exported JSON generated from the live routed board. It does not re-run model
          logic, routing, or deploy decisions in the browser.
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
          <p class="muted">Ingest, enrichment, routing, and slip logic remain in Python. The frontend only displays approved exports.</p>
        </article>
        <article class="panel">
          <h3>Safety posture</h3>
          <p class="muted">Allowlist-only publishing protects private thresholds, internal gates, raw features, and file paths.</p>
        </article>
        <article class="panel">
          <h3>Evidence posture</h3>
          <p class="muted">This layer is designed for walk-forward-tested product output rather than vague tipster language.</p>
        </article>
      </div>
    </section>
  `;

  const accountView = () => {
    const checkoutNotice =
      checkoutState === "success"
        ? "Checkout returned successfully. Subscription state still depends on webhook processing and verified login or token issuance."
        : checkoutState === "cancelled"
          ? "Checkout was cancelled. No premium access was granted."
          : "";

    return `
      <section class="section split">
        <article class="hero-main">
          <p class="hero-kicker">Account</p>
          <h1>Worker-aware account placeholder.</h1>
          <p>
            This page marks the future home for sign in, active membership status, and Stripe-backed subscription
            management. Developer/test token handling exists here only to help verify the Worker route locally.
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
            <span class="metric-label">Account mode</span>
            <span class="metric-value">Dev/Test only</span>
          </div>
        </aside>
      </section>

      <section class="section">
        ${renderNotice(checkoutNotice, checkoutState === "success" ? "success" : "warning")}
        ${renderNotice(state.runtime.accountMessage, state.runtime.accountMessage ? "warning" : "default")}
      </section>

      <section class="section split">
        <article class="panel">
          <h3>Developer/test token flow</h3>
          <p class="muted">
            This is not public auth. It exists only so a verified Worker token can be stored locally during
            controlled testing before magic-link or authenticated session issuance is built.
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
          <p class="footer-note">
            Final public access should use magic-link email verification or authenticated session issuance, not
            manual token pasting.
          </p>
        </article>
        <article class="panel">
          <h3>${accountIntent === "checkout" ? "Checkout handoff" : "Subscription management placeholder"}</h3>
          <ul class="feature-list">
            <li>Worker checkout route is used only when WORKER_API_BASE is configured.</li>
            <li>Webhook-backed subscriber state remains the authority for premium access.</li>
            <li>Customer Portal and real sign-in still need production wiring later.</li>
          </ul>
        </article>
      </section>
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
