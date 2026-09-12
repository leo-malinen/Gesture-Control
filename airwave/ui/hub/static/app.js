/* Airwave hub client.
 *
 * No framework and no build step: the page has one video element, one list,
 * and a handful of readouts, all driven by a single server-sent event stream.
 * Adding React here would add a toolchain to a project whose install story is
 * "pip install -r requirements.txt".
 *
 * The server is the only source of truth. This file never computes state, it
 * only renders what arrives — which is the same producer/consumer split the
 * Python side is built around.
 */

(() => {
  "use strict";

  const MAX_ROWS = 300;          // DOM rows kept; older ones are dropped
  const SUPPRESSED = new Set(["cooldown", "unarmed", "unbound", "paused"]);

  const el = {
    video: document.getElementById("video"),
    frame: document.getElementById("video-frame"),
    emptyTitle: document.getElementById("empty-title"),
    emptyBody: document.getElementById("empty-body"),
    emptyFix: document.getElementById("empty-fix"),
    gesture: document.getElementById("hud-gesture"),
    confidence: document.getElementById("hud-confidence"),
    stabilityFill: document.getElementById("stability-fill"),
    stabilityMeter: document.getElementById("stability-meter"),
    stabilityValue: document.getElementById("hud-stability-value"),
    armBlock: document.getElementById("hud-arm-block"),
    arm: document.getElementById("hud-arm"),
    armSub: document.getElementById("hud-arm-sub"),
    chipFps: document.getElementById("chip-fps"),
    chipMode: document.getElementById("chip-mode"),
    chipAudio: document.getElementById("chip-audio"),
    statFired: document.getElementById("stat-fired"),
    statSuppressed: document.getElementById("stat-suppressed"),
    statFps: document.getElementById("stat-fps"),
    statLatency: document.getElementById("stat-latency"),
    log: document.getElementById("log"),
    logEmpty: document.getElementById("log-empty"),
    pill: document.getElementById("connection-pill"),
    pillText: document.getElementById("connection-text"),
    clear: document.getElementById("clear-log"),
  };

  let filter = "all";
  let videoStarted = false;
  let rowCount = 0;

  /* ------------------------------------------------------------- video --- */

  function startVideo() {
    if (videoStarted) return;
    videoStarted = true;
    // Cache-buster: without it a reconnect can latch onto the finished stream.
    el.video.src = `/stream.mjpg?t=${Date.now()}`;
  }

  function stopVideo() {
    videoStarted = false;
    el.video.removeAttribute("src");
  }

  /* ------------------------------------------------------------ status --- */

  function setConnection(state, text) {
    el.pill.dataset.state = state;
    el.pillText.textContent = text;
  }

  function renderStatus(s) {
    const camera = s.camera || "unknown";

    if (camera === "live") {
      el.frame.dataset.state = "live";
      startVideo();
      setConnection("live", "Live");
    } else {
      el.frame.dataset.state = camera === "starting" ? "loading" : "empty";
      stopVideo();
      setConnection(camera === "starting" ? "connecting" : "error",
                    camera === "starting" ? "Starting" : "No camera");
      el.emptyTitle.textContent = s.camera_title || "Camera unavailable";
      el.emptyBody.textContent = s.camera_detail || "";
      if (s.camera_fix) {
        el.emptyFix.textContent = s.camera_fix;
        el.emptyFix.hidden = false;
      } else {
        el.emptyFix.hidden = true;
      }
    }

    // Gesture readout
    const label = s.gesture && s.gesture !== "none" ? s.gesture : "—";
    el.gesture.textContent = label;
    el.confidence.textContent =
      s.confidence != null && label !== "—" ? `confidence ${Number(s.confidence).toFixed(2)}` : "";

    // Stability meter
    const stability = Math.round((s.stability || 0) * 100);
    el.stabilityFill.style.width = `${stability}%`;
    el.stabilityFill.classList.toggle("is-complete", stability >= 100);
    el.stabilityValue.textContent = `${stability}%`;
    el.stabilityMeter.setAttribute("aria-valuenow", String(stability));

    // Arming
    if (s.arming_enabled) {
      el.armBlock.dataset.armed = String(Boolean(s.armed));
      el.arm.textContent = s.armed ? "Armed" : "Not armed";
      el.armSub.textContent = s.armed
        ? `${Number(s.arm_remaining || 0).toFixed(1)}s left`
        : `hold ${s.arming_gesture || "thumbs_up"}`;
    } else {
      el.armBlock.dataset.armed = "false";
      el.arm.textContent = "Off";
      el.armSub.textContent = "always live";
    }

    // Chips + stats
    const fps = s.fps ? Number(s.fps).toFixed(1) : null;
    el.chipFps.textContent = fps ? `${fps} fps` : "— fps";
    el.statFps.textContent = fps || "—";
    el.chipMode.textContent = s.mouse_active ? "Mouse" : "Gestures";
    el.chipAudio.textContent = s.audio_active ? "Audio on" : "Audio off";
    el.chipAudio.classList.toggle("chip-quiet", !s.audio_active);

    el.statFired.textContent = s.fired ?? 0;
    el.statSuppressed.textContent =
      (s.cooldown ?? 0) + (s.unarmed ?? 0) + (s.unbound ?? 0);
    el.statLatency.textContent =
      s.mean_latency_ms ? `${Math.round(s.mean_latency_ms)}ms` : "—";
  }

  /* --------------------------------------------------------------- log --- */

  function matchesFilter(outcome) {
    if (filter === "all") return true;
    if (filter === "fired") return outcome === "fired";
    return SUPPRESSED.has(outcome);
  }

  function buildRow(evt) {
    const li = document.createElement("li");
    li.className = "log-row";
    li.dataset.outcome = evt.outcome;
    li.dataset.kind = evt.kind;
    li.hidden = !matchesFilter(evt.outcome);

    const time = document.createElement("span");
    time.className = "log-time";
    time.textContent = evt.time;
    const ms = document.createElement("span");
    ms.className = "log-millis";
    ms.textContent = `.${evt.millis}`;
    time.appendChild(ms);
    // The full ISO instant lives in the tooltip so the row stays scannable.
    time.title = `${evt.ts} (UTC)`;

    const main = document.createElement("div");
    main.className = "log-main";

    const trigger = document.createElement("div");
    trigger.className = "log-trigger";
    const kind = document.createElement("span");
    kind.className = "log-kind";
    kind.textContent = `${evt.kind} · `;
    trigger.appendChild(kind);
    trigger.appendChild(document.createTextNode(evt.value || "—"));
    main.appendChild(trigger);

    const detailText = describe(evt);
    if (detailText) {
      const detail = document.createElement("div");
      detail.className = "log-detail";
      detail.textContent = detailText;
      main.appendChild(detail);
    }

    const outcome = document.createElement("span");
    outcome.className = "log-outcome";
    outcome.textContent = evt.outcome;

    li.append(time, main, outcome);
    return li;
  }

  function describe(evt) {
    const bits = [];
    if (evt.binding) bits.push(evt.binding);
    if (evt.detail) bits.push(evt.detail);
    if (evt.outcome === "fired" && evt.latency_ms) bits.push(`${Math.round(evt.latency_ms)}ms`);
    else if (evt.confidence != null) bits.push(`conf ${evt.confidence}`);
    return bits.join(" · ");
  }

  function addEvent(evt) {
    const atBottom =
      el.log.scrollTop + el.log.clientHeight >= el.log.scrollHeight - 24;

    el.log.appendChild(buildRow(evt));
    rowCount += 1;

    while (rowCount > MAX_ROWS && el.log.firstElementChild) {
      el.log.removeChild(el.log.firstElementChild);
      rowCount -= 1;
    }

    el.logEmpty.hidden = true;
    // Only autoscroll if the user has not scrolled up to read history.
    if (atBottom) el.log.scrollTop = el.log.scrollHeight;
  }

  function applyFilter(next) {
    filter = next;
    document.querySelectorAll(".filter-pill").forEach((pill) => {
      pill.classList.toggle("is-active", pill.dataset.filter === next);
    });
    let visible = 0;
    el.log.querySelectorAll(".log-row").forEach((row) => {
      const show = matchesFilter(row.dataset.outcome);
      row.hidden = !show;
      if (show) visible += 1;
    });
    el.logEmpty.hidden = visible > 0;
  }

  /* ------------------------------------------------------------ stream --- */

  let source = null;

  function connect() {
    if (source) source.close();
    source = new EventSource("/events");

    source.addEventListener("status", (e) => {
      try { renderStatus(JSON.parse(e.data)); } catch (err) { /* malformed frame */ }
    });

    source.addEventListener("event", (e) => {
      try { addEvent(JSON.parse(e.data)); } catch (err) { /* malformed frame */ }
    });

    source.addEventListener("bye", () => {
      setConnection("error", "Stopped");
      stopVideo();
      el.frame.dataset.state = "empty";
      el.emptyTitle.textContent = "Airwave stopped";
      el.emptyBody.textContent = "The app exited. Start it again to resume the feed.";
      el.emptyFix.hidden = true;
      source.close();
      source = null;
    });

    // EventSource reconnects on its own; this only reflects it in the UI.
    source.onerror = () => setConnection("connecting", "Reconnecting");
  }

  /* --------------------------------------------------------------- init -- */

  document.querySelectorAll(".filter-pill").forEach((pill) => {
    pill.addEventListener("click", () => applyFilter(pill.dataset.filter));
  });

  el.clear.addEventListener("click", () => {
    el.log.replaceChildren();
    rowCount = 0;
    el.logEmpty.hidden = false;
  });

  document.querySelectorAll(".nav-link.is-disabled").forEach((link) => {
    link.addEventListener("click", (e) => e.preventDefault());
  });

  el.video.addEventListener("error", () => {
    // The stream died (app restart). Let the status channel decide what to show.
    videoStarted = false;
  });

  connect();
})();
