from __future__ import annotations

CUSTOM_CSS = """
:root {
  --wa-bg: #f3ecdf;
  --wa-bg-2: #efe6d8;
  --wa-panel: rgba(255, 250, 244, 0.9);
  --wa-panel-strong: rgba(255, 248, 240, 0.97);
  --wa-panel-soft: rgba(253, 246, 238, 0.76);
  --wa-ink: #151515;
  --wa-muted: #5f5a53;
  --wa-accent: #ca5a32;
  --wa-accent-strong: #b54923;
  --wa-teal: #0a8c78;
  --wa-line: rgba(21, 21, 21, 0.08);
  --wa-line-strong: rgba(21, 21, 21, 0.14);
  --wa-shadow: 0 24px 70px rgba(48, 34, 18, 0.12);
}

body, .gradio-container {
  background:
    radial-gradient(circle at top left, rgba(202, 90, 50, 0.14), transparent 26%),
    radial-gradient(circle at top right, rgba(10, 140, 120, 0.1), transparent 18%),
    linear-gradient(180deg, var(--wa-bg) 0%, var(--wa-bg-2) 100%);
  color: var(--wa-ink);
  font-family: "IBM Plex Sans", sans-serif;
}

.gradio-container {
  max-width: 1660px !important;
  padding: 18px 18px 28px !important;
}

.gradio-container::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background-image:
    linear-gradient(to right, rgba(21, 21, 21, 0.03) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(21, 21, 21, 0.03) 1px, transparent 1px);
  background-size: 26px 26px;
  mask-image: linear-gradient(180deg, rgba(0,0,0,0.18), rgba(0,0,0,0.04));
}

.wa-hero {
  display: grid;
  grid-template-columns: minmax(0, 1.8fr) minmax(320px, 1fr);
  gap: 18px;
  margin-bottom: 18px;
  padding: 20px 22px;
  border-radius: 28px;
  border: 1px solid var(--wa-line);
  background:
    linear-gradient(135deg, rgba(255, 255, 255, 0.74), rgba(255, 255, 255, 0.34)),
    linear-gradient(120deg, rgba(202, 90, 50, 0.08), rgba(10, 140, 120, 0.08));
  box-shadow: var(--wa-shadow);
  position: relative;
  overflow: hidden;
}

.wa-hero::after {
  content: "";
  position: absolute;
  right: -30px;
  bottom: -70px;
  width: 220px;
  height: 220px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(202, 90, 50, 0.18), transparent 70%);
}

.wa-hero-copy {
  position: relative;
  z-index: 1;
}

.wa-hero-eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 7px 11px;
  margin-bottom: 14px;
  border-radius: 999px;
  border: 1px solid var(--wa-line);
  background: rgba(255, 255, 255, 0.62);
  font: 500 11px/1 "IBM Plex Mono", monospace;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.wa-hero h1 {
  margin: 0 0 8px;
  font: 700 40px/1.02 "Space Grotesk", sans-serif;
  letter-spacing: -0.04em;
}

.wa-hero p {
  margin: 0;
  max-width: 760px;
  color: var(--wa-muted);
  font-size: 15px;
  line-height: 1.62;
}

.wa-hero-context {
  position: relative;
  z-index: 1;
  display: grid;
  gap: 12px;
}

.wa-hero-focus {
  padding: 16px 18px;
  border-radius: 22px;
  border: 1px solid var(--wa-line);
  background: rgba(255, 255, 255, 0.66);
}

.wa-hero-focus-label {
  margin-bottom: 8px;
  font: 500 11px/1 "IBM Plex Mono", monospace;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--wa-muted);
}

.wa-hero-focus strong {
  display: block;
  font: 700 26px/1.08 "Space Grotesk", sans-serif;
}

.wa-hero-focus span {
  display: block;
  margin-top: 8px;
  color: var(--wa-muted);
  font-size: 13px;
  line-height: 1.5;
}

.wa-metric-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.wa-metric {
  padding: 13px 14px;
  border-radius: 18px;
  border: 1px solid var(--wa-line);
  background: rgba(255, 255, 255, 0.62);
}

.wa-metric strong {
  display: block;
  font: 700 24px/1 "Space Grotesk", sans-serif;
}

.wa-metric span {
  display: block;
  margin-top: 6px;
  color: var(--wa-muted);
  font-size: 12px;
}

.wa-main-grid {
  align-items: start;
  gap: 18px;
}

.wa-left-rail,
.wa-right-rail {
  gap: 18px;
}

.wa-left-rail,
.wa-right-rail {
  position: sticky !important;
  top: 18px;
  align-self: start;
}

.wa-stage-col {
  min-width: 0 !important;
}

.wa-panel-block {
  border-radius: 24px !important;
  border: 1px solid var(--wa-line) !important;
  background: var(--wa-panel) !important;
  box-shadow: 0 14px 40px rgba(48, 34, 18, 0.08) !important;
  padding: 18px !important;
  backdrop-filter: blur(14px);
}

.wa-panel-title {
  margin: 0 0 8px;
  font: 700 18px/1.08 "Space Grotesk", sans-serif;
}

.wa-panel-copy {
  margin: 0;
  color: var(--wa-muted);
  font-size: 13px;
  line-height: 1.58;
}

.wa-profile {
  padding: 16px 18px;
  border-radius: 20px;
  border: 1px solid var(--wa-line);
  background: var(--wa-panel-strong);
}

.wa-profile-eyebrow {
  margin-bottom: 8px;
  font: 500 11px/1 "IBM Plex Mono", monospace;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--wa-muted);
}

.wa-profile h3 {
  margin: 0 0 8px;
  font: 700 24px/1.08 "Space Grotesk", sans-serif;
}

.wa-profile p {
  margin: 0 0 14px;
  color: var(--wa-muted);
  font-size: 14px;
  line-height: 1.58;
}

.wa-pillrow,
.wa-summary-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.wa-pill,
.wa-summary-pill {
  display: inline-flex;
  align-items: center;
  padding: 7px 10px;
  border-radius: 999px;
  border: 1px solid var(--wa-line);
  background: rgba(21, 21, 21, 0.04);
  font: 500 11px/1 "IBM Plex Mono", monospace;
}

.wa-runnote {
  margin-top: 14px;
  padding: 12px 13px;
  border-radius: 16px;
  border: 1px solid rgba(202, 90, 50, 0.16);
  background: linear-gradient(90deg, rgba(202, 90, 50, 0.1), rgba(10, 140, 120, 0.08));
  color: #3d3a35;
  font-size: 13px;
  line-height: 1.55;
}

.wa-status {
  padding: 13px 14px;
  border-radius: 18px;
  background: #151515;
  color: #f7f0e7;
  font: 500 12px/1.65 "IBM Plex Mono", monospace;
  white-space: pre-wrap;
}

.wa-summary-card {
  margin-top: 12px;
  padding: 15px 16px;
  border-radius: 18px;
  border: 1px solid var(--wa-line);
  background: var(--wa-panel-soft);
}

.wa-summary-card.is-danger {
  border-color: rgba(177, 63, 45, 0.24);
  background: rgba(255, 236, 232, 0.92);
}

.wa-summary-card h4 {
  margin: 0 0 5px;
  font: 700 17px/1.08 "Space Grotesk", sans-serif;
}

.wa-summary-subtitle {
  margin: 0 0 12px;
  color: var(--wa-muted);
  font-size: 13px;
  line-height: 1.55;
}

.wa-summary-lines {
  display: grid;
  gap: 6px;
  margin-top: 12px;
}

.wa-summary-lines div {
  color: #3f3b35;
  font-size: 13px;
  line-height: 1.5;
}

.wa-run-row,
.wa-preset-row {
  gap: 8px;
}

.wa-run-row button,
.wa-preset-row button,
.wa-live-grid button {
  min-height: 42px !important;
  border-radius: 16px !important;
  font-weight: 600 !important;
  border: 1px solid var(--wa-line-strong) !important;
  box-shadow: none !important;
}

.wa-run-primary button,
.wa-run-primary {
  background: linear-gradient(135deg, var(--wa-accent), var(--wa-accent-strong)) !important;
  color: #fff !important;
  border-color: transparent !important;
}

.wa-run-secondary button,
.wa-run-secondary {
  background: linear-gradient(135deg, rgba(10, 140, 120, 0.14), rgba(10, 140, 120, 0.22)) !important;
  color: #0f5a4f !important;
}

.wa-run-muted button,
.wa-run-muted {
  background: rgba(255, 255, 255, 0.58) !important;
  color: #36322d !important;
}

.wa-preview-panel {
  min-width: 0;
}

.wa-preview-panel .tabs {
  border: 0 !important;
}

.gradio-container .tab-nav {
  padding: 4px;
  border: 1px solid var(--wa-line) !important;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.54);
}

.gradio-container .tab-nav button {
  min-height: 31px !important;
  border-radius: 999px !important;
  color: var(--wa-muted) !important;
  font-size: 11px !important;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.gradio-container .tab-nav button.selected {
  background: rgba(21, 21, 21, 0.08) !important;
  color: var(--wa-ink) !important;
}

#wa-main-preview-video,
#wa-main-preview-image,
.wa-preview-panel .tabitem {
  min-height: 520px;
}

.wa-live-dock {
  margin-top: 18px;
}

.wa-live-caption {
  margin: 0 0 12px;
  color: var(--wa-muted);
  font-size: 13px;
  line-height: 1.58;
}

.wa-live-grid {
  gap: 8px;
}

.wa-live-group-title {
  margin: 0 0 8px;
  color: var(--wa-muted);
  font: 500 11px/1 "IBM Plex Mono", monospace;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.wa-live-grid button {
  background: rgba(255, 255, 255, 0.6) !important;
  color: #2f2c27 !important;
}

.wa-live-grid button:hover {
  border-color: rgba(202, 90, 50, 0.24) !important;
}

.wa-keyflash {
  transform: translateY(-1px);
  box-shadow: 0 0 0 2px rgba(202, 90, 50, 0.18) !important;
}

.wa-dataframe table {
  border-radius: 16px !important;
  overflow: hidden !important;
}

.wa-dataframe table th {
  background: rgba(21, 21, 21, 0.05) !important;
}

.wa-dataframe table td,
.wa-dataframe table th {
  border-color: var(--wa-line) !important;
}

.wa-dataframe table td {
  background: rgba(255, 255, 255, 0.46) !important;
}

.wa-download-hint {
  padding: 10px 12px;
  border-left: 3px solid var(--wa-teal);
  border-radius: 10px;
  background: rgba(10, 140, 120, 0.08);
  color: var(--wa-muted);
  font-size: 12px;
  line-height: 1.5;
}

.gradio-container .accordion {
  border: 1px solid var(--wa-line) !important;
  border-radius: 18px !important;
  background: rgba(255, 255, 255, 0.46) !important;
}

.gradio-container .accordion summary {
  font-weight: 600;
}

.gradio-container label span {
  color: var(--wa-muted) !important;
}

.gradio-container textarea,
.gradio-container input,
.gradio-container select {
  background: rgba(255, 255, 255, 0.74) !important;
  color: var(--wa-ink) !important;
  border-color: var(--wa-line-strong) !important;
}

.gradio-container textarea::placeholder,
.gradio-container input::placeholder {
  color: #888178 !important;
}

.gradio-container .cm-editor,
.gradio-container .cm-gutters,
.gradio-container .upload-container,
.gradio-container .image-container,
.gradio-container .video-container,
.gradio-container .empty {
  background: rgba(255, 255, 255, 0.56) !important;
  border-color: var(--wa-line) !important;
}

.gradio-container .block,
.gradio-container .form {
  gap: 12px;
}

.wa-muted {
  color: var(--wa-muted);
}

@media (max-width: 1320px) {
  .wa-left-rail,
  .wa-right-rail {
    position: static !important;
  }
}

@media (max-width: 1080px) {
  .wa-hero {
    grid-template-columns: 1fr;
  }

  .wa-main-grid {
    gap: 14px;
  }

  #wa-main-preview-video,
  #wa-main-preview-image,
  .wa-preview-panel .tabitem {
    min-height: 440px;
  }
}

@media (max-width: 760px) {
  .gradio-container {
    padding: 14px 10px 26px !important;
  }

  .wa-hero {
    padding: 16px;
    border-radius: 22px;
  }

  .wa-hero h1 {
    font-size: 30px;
  }

  .wa-metric-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  #wa-main-preview-video,
  #wa-main-preview-image,
  .wa-preview-panel .tabitem {
    min-height: 320px;
  }

  .wa-panel-block {
    padding: 14px !important;
  }
}

/* Dark player override baseline */
:root {
  --wa-bg: #1d1e22;
  --wa-bg-2: #17181c;
  --wa-panel: rgba(43, 46, 53, 0.92);
  --wa-panel-strong: rgba(51, 54, 62, 0.96);
  --wa-ink: #f3f4f6;
  --wa-muted: #9ca2ad;
  --wa-accent: #ff7b31;
  --wa-accent-strong: #ff611f;
  --wa-line: rgba(255, 255, 255, 0.08);
  --wa-line-strong: rgba(255, 255, 255, 0.12);
}

body, .gradio-container {
  background:
    radial-gradient(circle at top center, rgba(255, 123, 49, 0.09), transparent 18%),
    linear-gradient(180deg, var(--wa-bg) 0%, var(--wa-bg-2) 100%) !important;
  color: var(--wa-ink);
}

html, body {
  height: 100%;
}

.gradio-container {
  max-width: none !important;
  padding: 10px 12px 18px !important;
}

.gradio-container::before,
.wa-hero {
  display: none !important;
}

.wa-studio-header {
  display: none !important;
}

.wa-site-nav-shell {
  margin-bottom: 18px;
}

.wa-site-nav {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 4px 8px 0;
  color: #d8dbe0;
  font: 400 10px/1 "Press Start 2P", monospace;
  letter-spacing: 0.08em;
}

.wa-site-nav-right {
  display: flex;
  align-items: center;
  gap: 14px;
}

.wa-site-icon {
  color: #9aa0aa;
  font-size: 12px;
}

.wa-site-chip {
  padding: 9px 12px;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  background: rgba(255, 255, 255, 0.04);
  font-size: 8px;
}

.wa-main-grid {
  position: relative;
  display: block !important;
  min-height: calc(100vh - 18px);
  margin: 0 !important;
  overflow: visible;
}

.wa-left-rail,
.wa-right-rail {
  position: absolute !important;
  top: 50%;
  width: 224px !important;
  min-width: 224px !important;
  max-width: 224px !important;
  flex: 0 0 224px !important;
  z-index: 20;
  opacity: 0.48;
  transition: transform 180ms ease, opacity 180ms ease;
}

.wa-left-rail {
  left: 10px;
  transform: translate(-262px, -50%);
}

.wa-right-rail {
  right: 10px;
  transform: translate(262px, -50%);
}

.wa-left-rail:hover,
.wa-left-rail:focus-within {
  transform: translate(0, -50%);
  opacity: 1;
}

.wa-right-rail:hover,
.wa-right-rail:focus-within {
  transform: translate(0, -50%);
  opacity: 1;
}

.wa-left-rail::after,
.wa-right-rail::before {
  position: absolute;
  top: 50%;
  width: 10px;
  height: 84px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 999px;
  background: rgba(49, 52, 60, 0.52);
  display: flex;
  align-items: center;
  justify-content: center;
  color: transparent;
}

.wa-left-rail::after {
  content: "";
  right: -7px;
  transform: translateY(-50%);
}

.wa-right-rail::before {
  content: "";
  left: -7px;
  transform: translateY(-50%);
}

.wa-stage-col {
  width: 100% !important;
  max-width: none !important;
  padding: clamp(18px, 3vw, 30px) clamp(120px, 12vw, 170px) clamp(110px, 10vw, 140px) !important;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.wa-stage-col > .block,
.wa-stage-col > .form {
  width: min(660px, calc(100vw - clamp(260px, 28vw, 420px))) !important;
  max-width: min(660px, calc(100vw - clamp(260px, 28vw, 420px))) !important;
  margin: 0 auto !important;
}

.wa-left-rail .wa-panel-block,
.wa-right-rail .wa-panel-block {
  background: rgba(42, 45, 52, 0.96) !important;
  max-height: min(560px, calc(100vh - 120px));
  overflow: auto;
  border-radius: 18px !important;
  box-shadow: 0 16px 34px rgba(0, 0, 0, 0.2) !important;
}

.wa-panel-title {
  color: #d4d8df;
  font: 700 12px/1 "IBM Plex Mono", monospace;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.wa-panel-copy,
.wa-download-hint,
.wa-live-caption,
.wa-live-group-title {
  display: none !important;
}

.wa-preview-panel {
  position: relative;
  min-height: auto;
  max-height: none;
  padding: clamp(16px, 1.8vw, 18px) clamp(12px, 1.4vw, 14px) 142px !important;
  border-radius: 28px !important;
  border: 1px solid rgba(255, 255, 255, 0.06) !important;
  background: rgba(39, 41, 47, 0.96) !important;
  overflow: visible !important;
  box-shadow:
    inset 0 0 0 1px rgba(255, 255, 255, 0.03),
    0 24px 72px rgba(0, 0, 0, 0.3) !important;
}

.wa-preview-panel > .wa-panel-title,
.wa-preview-panel > .wa-panel-copy,
.wa-preview-panel .wa-summary-card {
  display: none !important;
}

.wa-status-host {
  margin: 0 !important;
}

.wa-run-overview {
  display: none !important;
}

.wa-player-chrome {
  margin: 0 !important;
  display: none !important;
}

.wa-player-topbar,
.wa-player-footer {
  position: absolute;
  left: 18px;
  right: 18px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  color: #8f95a0;
  font: 400 7px/1 "Press Start 2P", monospace;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  pointer-events: none;
  z-index: 8;
}

.wa-player-topbar {
  display: none !important;
}

.wa-player-footer {
  bottom: 104px;
}

.wa-player-topbar-left,
.wa-player-topbar-center,
.wa-player-topbar-right,
.wa-player-footer-left,
.wa-player-footer-center,
.wa-player-footer-right {
  min-width: 72px;
}

.wa-player-topbar-center,
.wa-player-footer-center {
  text-align: center;
}

.wa-player-topbar-right,
.wa-player-footer-right {
  text-align: right;
}

.wa-player-dot {
  display: inline-block;
  width: 5px;
  height: 5px;
  margin-right: 6px;
  border-radius: 999px;
  background: var(--wa-accent);
  box-shadow: 0 0 12px rgba(255, 123, 49, 0.55);
}

.wa-player-brand {
  color: #d6d9de;
}

.wa-preview-panel .wa-status {
  width: fit-content;
  max-width: 180px;
  margin: 4px auto 0;
  padding: 7px 12px;
  min-height: 18px;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(49, 52, 58, 0.52);
  color: #a3a9b3;
  font: 400 6px/1.35 "Press Start 2P", monospace;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

#wa-main-preview-video,
#wa-main-preview-image,
.wa-preview-panel .tabitem {
  min-height: auto !important;
}

#wa-main-preview-video,
#wa-main-preview-image,
.wa-preview-panel .model3D {
  width: 100% !important;
  aspect-ratio: 832 / 480;
  min-height: auto !important;
}

.wa-preview-panel .empty,
.wa-preview-panel .image-container,
.wa-preview-panel .video-container,
.wa-preview-panel .model3D,
.wa-preview-panel .block {
  background: #111318 !important;
  border-color: rgba(255, 255, 255, 0.06) !important;
}

.wa-preview-panel .tabitem {
  background: transparent !important;
}

#wa-main-preview-video video,
#wa-main-preview-image img,
.wa-preview-panel .image-container,
.wa-preview-panel .video-container,
.wa-preview-panel .model3D,
.wa-preview-panel model-viewer {
  width: 100% !important;
  height: 100% !important;
  border-radius: 20px !important;
  background: #06080d !important;
  object-fit: contain !important;
}

.wa-preview-panel .tabs [role="tablist"] {
  display: flex !important;
  flex-wrap: nowrap !important;
  align-items: center !important;
  gap: 6px !important;
  overflow-x: auto !important;
  overflow-y: hidden !important;
  width: calc(100% - 12px);
  max-width: 100%;
  margin: -2px auto 12px !important;
  padding: 8px 6px !important;
  scrollbar-width: thin;
}

.wa-preview-panel .tabs [role="tablist"] button {
  flex: 0 0 auto !important;
  min-height: 34px !important;
  padding: 0 12px !important;
  margin: 0 !important;
  border-radius: 999px !important;
  border: 1px solid rgba(255, 255, 255, 0.22) !important;
  background: rgba(55, 58, 64, 0.95) !important;
  color: rgba(232, 234, 240, 0.95) !important;
  font: 600 11px / 1.25 "IBM Plex Mono", monospace !important;
  letter-spacing: 0.02em !important;
  text-transform: none !important;
  white-space: nowrap !important;
  outline: none !important;
  box-shadow: none !important;
}

.wa-preview-panel .tabs [role="tablist"] button.selected,
.wa-preview-panel .tabs [role="tablist"] button[aria-selected="true"] {
  border-color: rgba(251, 146, 60, 0.85) !important;
  background: rgba(251, 146, 60, 0.2) !important;
  color: #fff9f3 !important;
}

.wa-control-dock {
  display: flex !important;
  width: fit-content !important;
  position: absolute;
  left: 50%;
  bottom: 62px;
  transform: translateX(-50%);
  gap: 6px;
  justify-content: center;
  z-index: 7;
}

.wa-control-dock > *,
.wa-control-dock .wrap,
.wa-control-dock button {
  flex: 0 0 auto !important;
  width: auto !important;
}

.wa-control-dock button {
  min-height: 22px !important;
  min-width: 36px !important;
  border-radius: 999px !important;
  font: 400 6px/1 "Press Start 2P", monospace !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  padding: 0 10px !important;
  box-shadow: none !important;
}

.wa-action-primary {
  background: linear-gradient(180deg, var(--wa-accent), var(--wa-accent-strong)) !important;
  color: #ffffff !important;
  border-color: transparent !important;
}

.wa-action-reset {
  background: rgba(67, 70, 78, 0.98) !important;
  color: #e0e3e8 !important;
}

.wa-action-start {
  min-width: 48px !important;
}

.wa-action-hidden {
  display: none !important;
}

.wa-world-tray {
  display: flex;
  align-items: center;
  gap: 10px;
  position: absolute;
  left: 50%;
  bottom: 14px;
  transform: translateX(-50%);
  width: fit-content;
  min-height: 64px;
  padding: 10px 12px;
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.06);
  background: rgba(43, 46, 54, 0.62);
}

.wa-tray-item {
  flex: 0 0 auto;
  width: 52px;
  height: 52px;
  padding: 3px;
  border-radius: 12px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(255, 255, 255, 0.03);
  color: #e7eaef;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 4px;
  cursor: pointer;
}

.wa-tray-item.is-active {
  border-color: rgba(255, 255, 255, 0.18);
  box-shadow: inset 0 0 0 2px rgba(255, 255, 255, 0.08);
}

.wa-tray-thumb {
  display: block;
  width: 44px;
  height: 34px;
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: linear-gradient(180deg, rgba(113, 118, 129, 0.56), rgba(66, 70, 79, 0.86));
}

.wa-tray-thumb-video {
  background: linear-gradient(135deg, #d8dce3, #8c95a8);
}

.wa-tray-thumb-image {
  background: linear-gradient(135deg, #d6c4a6, #8b765a);
}

.wa-tray-thumb-3d {
  background: linear-gradient(135deg, #8ab5c9, #4f6477);
}

.wa-tray-thumb-gallery {
  background: linear-gradient(135deg, #98a76b, #47553a);
}

.wa-tray-thumb-embodied {
  background: linear-gradient(135deg, #a78bfa, #2563eb);
}

.wa-tray-thumb-artifacts {
  background: linear-gradient(135deg, #7e7f96, #424454);
}

.wa-tray-label {
  display: none;
}

.wa-live-dock {
  position: absolute !important;
  left: -9999px !important;
  top: -9999px !important;
  width: 1px !important;
  height: 1px !important;
  overflow: hidden !important;
}

.wa-dom-bridge {
  position: absolute !important;
  left: -9999px !important;
  top: -9999px !important;
  width: 1px !important;
  height: 1px !important;
  overflow: hidden !important;
}

.wa-keyflash {
  box-shadow: 0 0 0 2px rgba(255, 123, 49, 0.28) !important;
}

.wa-progress-card {
  gap: 10px;
}

.wa-progress-shell {
  width: 100%;
  height: 8px;
  overflow: hidden;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(63, 63, 70, 0.42);
}

.wa-progress-bar {
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, rgba(59, 130, 246, 0.92), rgba(96, 165, 250, 0.92));
}

.wa-progress-bar.is-indeterminate {
  width: 34%;
  animation: wa-progress-slide 1.35s ease-in-out infinite;
}

@keyframes wa-progress-slide {
  0% { transform: translateX(-110%); }
  100% { transform: translateX(320%); }
}

.wa-stick-shell {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
}

.wa-floating-stick {
  position: absolute;
  bottom: 28px;
  z-index: 10;
  width: 82px;
  padding: 6px;
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(43, 45, 53, 0.55);
}

.wa-floating-stick-left {
  left: 14px !important;
  right: auto !important;
}

.wa-floating-stick-right {
  right: 14px !important;
  left: auto !important;
}

.wa-stick-label {
  color: #8e949f;
  font: 400 5px/1 "Press Start 2P", monospace;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.wa-stick {
  position: relative;
  width: 68px;
  height: 68px;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: radial-gradient(circle at 28% 28%, rgba(78, 82, 91, 0.96), rgba(43, 45, 52, 0.98));
  touch-action: none;
}

.wa-stick-ring {
  position: absolute;
  inset: 8px;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.wa-stick-thumb {
  position: absolute;
  top: 50%;
  left: 50%;
  width: 24px;
  height: 24px;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.12);
  background: linear-gradient(180deg, rgba(127, 132, 142, 0.96), rgba(82, 86, 96, 0.96));
  transform: translate(-50%, -50%);
}

.wa-stick-shell.is-key-active .wa-stick {
  border-color: rgba(147, 197, 253, 0.48);
  box-shadow: 0 0 0 1px rgba(96, 165, 250, 0.18), 0 0 28px rgba(59, 130, 246, 0.18);
}

.wa-stick-shell.is-key-active .wa-stick-thumb {
  background: linear-gradient(180deg, rgba(191, 219, 254, 0.98), rgba(96, 165, 250, 0.98));
  border-color: rgba(219, 234, 254, 0.72);
}

@media (max-width: 1200px) {
  .wa-left-rail,
  .wa-right-rail {
    position: static !important;
    width: auto !important;
    max-width: none !important;
    transform: none !important;
    opacity: 1 !important;
  }

  .wa-left-rail::after,
  .wa-right-rail::before {
    display: none;
  }

  .wa-stage-col {
    padding: 0 0 88px !important;
  }

  .wa-stage-col > .block,
  .wa-stage-col > .form {
    width: 100% !important;
    max-width: 100% !important;
  }

  .wa-floating-stick {
    display: none;
  }

  .wa-world-tray {
    overflow-x: auto;
  }
}

/* High-fidelity inspatio override */
:root {
  --wa-bg: #202226;
  --wa-bg-2: #191b20;
  --wa-shell: rgba(39, 41, 47, 0.9);
  --wa-panel: rgba(43, 46, 53, 0.96);
  --wa-panel-strong: rgba(51, 54, 62, 0.98);
  --wa-stage: #09090b;
  --wa-tray: rgba(24, 24, 27, 0.4);
  --wa-ink: #f3f4f6;
  --wa-muted: #9ca3af;
  --wa-muted-soft: #737a84;
  --wa-line: rgba(255, 255, 255, 0.08);
  --wa-line-strong: rgba(255, 255, 255, 0.14);
  --wa-status-dot: #fcd34d;
  --wa-status-color: #fcd34d;
  --wa-scene-thumb-1: linear-gradient(135deg, rgba(36, 64, 104, 0.95), rgba(47, 124, 113, 0.88));
  --wa-scene-thumb-2: linear-gradient(135deg, rgba(80, 56, 118, 0.95), rgba(42, 116, 143, 0.86));
  --wa-scene-thumb-3: linear-gradient(135deg, rgba(83, 78, 65, 0.95), rgba(30, 115, 106, 0.84));
  --wa-scene-thumb-4: linear-gradient(135deg, rgba(52, 70, 91, 0.95), rgba(125, 79, 62, 0.84));
  --wa-scene-thumb-5: linear-gradient(135deg, rgba(65, 88, 78, 0.95), rgba(87, 72, 122, 0.84));
  --wa-scene-thumb-6: linear-gradient(135deg, rgba(71, 60, 88, 0.95), rgba(118, 90, 58, 0.84));
  --wa-scene-thumb-7: linear-gradient(135deg, rgba(44, 85, 111, 0.95), rgba(98, 104, 61, 0.84));
  --wa-scene-thumb-8: linear-gradient(135deg, rgba(92, 63, 72, 0.95), rgba(51, 104, 123, 0.84));
}

html,
body {
  min-height: 100%;
  background: var(--wa-bg-2) !important;
  overflow: hidden;
}

body,
.gradio-container {
  background:
    radial-gradient(circle at top center, rgba(148, 163, 184, 0.08), transparent 18%),
    linear-gradient(180deg, var(--wa-bg) 0%, var(--wa-bg-2) 100%) !important;
  color: var(--wa-ink) !important;
}

.gradio-container {
  min-height: 100vh !important;
  max-width: none !important;
  padding: 0 !important;
  overflow: hidden !important;
}

.gradio-container::before,
.wa-hero,
.wa-studio-header {
  display: none !important;
}

.gradio-container footer,
.gradio-container .built-with-gradio {
  display: none !important;
}

.wa-notice-shell {
  margin: 0;
}

.wa-notice-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 8px 16px;
  border-bottom: 1px solid rgba(63, 63, 70, 0.35);
  background: rgba(39, 41, 47, 0.82);
  color: #d4d4d8;
  font: 400 11px/1.4 Menlo, "IBM Plex Mono", monospace;
}

.wa-notice-bar.is-hidden {
  display: none;
}

.wa-notice-main {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.wa-notice-icon {
  color: #8d95a1;
  flex: 0 0 auto;
}

.wa-notice-copy {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.wa-notice-close {
  flex: 0 0 auto;
  width: 24px;
  height: 24px;
  border-radius: 999px;
  border: 1px solid transparent;
  background: transparent;
  color: #8d95a1;
  cursor: pointer;
  transition: background-color 160ms ease, color 160ms ease;
}

.wa-notice-close:hover {
  background: rgba(255, 255, 255, 0.08);
  color: #e5e7eb;
}

.wa-site-nav-shell {
  margin-bottom: 0;
}

.wa-site-nav {
  width: 100%;
  margin: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 16px 8px;
  color: #e5e7eb;
}

.wa-site-brand {
  display: flex;
  align-items: center;
  gap: 8px;
  opacity: 0.9;
  font: 700 11px/1 Menlo, "IBM Plex Mono", monospace;
  letter-spacing: 0.2em;
  text-transform: uppercase;
}

.wa-site-brand-mark {
  color: #cfd5dd;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.wa-site-brand-mark svg {
  display: block;
  width: 16px;
  height: 16px;
}

.wa-site-nav-left {
  white-space: nowrap;
}

.wa-site-nav-right {
  display: flex;
  align-items: center;
  gap: 8px;
}

.wa-site-nav-icon {
  width: 36px;
  height: 36px;
  border-radius: 999px;
  border: 0;
  background: transparent;
  color: #a1a1aa;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: transform 150ms ease, background-color 150ms ease, color 150ms ease;
}

.wa-site-nav-icon:hover {
  transform: scale(1.08);
  background: rgba(63, 63, 70, 0.5);
  color: #f4f4f5;
}

.wa-site-nav-icon svg {
  width: 20px;
  height: 20px;
}

.wa-site-chip {
  min-width: 0;
  height: 28px;
  padding: 0 12px;
  border-radius: 999px;
  border: 1px solid rgba(82, 82, 92, 0.5);
  background: rgba(63, 63, 70, 0.5);
  color: #d4d4d8;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  white-space: nowrap;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.1);
  font: 400 10px/1 Menlo, "IBM Plex Mono", monospace;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.wa-main-grid {
  position: relative;
  display: flex !important;
  align-items: center;
  justify-content: center;
  min-height: calc(100vh - 82px);
  margin: 0 !important;
  padding: 0 12px 14px !important;
  overflow: visible;
}

.wa-left-rail,
.wa-right-rail {
  position: absolute !important;
  top: 50%;
  width: 320px !important;
  min-width: 320px !important;
  max-width: 320px !important;
  flex: 0 0 320px !important;
  z-index: 24;
  opacity: 0.04;
  will-change: transform, opacity;
  transition: transform 180ms ease, opacity 180ms ease, filter 180ms ease;
}

.wa-left-rail {
  left: 0;
  transform: translate(calc(-100% + 14px), -50%);
}

.wa-right-rail {
  right: 0;
  transform: translate(calc(100% - 14px), -50%);
}

.wa-left-rail:hover,
.wa-left-rail:focus-within,
.wa-right-rail:hover,
.wa-right-rail:focus-within {
  transform: translate(0, -50%);
  opacity: 1;
  filter: none;
}

.wa-left-rail::after,
.wa-right-rail::before {
  content: "";
  position: absolute;
  top: 50%;
  width: 12px;
  height: 110px;
  border: 1px solid rgba(82, 82, 92, 0.45);
  border-radius: 999px;
  background: rgba(63, 63, 70, 0.7);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
}

.wa-left-rail::after {
  right: -11px;
  transform: translateY(-50%);
}

.wa-right-rail::before {
  left: -11px;
  transform: translateY(-50%);
}

.wa-stage-col {
  width: 100% !important;
  max-width: none !important;
  padding: 0 !important;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.wa-stage-col > .block,
.wa-stage-col > .form {
  width: min(960px, calc(100vw - 36px)) !important;
  max-width: min(960px, calc(100vw - 36px)) !important;
  margin: 0 auto !important;
}

.wa-panel-block {
  background: var(--wa-panel) !important;
  border-radius: 20px !important;
  border: 1px solid var(--wa-line) !important;
  box-shadow: none !important;
  padding: 14px !important;
}

.wa-left-rail .wa-panel-block,
.wa-right-rail .wa-panel-block {
  background: rgba(43, 46, 53, 0.98) !important;
  max-height: min(620px, calc(100vh - 130px));
  overflow: auto;
  box-shadow: 0 16px 34px rgba(0, 0, 0, 0.28) !important;
  backdrop-filter: blur(12px);
}

.wa-panel-title {
  color: #d7dce4;
  font: 700 11px/1 "IBM Plex Mono", monospace;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.wa-panel-copy,
.wa-download-hint,
.wa-live-caption,
.wa-live-group-title,
.wa-run-overview {
  display: none !important;
}

.wa-profile {
  padding: 14px 14px 12px;
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(63, 67, 76, 0.36);
}

.wa-profile-eyebrow {
  color: #9ca3af;
  font: 500 10px/1 "IBM Plex Mono", monospace;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.wa-profile h3 {
  color: #f3f4f6;
  font: 700 20px/1.02 "Space Grotesk", sans-serif;
}

.wa-profile p {
  color: #b1b7c1;
}

.wa-pill,
.wa-summary-pill {
  border-color: rgba(255, 255, 255, 0.08);
  background: rgba(255, 255, 255, 0.04);
  color: #d7dce4;
  font: 500 10px/1 "IBM Plex Mono", monospace;
}

.wa-runnote {
  border-color: rgba(255, 255, 255, 0.08);
  background: rgba(255, 255, 255, 0.04);
  color: #c9d0d9;
}

.wa-preview-panel {
  position: relative;
  width: min(960px, calc(100vw - 36px)) !important;
  max-width: 960px !important;
  min-height: 688px;
  max-height: min(88vh, 760px);
  padding: 18px 20px 26px !important;
  border-radius: 48px !important;
  border: 1px solid rgba(63, 63, 70, 0.86) !important;
  background: var(--wa-shell) !important;
  overflow: visible !important;
  box-shadow: 0 20px 60px -12px rgba(0, 0, 0, 0.5) !important;
  --wa-status-dot: #fcd34d;
  --wa-status-color: #fcd34d;
  margin: 0 auto !important;
}

.wa-preview-panel[data-state="flowing"] {
  --wa-status-dot: #34c759;
  --wa-status-color: #34c759;
}

.wa-preview-panel[data-state="error"] {
  --wa-status-dot: #ef4444;
  --wa-status-color: #ef4444;
}

.wa-preview-panel::before {
  content: "";
  position: absolute;
  inset: 18px 20px 268px;
  border-radius: 20px;
  border: 1px solid rgba(113, 113, 122, 0.3);
  background: var(--wa-stage);
  box-shadow: inset 0 0 80px rgba(0, 0, 0, 0.3);
  pointer-events: none;
  z-index: 0;
}

.wa-stage-empty-shell,
.wa-status-host,
.wa-player-chrome,
.wa-control-dock,
.wa-world-tray-shell {
  margin: 0 !important;
}

.wa-stage-empty {
  position: absolute;
  inset: 18px 20px 268px;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0 32px;
  border-radius: 20px;
  background: var(--wa-stage);
  z-index: 4;
  pointer-events: none;
  opacity: 0;
  visibility: hidden;
  transition: opacity 180ms ease, visibility 180ms ease;
}

.wa-preview-panel[data-state="error"] .wa-stage-empty,
.wa-preview-panel[data-state="input"]:not(.wa-has-media) .wa-stage-empty {
  opacity: 1;
  visibility: visible;
}

.wa-preview-panel.wa-has-media .wa-stage-empty {
  opacity: 0;
  visibility: hidden;
}

.wa-preview-panel[data-state="loading"] .wa-stage-empty,
.wa-preview-panel[data-state="queue"] .wa-stage-empty {
  opacity: 0;
  visibility: hidden;
}

.wa-stage-empty-inner {
  width: 100%;
  max-width: 220px;
  display: grid;
  gap: 14px;
  text-align: center;
}

.wa-stage-empty-title {
  color: #d1d5db;
  font: 400 10px/1.45 "Press Start 2P", monospace;
  letter-spacing: 0.12em;
}

.wa-preview-panel[data-state="error"] .wa-stage-empty-title {
  color: #f87171;
}

.wa-stage-empty-copy {
  color: #71717a;
  font: 400 9px/1.6 Menlo, "IBM Plex Mono", monospace;
}

.wa-preview-panel[data-state="error"] .wa-stage-empty-copy {
  color: #fca5a5;
}

.wa-stage-loader {
  display: grid;
  grid-template-columns: repeat(16, minmax(0, 1fr));
  gap: 1px;
  width: 100%;
  padding: 2px;
  border: 2px solid rgba(82, 82, 92, 1);
  background: rgba(24, 24, 27, 1);
}

.wa-stage-loader-cell {
  height: 8px;
  background: #27272a;
  animation: wa-loader-cell 1.2s steps(1, end) infinite;
}

.wa-stage-loader-cell:nth-child(1) { animation-delay: 0s; }
.wa-stage-loader-cell:nth-child(2) { animation-delay: 0.075s; }
.wa-stage-loader-cell:nth-child(3) { animation-delay: 0.15s; }
.wa-stage-loader-cell:nth-child(4) { animation-delay: 0.225s; }
.wa-stage-loader-cell:nth-child(5) { animation-delay: 0.3s; }
.wa-stage-loader-cell:nth-child(6) { animation-delay: 0.375s; }
.wa-stage-loader-cell:nth-child(7) { animation-delay: 0.45s; }
.wa-stage-loader-cell:nth-child(8) { animation-delay: 0.525s; }
.wa-stage-loader-cell:nth-child(9) { animation-delay: 0.6s; }
.wa-stage-loader-cell:nth-child(10) { animation-delay: 0.675s; }
.wa-stage-loader-cell:nth-child(11) { animation-delay: 0.75s; }
.wa-stage-loader-cell:nth-child(12) { animation-delay: 0.825s; }
.wa-stage-loader-cell:nth-child(13) { animation-delay: 0.9s; }
.wa-stage-loader-cell:nth-child(14) { animation-delay: 0.975s; }
.wa-stage-loader-cell:nth-child(15) { animation-delay: 1.05s; }
.wa-stage-loader-cell:nth-child(16) { animation-delay: 1.125s; }

@keyframes wa-loader-cell {
  0%, 100% { background: #27272a; }
  50% { background: #9ca3af; }
}

.wa-status-host {
  display: none !important;
}

.wa-preview-panel .wa-status {
  width: fit-content;
  max-width: 220px;
  margin: 0;
  padding: 7px 12px;
  min-height: 18px;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(49, 52, 58, 0.52);
  color: #a6acb6;
  font: 400 6px/1.35 "Press Start 2P", monospace;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.wa-preview-panel .tabs,
.wa-preview-panel .tabitem,
.wa-preview-panel .tabitem > .block,
.wa-preview-panel .tabitem > .form {
  border: 0 !important;
  background: transparent !important;
}

.wa-preview-panel .tabs [role="tablist"] {
  display: flex !important;
  flex-wrap: nowrap !important;
  align-items: center !important;
  gap: 6px !important;
  overflow-x: auto !important;
  overflow-y: hidden !important;
  width: calc(100% - 12px);
  max-width: 100%;
  margin: -2px auto 14px !important;
  padding: 8px 6px !important;
  scrollbar-width: thin;
}

.wa-preview-panel .tabs [role="tablist"] button {
  flex: 0 0 auto !important;
  min-height: 34px !important;
  padding: 0 12px !important;
  margin: 0 !important;
  border-radius: 999px !important;
  border: 1px solid rgba(255, 255, 255, 0.22) !important;
  background: rgba(55, 58, 64, 0.95) !important;
  color: rgba(232, 234, 240, 0.95) !important;
  font: 600 11px / 1.25 "IBM Plex Mono", monospace !important;
  letter-spacing: 0.02em !important;
  text-transform: none !important;
  white-space: nowrap !important;
  outline: none !important;
  box-shadow: none !important;
}

.wa-preview-panel .tabs [role="tablist"] button.selected,
.wa-preview-panel .tabs [role="tablist"] button[aria-selected="true"] {
  border-color: rgba(251, 146, 60, 0.85) !important;
  background: rgba(251, 146, 60, 0.2) !important;
  color: #fff9f3 !important;
}

#wa-main-preview-video,
#wa-main-preview-image,
.wa-preview-panel .tabitem {
  position: relative;
  z-index: 2;
  min-height: 510px !important;
  height: auto !important;
  padding: 0 !important;
}

.wa-preview-panel .empty,
.wa-preview-panel .image-container,
.wa-preview-panel .video-container,
.wa-preview-panel .model3D,
.wa-preview-panel .block,
.wa-preview-panel .upload-container,
.wa-preview-panel .cm-editor,
.wa-preview-panel .cm-gutters {
  background: transparent !important;
  border-color: transparent !important;
  box-shadow: none !important;
}

#wa-main-preview-video .empty,
#wa-main-preview-image .empty,
#wa-main-preview-video .icon-buttons,
#wa-main-preview-image .icon-buttons {
  display: none !important;
}

#wa-main-preview-video .wrap,
#wa-main-preview-image .wrap {
  min-height: 510px !important;
}

#wa-main-preview-video video,
#wa-main-preview-image img,
.wa-preview-panel model-viewer {
  width: 100% !important;
  height: auto !important;
  max-height: calc(88vh - 246px);
  aspect-ratio: 832 / 480;
  border-radius: 20px !important;
  background: var(--wa-stage) !important;
  object-fit: contain !important;
  border: 1px solid rgba(113, 113, 122, 0.3);
  box-shadow: none;
}

.wa-stage-gallery {
  padding-top: 16px;
}

.wa-preview-panel .wa-stage-points-host,
.wa-preview-panel .wa-stage-embodied-host {
  box-sizing: border-box;
  width: 100% !important;
  min-height: 510px !important;
}

.wa-preview-panel .wa-stage-points-host section.wa-points-viewport[data-wa-viser="1"] {
  display: flex;
  flex-direction: column;
  gap: 10px;
  height: 100%;
  min-height: 480px;
}

.wa-stage-gallery img {
  border-radius: 12px !important;
}

.wa-stage-artifacts pre,
.wa-stage-artifacts code {
  color: #d7dce4;
}

.wa-player-footer {
  position: absolute;
  left: 20px;
  right: 20px;
  bottom: 214px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  color: #9ca3af;
  padding: 2px 0;
  font: 400 10px/1.2 Menlo, "IBM Plex Mono", monospace;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  pointer-events: none;
  z-index: 8;
}

.wa-player-footer-left,
.wa-player-footer-right {
  flex: 1 1 0;
  min-width: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.wa-player-footer-left {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--wa-status-color);
}

.wa-player-footer-center {
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  max-width: min(34vw, 280px);
  color: #a1a1aa;
  font-family: Menlo, "IBM Plex Mono", monospace;
  font-style: italic;
  letter-spacing: -0.05em;
  line-height: 1.2;
  text-transform: none;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.wa-player-footer-right {
  text-align: right;
  color: #9ca3af;
}

.wa-player-dot {
  width: 6px;
  height: 6px;
  margin: 0;
  border-radius: 999px;
  background: var(--wa-status-dot);
  box-shadow: 0 0 12px rgba(252, 211, 77, 0.55);
}

.wa-preview-panel[data-state="flowing"] .wa-player-dot {
  box-shadow: 0 0 12px rgba(52, 199, 89, 0.55);
}

.wa-preview-panel[data-state="error"] .wa-player-dot {
  box-shadow: 0 0 12px rgba(239, 68, 68, 0.55);
}

.wa-control-dock {
  position: relative;
  z-index: 8;
  display: flex !important;
  width: fit-content !important;
  margin: 8px auto 0 !important;
  gap: 6px;
  justify-content: center;
}

.wa-control-dock > *,
.wa-control-dock .wrap,
.wa-control-dock button {
  flex: 0 0 auto !important;
  width: auto !important;
}

.wa-control-dock button {
  min-width: 72px !important;
  min-height: 32px !important;
  padding: 0 12px !important;
  border-radius: 999px !important;
  border: 1px solid rgba(82, 82, 92, 0.5) !important;
  background: rgba(63, 63, 70, 0.5) !important;
  color: #d4d4d8 !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  font: 400 10px/1.2 Menlo, "IBM Plex Mono", monospace !important;
  letter-spacing: 0.05em !important;
  text-transform: uppercase !important;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.1) !important;
}

.wa-action-primary {
  display: none !important;
}

.wa-action-reset {
  background: rgba(63, 63, 70, 0.5) !important;
}

.wa-action-hidden {
  display: none !important;
}

.wa-world-tray-shell .wa-world-tray {
  position: relative !important;
  inset: auto !important;
  left: auto !important;
  bottom: auto !important;
  transform: none !important;
  margin: 0 auto !important;
  width: 100% !important;
  box-sizing: border-box !important;
}

.wa-input-tray-gallery {
  margin-top: 10px;
  padding: 12px;
  border-radius: 16px;
  border: 1px solid rgba(63, 63, 70, 0.5);
  background: var(--wa-tray);
}

.wa-input-tray-gallery .grid-wrap,
.wa-input-tray-gallery .grid-container,
.wa-input-tray-gallery .grid {
  gap: 12px !important;
}

.wa-input-tray-gallery img {
  border-radius: 12px;
}

.wa-world-tray {
  display: flex;
  align-items: center;
  gap: 12px;
  min-height: 98px;
  padding: 12px;
  border-radius: 16px;
  border: 1px solid rgba(63, 63, 70, 0.5);
  background: var(--wa-tray);
}

.wa-tray-item {
  position: relative;
  flex: 0 0 auto;
  width: 64px;
  height: 64px;
  padding: 0;
  overflow: hidden;
  border-radius: 12px;
  border: 2px solid transparent;
  background: transparent;
  cursor: pointer;
  opacity: 0.7;
  transition: transform 150ms ease, border-color 150ms ease, opacity 150ms ease;
}

.wa-tray-item:hover {
  opacity: 1;
}

.wa-tray-item.is-active {
  border-color: rgba(96, 165, 250, 0.5);
  transform: scale(1.05);
  opacity: 1;
  box-shadow: none;
}

.wa-tray-thumb {
  display: block;
  width: 100%;
  height: 100%;
  background-position: center;
  background-size: cover;
}

.wa-tray-item:nth-child(1) .wa-tray-thumb {
  background-image: var(--wa-scene-thumb-1);
}

.wa-tray-item:nth-child(2) .wa-tray-thumb {
  background-image: var(--wa-scene-thumb-2);
}

.wa-tray-item:nth-child(3) .wa-tray-thumb {
  background-image: var(--wa-scene-thumb-3);
}

.wa-tray-thumb-points {
  background: linear-gradient(135deg, rgba(129, 140, 248, 0.55), rgba(56, 189, 248, 0.45)) !important;
}

.wa-tray-item:nth-child(5) .wa-tray-thumb {
  background-image: var(--wa-scene-thumb-4);
}

.wa-tray-item:nth-child(6) .wa-tray-thumb {
  background-image: var(--wa-scene-thumb-5);
}

.wa-tray-item:nth-child(7) .wa-tray-thumb {
  background-image: var(--wa-scene-thumb-6);
}

.wa-tray-item:nth-child(8) .wa-tray-thumb {
  background-image: var(--wa-scene-thumb-7);
}

.wa-tray-item:nth-child(9) .wa-tray-thumb {
  background-image: var(--wa-scene-thumb-8);
}

.wa-tray-item:nth-child(10) .wa-tray-thumb {
  background-image: var(--wa-scene-thumb-9);
}

.wa-tray-item.has-thumb .wa-tray-thumb {
  background-image: var(--wa-thumb-image) !important;
}

.wa-tray-thumb-video {
  background-color: #3f3f46;
}

.wa-tray-thumb-image {
  background-color: #3f3f46;
}

.wa-tray-thumb-3d {
  background-color: #3f3f46;
}

.wa-tray-thumb-gallery {
  background-color: #3f3f46;
}

.wa-tray-thumb-embodied {
  background: linear-gradient(135deg, rgba(167, 139, 250, 0.62), rgba(37, 99, 235, 0.48)) !important;
}

.wa-tray-thumb-artifacts {
  background-color: #3f3f46;
}

.wa-tray-label {
  display: none;
}

.wa-live-dock {
  position: absolute !important;
  left: -9999px !important;
  top: -9999px !important;
  width: 1px !important;
  height: 1px !important;
  overflow: hidden !important;
}

.wa-dom-bridge {
  position: absolute !important;
  left: -9999px !important;
  top: -9999px !important;
  width: 1px !important;
  height: 1px !important;
  overflow: hidden !important;
}

.wa-keyflash {
  box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.18) !important;
}

.wa-progress-card {
  gap: 10px;
}

.wa-progress-shell {
  width: 100%;
  height: 8px;
  overflow: hidden;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(148, 163, 184, 0.22);
}

.wa-progress-bar {
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, rgba(37, 99, 235, 0.88), rgba(59, 130, 246, 0.88));
}

.wa-progress-bar.is-indeterminate {
  width: 34%;
  animation: wa-progress-slide 1.35s ease-in-out infinite;
}

.wa-floating-stick {
  display: none;
}

.wa-stick-shell {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
}

.wa-stick-label {
  color: #8f95a0;
  font: 400 5px/1 "Press Start 2P", monospace;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.wa-stick {
  position: relative;
  width: 68px;
  height: 68px;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: radial-gradient(circle at 28% 28%, rgba(78, 82, 91, 0.96), rgba(43, 45, 52, 0.98));
  touch-action: none;
}

.wa-stick-ring {
  position: absolute;
  inset: 8px;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.wa-stick-thumb {
  position: absolute;
  top: 50%;
  left: 50%;
  width: 24px;
  height: 24px;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.12);
  background: linear-gradient(180deg, rgba(127, 132, 142, 0.96), rgba(82, 86, 96, 0.96));
  transform: translate(-50%, -50%);
}

.gradio-container .accordion {
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  border-radius: 16px !important;
  background: rgba(255, 255, 255, 0.04) !important;
}

.gradio-container .accordion summary,
.gradio-container label span {
  color: #b1b7c1 !important;
}

.gradio-container textarea,
.gradio-container input,
.gradio-container select {
  background: rgba(255, 255, 255, 0.06) !important;
  color: #f3f4f6 !important;
  border-color: rgba(255, 255, 255, 0.12) !important;
}

.gradio-container textarea::placeholder,
.gradio-container input::placeholder {
  color: #737a84 !important;
}

.wa-dataframe table th,
.wa-dataframe table td {
  border-color: rgba(255, 255, 255, 0.08) !important;
}

.wa-dataframe table th {
  background: rgba(255, 255, 255, 0.04) !important;
}

.wa-dataframe table td {
  background: rgba(255, 255, 255, 0.03) !important;
}

@media (pointer: coarse) {
  .wa-floating-stick {
    display: flex;
  }
}

@media (max-width: 980px) {
  .wa-left-rail,
  .wa-right-rail {
    position: static !important;
    width: auto !important;
    min-width: 0 !important;
    max-width: none !important;
    transform: none !important;
    opacity: 1 !important;
  }

  .wa-left-rail::after,
  .wa-right-rail::before {
    display: none;
  }

  .wa-stage-col {
    padding: 0 !important;
  }

  .wa-stage-col > .block,
  .wa-stage-col > .form {
    width: min(960px, 100%) !important;
    max-width: min(960px, 100%) !important;
  }

  .wa-preview-panel {
    border-radius: 32px !important;
  }

  .wa-preview-panel::before {
    inset: 16px 16px 246px;
  }

  .wa-stage-empty {
    inset: 16px 16px 246px;
  }

  .wa-world-tray {
    overflow-x: auto;
  }
}

@media (max-width: 760px) {
  .gradio-container {
    padding: 0 8px 18px !important;
  }

  .wa-notice-copy {
    font-size: 10px;
  }

  .wa-site-nav {
    width: calc(100vw - 16px);
  }

  .wa-stage-col > .block,
  .wa-stage-col > .form {
    width: 100% !important;
    max-width: 100% !important;
  }

  .wa-preview-panel {
    min-height: 560px;
    padding: 12px 12px 16px !important;
    border-radius: 24px !important;
  }

  .wa-preview-panel::before {
    inset: 12px 12px 214px;
    border-radius: 18px;
  }

  .wa-stage-empty {
    inset: 14px 14px 214px;
    padding: 0 20px;
  }

  #wa-main-preview-video,
  #wa-main-preview-image,
  .wa-preview-panel .tabitem {
    min-height: 300px !important;
  }

  .wa-status-host {
    bottom: 112px;
  }

  .wa-player-footer {
    bottom: 172px;
    font-size: 9px;
  }

  .wa-world-tray {
    gap: 8px;
    padding: 10px 12px;
  }

  .wa-tray-item {
    width: 40px;
    height: 40px;
  }
}

/* WorldFoundry product refinements */
:root {
  --wa-page-bg: #202226;
  --wa-page-bg-2: #191b20;
  --wa-page-glow: rgba(148, 163, 184, 0.08);
  --wa-nav-bg: rgba(39, 41, 47, 0.82);
  --wa-shell-bg: rgba(39, 41, 47, 0.92);
  --wa-drawer-bg: rgba(43, 46, 53, 0.98);
  --wa-drawer-soft: rgba(63, 67, 76, 0.36);
  --wa-stage-bg: #09090b;
  --wa-stage-ring: rgba(113, 113, 122, 0.3);
  --wa-stage-inset: rgba(0, 0, 0, 0.3);
  --wa-tray-bg: rgba(24, 24, 27, 0.4);
  --wa-input-bg: rgba(255, 255, 255, 0.06);
  --wa-chip-bg: rgba(63, 63, 70, 0.5);
  --wa-chip-bg-hover: rgba(82, 82, 92, 0.72);
  --wa-handle-bg: rgba(63, 63, 70, 0.7);
  --wa-ink: #f3f4f6;
  --wa-ink-strong: #f8fafc;
  --wa-ink-soft: #d7dce4;
  --wa-muted: #b1b7c1;
  --wa-muted-soft: #737a84;
  --wa-line: rgba(255, 255, 255, 0.08);
  --wa-line-strong: rgba(255, 255, 255, 0.14);
  --wa-atlas-bg: linear-gradient(180deg, rgba(96, 165, 250, 0.16), rgba(37, 99, 235, 0.08));
  --wa-atlas-metric-bg: rgba(255, 255, 255, 0.04);
  --wa-guide-bg: rgba(24, 24, 27, 0.94);
  --wa-guide-key-bg: rgba(63, 63, 70, 0.72);
  --wa-guide-key-bg-hover: rgba(96, 165, 250, 0.18);
  --wa-guide-key-ink: #f3f4f6;
  --wa-guide-key-muted: #a1a1aa;
  --wa-loader-cell: #27272a;
  --wa-loader-active: #9ca3af;
  --wa-summary-bg: rgba(255, 255, 255, 0.04);
}

html[data-wa-theme="light"] {
  --wa-page-bg: #f3f6fb;
  --wa-page-bg-2: #e4ebf5;
  --wa-page-glow: rgba(37, 99, 235, 0.12);
  --wa-nav-bg: rgba(255, 255, 255, 0.78);
  --wa-shell-bg: rgba(255, 255, 255, 0.88);
  --wa-drawer-bg: rgba(255, 255, 255, 0.94);
  --wa-drawer-soft: rgba(37, 99, 235, 0.08);
  --wa-stage-bg: #f8fafc;
  --wa-stage-ring: rgba(148, 163, 184, 0.35);
  --wa-stage-inset: rgba(148, 163, 184, 0.14);
  --wa-tray-bg: rgba(255, 255, 255, 0.72);
  --wa-input-bg: rgba(255, 255, 255, 0.78);
  --wa-chip-bg: rgba(226, 232, 240, 0.9);
  --wa-chip-bg-hover: rgba(191, 219, 254, 0.86);
  --wa-handle-bg: rgba(203, 213, 225, 0.95);
  --wa-ink: #0f172a;
  --wa-ink-strong: #020617;
  --wa-ink-soft: #0f172a;
  --wa-muted: #475569;
  --wa-muted-soft: #64748b;
  --wa-line: rgba(15, 23, 42, 0.08);
  --wa-line-strong: rgba(15, 23, 42, 0.12);
  --wa-atlas-bg: linear-gradient(180deg, rgba(59, 130, 246, 0.14), rgba(125, 211, 252, 0.08));
  --wa-atlas-metric-bg: rgba(255, 255, 255, 0.72);
  --wa-guide-bg: rgba(248, 250, 252, 0.96);
  --wa-guide-key-bg: rgba(226, 232, 240, 0.96);
  --wa-guide-key-bg-hover: rgba(191, 219, 254, 0.92);
  --wa-guide-key-ink: #0f172a;
  --wa-guide-key-muted: #475569;
  --wa-loader-cell: #dbe2ea;
  --wa-loader-active: #64748b;
  --wa-summary-bg: rgba(255, 255, 255, 0.74);
}

html,
body {
  background: var(--wa-page-bg-2) !important;
}

body,
.gradio-container {
  background:
    radial-gradient(circle at top center, var(--wa-page-glow), transparent 18%),
    linear-gradient(180deg, var(--wa-page-bg) 0%, var(--wa-page-bg-2) 100%) !important;
  color: var(--wa-ink) !important;
}

.wa-notice-bar {
  background: var(--wa-nav-bg);
  border-bottom-color: var(--wa-line);
  color: var(--wa-ink-soft);
}

.wa-notice-icon,
.wa-notice-close {
  color: var(--wa-muted-soft);
}

.wa-notice-close:hover {
  background: var(--wa-chip-bg);
  color: var(--wa-ink-strong);
}

.wa-site-nav {
  color: var(--wa-ink-soft);
}

.wa-site-brand {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 4px 12px 4px 4px;
  border-radius: 999px;
  border: 1px solid var(--wa-line);
  background: var(--wa-chip-bg);
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
  opacity: 1;
  color: var(--wa-ink-strong);
  letter-spacing: 0.02em;
  line-height: 1.1;
  text-transform: none;
}

.wa-site-brand-mark {
  width: 88px;
  height: 44px;
  flex: 0 0 auto;
  overflow: hidden;
  border-radius: 12px;
  padding: 4px;
  background: #0b2268;
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.08);
}

.wa-site-brand-mark img {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
  object-position: center;
  transform: none;
}

.wa-site-brand-fallback {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  border-radius: 10px;
  background: linear-gradient(135deg, #2563eb, #60a5fa);
  color: #eff6ff;
  font: 700 11px/1 "IBM Plex Mono", monospace;
}

.wa-site-nav-left {
  white-space: nowrap;
  display: block;
  color: var(--wa-ink-strong) !important;
  font: 700 15px/1.15 "Space Grotesk", sans-serif;
  letter-spacing: 0.01em;
  text-shadow: 0 1px 0 rgba(255, 255, 255, 0.28);
}

.wa-notice-copy {
  color: var(--wa-muted) !important;
  opacity: 1 !important;
}

.wa-notice-shell {
  display: none !important;
}

.wa-site-nav-icon {
  border: 1px solid var(--wa-line);
  background: var(--wa-chip-bg);
  color: var(--wa-muted);
  box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
}

.wa-site-nav-icon:hover,
.wa-site-nav-icon.is-active {
  transform: translateY(-1px);
  background: var(--wa-chip-bg-hover);
  border-color: var(--wa-line-strong);
  color: var(--wa-ink-strong);
}

.wa-theme-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.wa-theme-icon-sun {
  display: none;
}

html[data-wa-theme="light"] .wa-theme-icon-sun {
  display: inline-flex;
}

html[data-wa-theme="light"] .wa-theme-icon-moon {
  display: none;
}

.wa-left-rail,
.wa-right-rail {
  width: 356px !important;
  min-width: 356px !important;
  max-width: 356px !important;
  opacity: 0.16;
}

.wa-left-rail {
  transform: translate(calc(-100% + 26px), -50%);
}

.wa-right-rail {
  transform: translate(calc(100% - 26px), -50%);
}

.wa-left-rail::after,
.wa-right-rail::before {
  width: 16px;
  height: 148px;
  background: var(--wa-handle-bg);
  border-color: var(--wa-line-strong);
}

.wa-left-rail .wa-panel-block,
.wa-right-rail .wa-panel-block {
  background: var(--wa-drawer-bg) !important;
  max-height: min(680px, calc(100vh - 118px));
  padding: 16px !important;
}

.wa-panel-title {
  color: var(--wa-ink-strong);
  line-height: 1.18;
  overflow-wrap: anywhere;
}

.wa-left-rail .wa-panel-copy,
.wa-right-rail .wa-panel-copy {
  display: block !important;
  margin: 0 0 12px;
  color: var(--wa-muted) !important;
  font-size: 13px;
  line-height: 1.55;
  overflow-wrap: anywhere;
}

.wa-atlas-card {
  padding: 16px;
  border-radius: 18px;
  border: 1px solid var(--wa-line);
  background: var(--wa-atlas-bg);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
}

.wa-atlas-kicker {
  color: var(--wa-muted);
  font: 500 10px/1.2 "IBM Plex Mono", monospace;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.wa-atlas-title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-top: 10px;
}

.wa-atlas-title-row h4 {
  margin: 0;
  color: var(--wa-ink-strong);
  font: 700 22px/1.08 "Space Grotesk", sans-serif;
}

.wa-atlas-title-row span {
  display: inline-flex;
  align-items: center;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid var(--wa-line);
  background: var(--wa-atlas-metric-bg);
  color: var(--wa-ink-soft);
  font: 500 10px/1.2 "IBM Plex Mono", monospace;
}

.wa-atlas-copy {
  margin: 10px 0 0;
  color: var(--wa-muted);
  font-size: 13px;
  line-height: 1.58;
  overflow-wrap: anywhere;
}

.wa-atlas-metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
  margin-top: 14px;
}

.wa-atlas-metric {
  padding: 10px 10px 11px;
  border-radius: 14px;
  border: 1px solid var(--wa-line);
  background: var(--wa-atlas-metric-bg);
}

.wa-atlas-metric strong {
  display: block;
  color: var(--wa-ink-strong);
  font: 700 18px/1 "Space Grotesk", sans-serif;
}

.wa-atlas-metric span {
  display: block;
  margin-top: 4px;
  color: var(--wa-muted);
  font-size: 11px;
}

.wa-atlas-focus {
  margin-top: 12px;
  padding: 11px 12px;
  border-radius: 14px;
  border: 1px solid var(--wa-line);
  background: var(--wa-atlas-metric-bg);
}

.wa-atlas-focus-label {
  color: var(--wa-muted);
  font: 500 10px/1 "IBM Plex Mono", monospace;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.wa-atlas-focus strong {
  display: block;
  margin-top: 6px;
  color: var(--wa-ink-strong);
  font-size: 14px;
  line-height: 1.4;
  overflow-wrap: anywhere;
}

.wa-profile {
  background: var(--wa-drawer-soft);
  border-color: var(--wa-line);
}

.wa-profile-eyebrow {
  color: var(--wa-muted);
}

.wa-profile h3 {
  color: var(--wa-ink-strong);
}

.wa-profile p {
  color: var(--wa-muted);
  overflow-wrap: anywhere;
}

.wa-pill,
.wa-summary-pill {
  border-color: var(--wa-line);
  background: var(--wa-summary-bg);
  color: var(--wa-ink-soft);
}

.wa-runnote {
  border-color: var(--wa-line);
  background: var(--wa-summary-bg);
  color: var(--wa-muted);
}

.wa-summary-card {
  display: block !important;
  background: var(--wa-summary-bg);
  border-color: var(--wa-line);
}

.wa-summary-card h4 {
  color: var(--wa-ink-strong);
}

.wa-summary-subtitle,
.wa-summary-lines div {
  color: var(--wa-muted);
  overflow-wrap: anywhere;
}

.wa-stage-col > .block,
.wa-stage-col > .form {
  width: min(980px, calc(100vw - 72px)) !important;
  max-width: min(980px, calc(100vw - 72px)) !important;
  transition: width 180ms ease, max-width 180ms ease;
}

.wa-stage-col {
  transition: transform 180ms ease;
}

.wa-preview-panel {
  width: min(980px, calc(100vw - 72px)) !important;
  max-width: 980px !important;
  min-height: 720px;
  max-height: none !important;
  background: var(--wa-shell-bg) !important;
  border-color: var(--wa-line-strong) !important;
  backdrop-filter: blur(20px);
  transition: width 180ms ease, max-width 180ms ease, transform 180ms ease;
}

.wa-main-grid:has(.wa-left-rail:hover) .wa-stage-col,
.wa-main-grid:has(.wa-left-rail:focus-within) .wa-stage-col {
  transform: translateX(184px);
}

.wa-main-grid:has(.wa-right-rail:hover) .wa-stage-col,
.wa-main-grid:has(.wa-right-rail:focus-within) .wa-stage-col {
  transform: translateX(-184px);
}

.wa-main-grid:has(.wa-left-rail:hover) .wa-preview-panel,
.wa-main-grid:has(.wa-left-rail:focus-within) .wa-preview-panel,
.wa-main-grid:has(.wa-right-rail:hover) .wa-preview-panel,
.wa-main-grid:has(.wa-right-rail:focus-within) .wa-preview-panel,
.wa-main-grid:has(.wa-left-rail:hover) .wa-stage-col > .block,
.wa-main-grid:has(.wa-left-rail:focus-within) .wa-stage-col > .block,
.wa-main-grid:has(.wa-right-rail:hover) .wa-stage-col > .block,
.wa-main-grid:has(.wa-right-rail:focus-within) .wa-stage-col > .block,
.wa-main-grid:has(.wa-left-rail:hover) .wa-stage-col > .form,
.wa-main-grid:has(.wa-left-rail:focus-within) .wa-stage-col > .form,
.wa-main-grid:has(.wa-right-rail:hover) .wa-stage-col > .form,
.wa-main-grid:has(.wa-right-rail:focus-within) .wa-stage-col > .form {
  width: min(900px, calc(100vw - 432px)) !important;
  max-width: min(900px, calc(100vw - 432px)) !important;
}

.wa-preview-panel::before {
  background: var(--wa-stage-bg);
  border-color: var(--wa-stage-ring);
  box-shadow: inset 0 0 80px var(--wa-stage-inset);
}

.wa-stage-empty {
  background: var(--wa-stage-bg);
  align-items: flex-start;
  justify-content: flex-start;
  padding: clamp(34px, 4vw, 46px) 32px 24px;
  box-sizing: border-box;
  overflow: hidden;
}

.wa-stage-empty-inner {
  width: min(100%, 360px);
  max-width: 360px;
  margin: 0 auto;
  gap: 12px;
}

.wa-stage-empty-title {
  color: var(--wa-ink-soft);
  font-size: 11px;
  line-height: 1.35;
}

.wa-stage-empty-copy {
  color: var(--wa-muted-soft);
  font-size: 10px;
  line-height: 1.6;
  overflow-wrap: anywhere;
  text-wrap: balance;
}

.wa-stage-loader {
  border-color: var(--wa-line-strong);
  background: var(--wa-stage-bg);
}

.wa-stage-loader-cell {
  background: var(--wa-loader-cell);
  animation-name: wa-loader-cell-theme;
}

@keyframes wa-loader-cell-theme {
  0%, 100% { background: var(--wa-loader-cell); }
  50% { background: var(--wa-loader-active); }
}

#wa-main-preview-video video,
#wa-main-preview-image img,
.wa-preview-panel model-viewer {
  background: var(--wa-stage-bg) !important;
  border-color: var(--wa-stage-ring);
}

.wa-stage-artifacts pre,
.wa-stage-artifacts code {
  color: var(--wa-ink-soft);
}

.wa-player-footer {
  color: var(--wa-muted);
}

.wa-player-footer-center {
  color: var(--wa-muted);
}

.wa-player-footer-right {
  color: var(--wa-muted);
}

.wa-control-dock button {
  background: var(--wa-chip-bg) !important;
  border-color: var(--wa-line-strong) !important;
  color: var(--wa-ink-soft) !important;
}

.wa-action-primary {
  display: inline-flex !important;
  align-items: center;
  justify-content: center;
  background: linear-gradient(180deg, #3b82f6, #2563eb) !important;
  color: #eff6ff !important;
  border-color: transparent !important;
}

.wa-action-reset {
  background: var(--wa-chip-bg) !important;
}

.wa-world-tray {
  border-color: var(--wa-line-strong);
  background: var(--wa-tray-bg);
}

.wa-tray-item.is-active {
  border-color: rgba(96, 165, 250, 0.6);
  box-shadow: 0 0 0 1px rgba(96, 165, 250, 0.18);
}

.wa-tray-thumb-video,
.wa-tray-thumb-image,
.wa-tray-thumb-3d,
.wa-tray-thumb-gallery,
.wa-tray-thumb-embodied,
.wa-tray-thumb-artifacts {
  background-color: rgba(148, 163, 184, 0.22);
}

.gradio-container .accordion {
  border-color: var(--wa-line) !important;
  background: var(--wa-summary-bg) !important;
}

.gradio-container .accordion summary,
.gradio-container label span {
  color: var(--wa-muted) !important;
  line-height: 1.3;
}

.gradio-container textarea,
.gradio-container input,
.gradio-container select {
  background: var(--wa-input-bg) !important;
  color: var(--wa-ink) !important;
  border-color: var(--wa-line-strong) !important;
}

.gradio-container textarea::placeholder,
.gradio-container input::placeholder {
  color: var(--wa-muted-soft) !important;
}

.wa-dataframe table th,
.wa-dataframe table td {
  border-color: var(--wa-line) !important;
}

.wa-dataframe table th {
  background: var(--wa-summary-bg) !important;
}

.wa-dataframe table td {
  background: transparent !important;
}

.wa-joystick-dock {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 18px;
  position: relative;
  min-height: 116px;
  height: 116px;
  margin-top: 10px;
  padding: 10px 12px 0;
  box-sizing: border-box;
  overflow: visible;
  opacity: 0;
  visibility: hidden;
  pointer-events: none;
  transition: opacity 180ms ease, visibility 180ms ease;
}

.wa-preview-panel.wa-joystick-open .wa-joystick-dock {
  opacity: 1;
  visibility: visible;
  pointer-events: auto;
}

.wa-preview-panel.wa-joystick-open #wa-main-preview-video,
.wa-preview-panel.wa-joystick-open #wa-main-preview-image,
.wa-preview-panel.wa-joystick-open .tabitem {
  min-height: 452px !important;
}

.wa-preview-panel.wa-joystick-open #wa-main-preview-video .wrap,
.wa-preview-panel.wa-joystick-open #wa-main-preview-image .wrap {
  min-height: 452px !important;
}

.wa-joystick-dock-copy {
  display: grid;
  gap: 6px;
  position: absolute;
  left: 50%;
  bottom: 10px;
  transform: translateX(-50%);
  max-width: 260px;
  padding: 0;
  text-align: center;
}

.wa-joystick-dock-title {
  color: var(--wa-ink-strong);
  font: 700 12px/1.2 "IBM Plex Mono", monospace;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.wa-joystick-dock-note {
  color: var(--wa-muted);
  font-size: 12px;
  line-height: 1.5;
  overflow-wrap: anywhere;
}

.wa-joystick-dock.is-disabled .wa-floating-stick {
  opacity: 0.36;
}

.wa-joystick-dock .wa-floating-stick {
  position: relative;
  bottom: auto;
  left: auto !important;
  right: auto !important;
  display: flex;
  width: auto;
  padding: 0;
  border: 0;
  background: transparent;
  box-shadow: none;
}

.wa-joystick-dock .wa-stick-shell {
  gap: 8px;
  padding-top: 4px;
}

.wa-joystick-dock .wa-stick-label {
  color: var(--wa-muted);
  font: 500 10px/1 "IBM Plex Mono", monospace;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.wa-joystick-dock .wa-stick {
  width: 82px;
  height: 82px;
  border-color: var(--wa-line-strong);
  background: radial-gradient(circle at 28% 28%, rgba(148, 163, 184, 0.18), var(--wa-chip-bg));
}

.wa-joystick-dock .wa-stick-ring {
  border-color: var(--wa-line-strong);
}

.wa-joystick-dock .wa-stick-thumb {
  width: 28px;
  height: 28px;
  border-color: var(--wa-line-strong);
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.88), rgba(148, 163, 184, 0.78));
}

html[data-wa-theme="light"] .wa-site-brand-mark {
  box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.06), 0 10px 24px rgba(37, 99, 235, 0.12);
}

html[data-wa-theme="light"] .wa-site-brand {
  background: rgba(255, 255, 255, 0.94);
  border-color: rgba(15, 23, 42, 0.08);
  box-shadow: 0 10px 24px rgba(37, 99, 235, 0.12);
}

html[data-wa-theme="light"] .wa-left-rail,
html[data-wa-theme="light"] .wa-right-rail {
  opacity: 0.24;
}

html[data-wa-theme="light"] .wa-site-brand,
html[data-wa-theme="light"] .wa-player-footer,
html[data-wa-theme="light"] .wa-stage-empty-title,
html[data-wa-theme="light"] .wa-stage-empty-copy,
html[data-wa-theme="light"] .wa-preview-panel .wa-status,
html[data-wa-theme="light"] .wa-profile p,
html[data-wa-theme="light"] .wa-runnote,
html[data-wa-theme="light"] .wa-summary-subtitle,
html[data-wa-theme="light"] .wa-summary-lines div,
html[data-wa-theme="light"] .wa-atlas-copy,
html[data-wa-theme="light"] .wa-atlas-title-row span,
html[data-wa-theme="light"] .wa-atlas-focus-label,
html[data-wa-theme="light"] .wa-atlas-metric span,
html[data-wa-theme="light"] .wa-joystick-dock-note,
html[data-wa-theme="light"] .wa-left-rail .wa-panel-copy,
html[data-wa-theme="light"] .wa-right-rail .wa-panel-copy {
  color: var(--wa-muted) !important;
  opacity: 1 !important;
}

html[data-wa-theme="light"] .wa-panel-title,
html[data-wa-theme="light"] .wa-profile h3,
html[data-wa-theme="light"] .wa-summary-card h4,
html[data-wa-theme="light"] .wa-atlas-title-row h4,
html[data-wa-theme="light"] .wa-atlas-focus strong,
html[data-wa-theme="light"] .wa-live-group-title,
html[data-wa-theme="light"] .wa-joystick-dock-title,
html[data-wa-theme="light"] .wa-joystick-dock .wa-stick-label,
html[data-wa-theme="light"] .wa-site-nav-icon,
html[data-wa-theme="light"] .wa-spark-title,
html[data-wa-theme="light"] .wa-spark-copy,
html[data-wa-theme="light"] .wa-spark-loading,
html[data-wa-theme="light"] .wa-spatial-caption p {
  color: #0f172a !important;
  opacity: 1 !important;
  text-shadow: none !important;
}

html[data-wa-theme="light"] .wa-preview-panel[data-state="error"] .wa-stage-empty-title {
  color: #dc2626 !important;
}

html[data-wa-theme="light"] .wa-preview-panel[data-state="error"] .wa-stage-empty-copy {
  color: #b91c1c !important;
}

html[data-wa-theme="light"] .wa-preview-panel[data-state="flowing"] .wa-player-footer-left {
  color: #15803d !important;
}

html[data-wa-theme="light"] .wa-preview-panel[data-state="error"] .wa-player-footer-left {
  color: #dc2626 !important;
}

html[data-wa-theme="light"] .wa-control-dock button,
html[data-wa-theme="light"] .wa-action-reset,
html[data-wa-theme="light"] .wa-run-muted,
html[data-wa-theme="light"] .wa-run-muted button {
  color: #0f172a !important;
  opacity: 1 !important;
}

html[data-wa-theme="light"] .wa-action-primary,
html[data-wa-theme="light"] .wa-action-primary button {
  color: #eff6ff !important;
  opacity: 1 !important;
}

html[data-wa-theme="light"] .wa-joystick-dock .wa-stick {
  background: radial-gradient(circle at 28% 28%, rgba(255, 255, 255, 0.96), rgba(226, 232, 240, 0.96));
}

html[data-wa-theme="light"] .wa-joystick-dock .wa-stick-thumb {
  background: linear-gradient(180deg, #ffffff, #cbd5e1);
}

@media (max-width: 980px) {
  .wa-main-grid:has(.wa-left-rail:hover) .wa-stage-col,
  .wa-main-grid:has(.wa-left-rail:focus-within) .wa-stage-col,
  .wa-main-grid:has(.wa-right-rail:hover) .wa-stage-col,
  .wa-main-grid:has(.wa-right-rail:focus-within) .wa-stage-col {
    transform: none;
  }

  .wa-main-grid:has(.wa-left-rail:hover) .wa-preview-panel,
  .wa-main-grid:has(.wa-left-rail:focus-within) .wa-preview-panel,
  .wa-main-grid:has(.wa-right-rail:hover) .wa-preview-panel,
  .wa-main-grid:has(.wa-right-rail:focus-within) .wa-preview-panel,
  .wa-main-grid:has(.wa-left-rail:hover) .wa-stage-col > .block,
  .wa-main-grid:has(.wa-left-rail:focus-within) .wa-stage-col > .block,
  .wa-main-grid:has(.wa-right-rail:hover) .wa-stage-col > .block,
  .wa-main-grid:has(.wa-right-rail:focus-within) .wa-stage-col > .block,
  .wa-main-grid:has(.wa-left-rail:hover) .wa-stage-col > .form,
  .wa-main-grid:has(.wa-left-rail:focus-within) .wa-stage-col > .form,
  .wa-main-grid:has(.wa-right-rail:hover) .wa-stage-col > .form,
  .wa-main-grid:has(.wa-right-rail:focus-within) .wa-stage-col > .form {
    width: min(960px, 100%) !important;
    max-width: min(960px, 100%) !important;
  }

  .wa-preview-panel.wa-joystick-open .wa-joystick-dock {
    height: 104px;
    min-height: 104px;
  }

  .wa-joystick-dock-copy {
    max-width: 220px;
  }
}

@media (max-width: 760px) {
  .wa-joystick-dock-copy {
    position: static;
    transform: none;
    max-width: 160px;
    text-align: left;
  }

  .wa-preview-panel.wa-joystick-open .wa-joystick-dock {
    height: 92px;
    min-height: 92px;
    justify-content: space-between;
    gap: 14px;
  }

  .wa-joystick-dock .wa-stick {
    width: 72px;
    height: 72px;
  }
}

.wa-stage-splat-host {
  margin-bottom: 12px;
}

.wa-stage-splat-host > div {
  width: 100%;
}

.wa-stage-points-host {
  margin-bottom: 12px;
  min-height: 480px;
  width: 100%;
}

.wa-stage-points-host section.wa-points-viewport[data-wa-viser="1"] {
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-height: 460px;
}

.wa-points-viewport[data-wa-viser="1"] .wa-viser-frame {
  flex: 1 1 auto;
  width: 100%;
  min-height: 420px;
}

.wa-points-viewport {
  width: 100%;
}

.wa-points-viewport--idle {
  border: 1px solid var(--wa-stage-ring);
  border-radius: 22px;
  padding: 22px 24px;
  min-height: 200px;
  background: linear-gradient(135deg, rgba(8, 12, 22, 0.95), rgba(5, 10, 18, 0.98));
}

.wa-points-fallback-title {
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  opacity: 0.86;
  margin-bottom: 10px;
}

.wa-points-fallback-detail {
  font-size: 14px;
  line-height: 1.5;
  opacity: 0.75;
}

.wa-viser-frame {
  width: 100%;
  height: 560px;
  border: 0;
  border-radius: 22px;
  background: #04060b;
}

.wa-points-caption {
  margin-top: 10px;
  font-size: 12px;
  letter-spacing: 0.02em;
  opacity: 0.65;
}

.wa-stage-embodied-host {
  margin-bottom: 12px;
  min-height: 480px;
  width: 100%;
}

.wa-embodied-viewport {
  width: 100%;
  min-height: 460px;
  display: grid;
  gap: 14px;
  align-content: start;
  border: 1px solid var(--wa-stage-ring);
  border-radius: 22px;
  padding: 18px;
  background:
    linear-gradient(135deg, rgba(15, 23, 42, 0.94), rgba(2, 6, 23, 0.98)),
    radial-gradient(circle at top right, rgba(37, 99, 235, 0.18), transparent 32%);
}

.wa-embodied-head {
  display: grid;
  gap: 6px;
}

.wa-embodied-head span {
  color: rgba(191, 219, 254, 0.86);
  font: 700 11px/1 "IBM Plex Mono", monospace;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.wa-embodied-head strong {
  color: #f8fafc;
  font: 700 22px/1.08 "Space Grotesk", sans-serif;
}

.wa-embodied-copy {
  color: rgba(226, 232, 240, 0.78);
  font-size: 14px;
  line-height: 1.55;
}

.wa-embodied-pill-row,
.wa-embodied-artifacts {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.wa-embodied-pill-row span,
.wa-embodied-artifact {
  display: inline-flex;
  align-items: center;
  min-height: 34px;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.24);
  background: rgba(15, 23, 42, 0.62);
  color: rgba(226, 232, 240, 0.86);
  padding: 0 12px;
  font: 600 11px/1 "IBM Plex Mono", monospace;
  text-decoration: none;
}

.wa-embodied-artifact {
  display: grid;
  grid-template-columns: auto;
  align-items: start;
  gap: 4px;
  min-width: 190px;
  border-radius: 14px;
  padding: 12px 14px;
}

.wa-embodied-artifact strong {
  color: #f8fafc;
  font-size: 12px;
  line-height: 1.25;
  word-break: break-word;
}

.wa-embodied-artifact em {
  color: rgba(203, 213, 225, 0.66);
  font-style: normal;
  font-size: 11px;
}

.wa-embodied-video {
  width: 100%;
  max-height: 380px;
  border-radius: 18px;
  border: 1px solid rgba(148, 163, 184, 0.22);
  background: #020617;
}

.wa-spatial-shell {
  --wa-spatial-poster: none;
  position: relative;
  width: 100%;
  height: 560px;
  min-height: 560px;
  border-radius: 24px;
  overflow: hidden;
  border: 1px solid var(--wa-stage-ring);
  background:
    radial-gradient(circle at top, rgba(59, 130, 246, 0.18), transparent 32%),
    linear-gradient(180deg, rgba(5, 7, 11, 0.98), rgba(6, 10, 18, 0.98));
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.04);
}

.wa-spatial-shell::before {
  content: "";
  position: absolute;
  inset: 0;
  background:
    linear-gradient(180deg, rgba(5, 7, 11, 0.18), rgba(5, 7, 11, 0.54)),
    var(--wa-spatial-poster) center / cover no-repeat;
  opacity: 0.24;
  transform: scale(1.02);
  transition: opacity 180ms ease;
}

.wa-spatial-shell.is-ready::before {
  opacity: 0.08;
}

.wa-spark-canvas {
  position: absolute;
  inset: 0;
  display: block;
  width: 100%;
  height: 100%;
  min-height: 560px;
}

.wa-spatial-shell.is-empty .wa-spark-canvas,
.wa-spatial-shell.is-error .wa-spark-canvas {
  opacity: 0.18;
}

.wa-spark-overlay {
  position: absolute;
  inset: 0;
  z-index: 1;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding: 18px 20px;
  background: linear-gradient(180deg, rgba(5, 7, 11, 0.72), rgba(5, 7, 11, 0.16) 34%, rgba(5, 7, 11, 0.78));
  pointer-events: none;
}

.wa-spatial-shell.is-ready .wa-spark-overlay {
  background: linear-gradient(180deg, rgba(5, 7, 11, 0.32), rgba(5, 7, 11, 0.04) 32%, rgba(5, 7, 11, 0.64));
}

.wa-preview-panel[data-wa-active-tab="3D World"][data-wa-spatial="splat"] .wa-spatial-shell.is-ready::before,
.wa-preview-panel[data-wa-active-tab="3D World"][data-wa-spatial="splat"] .wa-spatial-shell.is-ready .wa-spark-overlay,
.wa-preview-panel[data-wa-active-tab="3D World"][data-wa-spatial="splat"] .wa-player-chrome {
  opacity: 0;
  transition: opacity 180ms ease;
}

.wa-preview-panel[data-wa-active-tab="3D World"][data-wa-spatial="splat"]:hover .wa-spatial-shell.is-ready::before,
.wa-preview-panel[data-wa-active-tab="3D World"][data-wa-spatial="splat"]:hover .wa-spatial-shell.is-ready .wa-spark-overlay,
.wa-preview-panel[data-wa-active-tab="3D World"][data-wa-spatial="splat"]:hover .wa-player-chrome {
  opacity: 1;
}

.wa-spark-copy-stack {
  display: grid;
  gap: 8px;
  max-width: 440px;
}

.wa-spark-kicker {
  display: inline-flex;
  width: fit-content;
  align-items: center;
  padding: 7px 11px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.12);
  color: rgba(226, 232, 240, 0.92);
  font: 600 11px/1 "IBM Plex Mono", monospace;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.wa-spark-title {
  color: #f8fafc;
  font: 700 28px/1.02 "Space Grotesk", sans-serif;
  letter-spacing: -0.03em;
}

.wa-spark-copy {
  color: rgba(226, 232, 240, 0.86);
  font-size: 14px;
  line-height: 1.55;
}

.wa-spark-hud {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.wa-spark-hud span {
  display: inline-flex;
  align-items: center;
  padding: 8px 11px;
  border-radius: 999px;
  background: rgba(15, 23, 42, 0.58);
  border: 1px solid rgba(148, 163, 184, 0.22);
  color: rgba(226, 232, 240, 0.88);
  font: 500 11px/1 "IBM Plex Mono", monospace;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.wa-spark-loading {
  position: absolute;
  top: 18px;
  right: 18px;
  z-index: 2;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 36px;
  padding: 0 14px;
  border-radius: 999px;
  background: rgba(15, 23, 42, 0.74);
  border: 1px solid rgba(148, 163, 184, 0.24);
  color: #dbeafe;
  font: 600 11px/1 "IBM Plex Mono", monospace;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  opacity: 0;
  transition: opacity 180ms ease;
}

.wa-spatial-shell.is-loading .wa-spark-loading,
.wa-spatial-shell.is-error .wa-spark-loading {
  opacity: 1;
}

.wa-spatial-shell.is-error .wa-spark-copy,
.wa-spatial-shell.is-error .wa-spark-loading {
  color: #fecaca;
}

.wa-preview-panel[data-wa-spatial="splat"] .wa-stage-model {
  display: none !important;
}

.wa-spatial-caption {
  min-height: 42px;
}

.wa-spatial-caption p {
  margin: 0;
  color: var(--wa-muted);
  font-size: 12px;
  line-height: 1.5;
}

html[data-wa-theme="light"] .wa-site-nav-left,
html[data-wa-theme="light"] .wa-site-brand .wa-site-nav-left {
  color: #0f172a !important;
  opacity: 1 !important;
  text-shadow: none !important;
}

html[data-wa-theme="light"] .wa-spark-kicker {
  background: rgba(255, 255, 255, 0.82);
  border-color: rgba(148, 163, 184, 0.28);
  color: #0f172a;
}

html[data-wa-theme="light"] .wa-spark-hud span {
  background: rgba(255, 255, 255, 0.86);
  border-color: rgba(148, 163, 184, 0.24);
  color: #0f172a;
}

@media (max-width: 1180px) {
  .wa-spatial-shell,
  .wa-spark-canvas {
    height: 500px;
    min-height: 500px;
  }
}

@media (max-width: 760px) {
  .wa-spatial-shell,
  .wa-spark-canvas {
    height: 420px;
    min-height: 420px;
  }

  .wa-spark-overlay {
    padding: 16px;
  }

  .wa-spark-title {
    font-size: 22px;
  }

  .wa-spark-copy {
    font-size: 13px;
  }
}

/* WorldFoundry responsive balance pass */
:root {
  --wa-shell-edge: clamp(14px, 2vw, 36px);
  --wa-rail-width: clamp(288px, 24vw, 356px);
  --wa-rail-peek: 26px;
  --wa-stage-shell-max: min(980px, calc(100vw - var(--wa-shell-edge) - var(--wa-shell-edge)));
  --wa-stage-shell-open: min(900px, calc(100vw - var(--wa-rail-width) - 76px));
  --wa-panel-pad-x: clamp(12px, 1.4vw, 20px);
  --wa-panel-pad-top: clamp(12px, 1.5vw, 18px);
  --wa-panel-pad-bottom: clamp(212px, 24vw, 268px);
  --wa-media-min: clamp(320px, 54vh, 510px);
  --wa-tray-size: clamp(48px, 4.8vw, 64px);
  --wa-tray-gap: clamp(8px, 1vw, 12px);
}

.wa-site-nav {
  gap: 16px;
  flex-wrap: wrap;
  align-items: center;
}

.wa-site-brand {
  min-width: 0;
  max-width: min(100%, calc(100vw - 144px));
  gap: clamp(8px, 1vw, 10px);
  padding-right: clamp(10px, 1.2vw, 12px);
}

.wa-site-brand-mark {
  width: clamp(72px, 8vw, 88px);
  height: clamp(38px, 4.2vw, 44px);
}

.wa-site-nav-left {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  font-size: clamp(14px, 1.2vw, 15px);
}

.wa-site-nav-right {
  flex: 0 0 auto;
  gap: clamp(8px, 1vw, 10px);
}

.wa-site-nav-icon {
  width: clamp(36px, 3vw, 40px);
  height: clamp(36px, 3vw, 40px);
}

.wa-main-grid {
  padding: 0 clamp(12px, 2vw, 24px) clamp(14px, 2vw, 22px) !important;
}

.wa-left-rail,
.wa-right-rail {
  top: 78px;
  width: var(--wa-rail-width) !important;
  min-width: var(--wa-rail-width) !important;
  max-width: var(--wa-rail-width) !important;
  max-height: calc(100vh - 96px);
  overflow-y: auto;
  overflow-x: visible;
  opacity: 0.18;
}

.wa-left-rail {
  transform: translateX(calc(-100% + var(--wa-rail-peek)));
}

.wa-right-rail {
  transform: translateX(calc(100% - var(--wa-rail-peek)));
}

.wa-left-rail::after,
.wa-right-rail::before {
  width: 14px;
  height: clamp(128px, 14vw, 148px);
}

.wa-left-rail:hover,
.wa-left-rail:focus-within,
.wa-right-rail:hover,
.wa-right-rail:focus-within {
  transform: translateX(0);
}

.wa-left-rail .wa-panel-block,
.wa-right-rail .wa-panel-block {
  padding: clamp(14px, 1.4vw, 16px) !important;
  max-height: min(680px, calc(100vh - 110px));
  transition: opacity 180ms ease;
}

.wa-right-rail:not(:hover):not(:focus-within) .wa-panel-block {
  opacity: 0;
  pointer-events: none;
}

.wa-left-rail:hover .wa-panel-block,
.wa-left-rail:focus-within .wa-panel-block,
.wa-right-rail:hover .wa-panel-block,
.wa-right-rail:focus-within .wa-panel-block {
  opacity: 1;
  pointer-events: auto;
}

.wa-panel-title {
  font-size: clamp(10px, 0.9vw, 11px);
}

.wa-left-rail .wa-panel-copy,
.wa-right-rail .wa-panel-copy {
  font-size: clamp(12px, 1vw, 13px);
}

.wa-atlas-title-row {
  flex-wrap: wrap;
  align-items: flex-start;
}

.wa-atlas-title-row h4 {
  font-size: clamp(18px, 2vw, 22px);
}

.wa-atlas-metrics {
  grid-template-columns: repeat(auto-fit, minmax(88px, 1fr));
}

.wa-profile h3 {
  font-size: clamp(18px, 2vw, 20px);
}

.wa-profile p,
.wa-summary-subtitle,
.wa-summary-lines div {
  font-size: clamp(12px, 1vw, 13px);
}

.wa-run-row,
.wa-preset-row,
.wa-live-grid {
  flex-wrap: wrap;
}

.wa-run-row > div,
.wa-preset-row > div,
.wa-live-grid > div {
  min-width: 0 !important;
  flex: 1 1 120px !important;
}

.wa-preset-row button,
.wa-live-grid button {
  width: 100% !important;
}

.wa-stage-col {
  flex: 0 1 var(--wa-stage-shell-max) !important;
  width: var(--wa-stage-shell-max) !important;
  max-width: var(--wa-stage-shell-max) !important;
  min-width: 0 !important;
  transition: transform 180ms ease, width 180ms ease, max-width 180ms ease;
}

.wa-stage-col > .block,
.wa-stage-col > .form {
  width: var(--wa-stage-shell-max) !important;
  max-width: var(--wa-stage-shell-max) !important;
}

.wa-preview-panel {
  width: var(--wa-stage-shell-max) !important;
  max-width: var(--wa-stage-shell-max) !important;
  min-height: clamp(600px, calc(100vh - 132px), 720px);
  padding: var(--wa-panel-pad-top) var(--wa-panel-pad-x) clamp(14px, 1.8vw, 18px) !important;
}

.wa-preview-panel::before,
.wa-stage-empty {
  inset: var(--wa-panel-pad-top) var(--wa-panel-pad-x) var(--wa-panel-pad-bottom);
  border-radius: clamp(18px, 1.8vw, 20px);
}

#wa-main-preview-video,
#wa-main-preview-image,
.wa-preview-panel .tabitem,
#wa-main-preview-video .wrap,
#wa-main-preview-image .wrap {
  min-height: var(--wa-media-min) !important;
}

#wa-main-preview-video video,
#wa-main-preview-image img,
.wa-preview-panel model-viewer {
  max-height: calc(100vh - clamp(250px, 30vw, 300px));
  border-radius: clamp(18px, 1.8vw, 20px) !important;
}

.wa-main-grid:has(.wa-left-rail:hover) .wa-stage-col,
.wa-main-grid:has(.wa-left-rail:focus-within) .wa-stage-col {
  transform: translateX(calc((var(--wa-rail-width) - var(--wa-rail-peek)) / 2));
}

.wa-main-grid:has(.wa-right-rail:hover) .wa-stage-col,
.wa-main-grid:has(.wa-right-rail:focus-within) .wa-stage-col {
  transform: translateX(calc((var(--wa-rail-peek) - var(--wa-rail-width)) / 2));
}

.wa-main-grid:has(.wa-left-rail:hover) .wa-preview-panel,
.wa-main-grid:has(.wa-left-rail:focus-within) .wa-preview-panel,
.wa-main-grid:has(.wa-right-rail:hover) .wa-preview-panel,
.wa-main-grid:has(.wa-right-rail:focus-within) .wa-preview-panel,
.wa-main-grid:has(.wa-left-rail:hover) .wa-stage-col > .block,
.wa-main-grid:has(.wa-left-rail:focus-within) .wa-stage-col > .block,
.wa-main-grid:has(.wa-right-rail:hover) .wa-stage-col > .block,
.wa-main-grid:has(.wa-right-rail:focus-within) .wa-stage-col > .block,
.wa-main-grid:has(.wa-left-rail:hover) .wa-stage-col > .form,
.wa-main-grid:has(.wa-left-rail:focus-within) .wa-stage-col > .form,
.wa-main-grid:has(.wa-right-rail:hover) .wa-stage-col > .form,
.wa-main-grid:has(.wa-right-rail:focus-within) .wa-stage-col > .form {
  width: var(--wa-stage-shell-open) !important;
  max-width: var(--wa-stage-shell-open) !important;
}

.wa-player-footer {
  left: var(--wa-panel-pad-x);
  right: var(--wa-panel-pad-x);
  bottom: clamp(198px, 22vw, 214px);
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
  align-items: center;
  gap: clamp(8px, 1.2vw, 14px);
  font-size: clamp(9px, 0.9vw, 10px);
}

.wa-player-footer-center {
  position: static;
  transform: none;
  justify-self: center;
  max-width: min(38vw, 260px);
}

.wa-player-footer-right {
  justify-self: end;
}

.wa-control-dock {
  display: flex !important;
  flex-wrap: nowrap;
  justify-content: center;
  align-items: center;
  position: relative;
  left: 50%;
  transform: translateX(-50%);
  width: max-content !important;
  max-width: calc(100% - 12px);
  margin: 18px 0 0 !important;
  gap: clamp(8px, 0.9vw, 10px);
}

.wa-control-dock > div,
.wa-control-dock > .wrap {
  flex: 0 0 auto !important;
  width: fit-content !important;
  min-width: 0 !important;
}

.wa-control-dock button {
  width: auto !important;
  min-width: clamp(86px, 8vw, 108px) !important;
  min-height: clamp(30px, 3vw, 34px) !important;
  padding: 0 clamp(10px, 1.1vw, 12px) !important;
  font-size: clamp(9px, 0.9vw, 10px) !important;
}

.wa-world-tray-shell {
  width: min(100%, 620px);
  max-width: calc(100% - 12px);
  margin: 18px auto 0 !important;
}

.wa-world-tray {
  justify-content: flex-start;
  flex-wrap: nowrap;
  overflow-x: auto;
  overflow-y: hidden;
  scrollbar-width: thin;
  gap: var(--wa-tray-gap);
  min-height: auto;
  padding: clamp(10px, 1.1vw, 12px);
}

.wa-tray-item {
  width: var(--wa-tray-size);
  height: var(--wa-tray-size);
  border-radius: clamp(10px, 1vw, 12px);
  flex: 0 0 auto;
}

.wa-joystick-dock {
  gap: clamp(14px, 2vw, 18px);
  min-height: clamp(110px, 12vw, 128px);
  height: clamp(110px, 12vw, 128px);
  padding: 10px clamp(8px, 1vw, 12px) 0;
}

.wa-preview-panel.wa-joystick-open #wa-main-preview-video,
.wa-preview-panel.wa-joystick-open #wa-main-preview-image,
.wa-preview-panel.wa-joystick-open .tabitem,
.wa-preview-panel.wa-joystick-open #wa-main-preview-video .wrap,
.wa-preview-panel.wa-joystick-open #wa-main-preview-image .wrap {
  min-height: var(--wa-media-min) !important;
}

.wa-joystick-dock-copy {
  max-width: min(38vw, 260px);
  bottom: 6px;
}

.wa-joystick-dock .wa-stick {
  width: clamp(70px, 6vw, 82px);
  height: clamp(70px, 6vw, 82px);
}

.wa-spatial-shell,
.wa-spark-canvas {
  height: clamp(360px, 52vh, 560px);
  min-height: clamp(360px, 52vh, 560px);
}

.wa-spark-copy-stack {
  max-width: min(62%, 440px);
}

@media (max-width: 1280px) {
  :root {
    --wa-rail-width: clamp(272px, 28vw, 320px);
  }
}

@media (max-width: 1120px) {
  :root {
    --wa-rail-width: clamp(248px, 30vw, 292px);
  }

  .wa-left-rail .wa-panel-block,
  .wa-right-rail .wa-panel-block {
    max-height: min(620px, calc(100vh - 102px));
  }
}

@media (max-width: 980px) {
  .wa-main-grid {
    min-height: auto;
    flex-direction: column;
    align-items: stretch;
    gap: 16px;
  }

  .wa-stage-col,
  .wa-left-rail,
  .wa-right-rail {
    width: 100% !important;
    min-width: 0 !important;
    max-width: none !important;
    margin: 0 auto !important;
  }

  .wa-stage-col {
    order: 1;
    transform: none !important;
  }

  .wa-left-rail,
  .wa-right-rail {
    position: static !important;
    top: auto;
    left: auto;
    right: auto;
    transform: none !important;
    opacity: 1 !important;
    flex: 1 1 auto !important;
  }

  .wa-left-rail {
    order: 2;
  }

  .wa-right-rail {
    order: 3;
  }

  .wa-left-rail::after,
  .wa-right-rail::before {
    display: none;
  }

  .wa-left-rail .wa-panel-block,
  .wa-right-rail .wa-panel-block {
    max-height: none;
  }

  .wa-stage-col > .block,
  .wa-stage-col > .form,
  .wa-preview-panel {
    width: min(100%, 920px) !important;
    max-width: min(100%, 920px) !important;
  }

  .wa-preview-panel {
    min-height: clamp(560px, calc(100vh - 144px), 680px);
    border-radius: clamp(24px, 3vw, 32px) !important;
  }

  .wa-main-grid:has(.wa-left-rail:hover) .wa-stage-col,
  .wa-main-grid:has(.wa-left-rail:focus-within) .wa-stage-col,
  .wa-main-grid:has(.wa-right-rail:hover) .wa-stage-col,
  .wa-main-grid:has(.wa-right-rail:focus-within) .wa-stage-col {
    transform: none !important;
  }

  .wa-main-grid:has(.wa-left-rail:hover) .wa-preview-panel,
  .wa-main-grid:has(.wa-left-rail:focus-within) .wa-preview-panel,
  .wa-main-grid:has(.wa-right-rail:hover) .wa-preview-panel,
  .wa-main-grid:has(.wa-right-rail:focus-within) .wa-preview-panel,
  .wa-main-grid:has(.wa-left-rail:hover) .wa-stage-col > .block,
  .wa-main-grid:has(.wa-left-rail:focus-within) .wa-stage-col > .block,
  .wa-main-grid:has(.wa-right-rail:hover) .wa-stage-col > .block,
  .wa-main-grid:has(.wa-right-rail:focus-within) .wa-stage-col > .block,
  .wa-main-grid:has(.wa-left-rail:hover) .wa-stage-col > .form,
  .wa-main-grid:has(.wa-left-rail:focus-within) .wa-stage-col > .form,
  .wa-main-grid:has(.wa-right-rail:hover) .wa-stage-col > .form,
  .wa-main-grid:has(.wa-right-rail:focus-within) .wa-stage-col > .form {
    width: min(100%, 920px) !important;
    max-width: min(100%, 920px) !important;
  }

  .wa-atlas-metrics {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 760px) {
  .wa-site-nav {
    width: 100%;
    padding: 8px 0 4px;
    align-items: flex-start;
  }

  .wa-site-brand {
    max-width: calc(100vw - 112px);
  }

  .wa-site-nav-right {
    margin-left: auto;
  }

  .wa-preview-panel {
    --wa-panel-pad-bottom: 188px;
    min-height: clamp(520px, calc(100vh - 120px), 620px);
  }

  .wa-player-footer {
    bottom: 174px;
  }

  .wa-player-footer-center {
    max-width: 140px;
  }

  .wa-stage-empty {
    padding: 28px 20px 18px;
  }

  .wa-control-dock {
    left: 50%;
    transform: translateX(-50%);
    width: max-content !important;
    max-width: calc(100% - 12px);
    margin-top: 10px !important;
  }

  .wa-world-tray {
    gap: 6px;
  }

  .wa-world-tray-shell {
    margin-top: 16px !important;
  }

  .wa-tray-item {
    width: 46px;
    height: 46px;
  }

  .wa-joystick-dock {
    flex-wrap: wrap;
    justify-content: center;
    min-height: 120px;
    height: auto;
  }

  .wa-joystick-dock-copy {
    position: static;
    transform: none;
    order: 3;
    max-width: 100%;
    text-align: center;
  }

  .wa-spark-overlay {
    padding: 14px;
  }

  .wa-spark-copy-stack {
    max-width: 100%;
    gap: 6px;
  }

  .wa-spark-title {
    font-size: clamp(20px, 5vw, 24px);
  }

  .wa-spark-copy {
    font-size: 12.5px;
    line-height: 1.45;
  }
}

@media (max-width: 460px) {
  .wa-site-brand {
    max-width: calc(100vw - 96px);
  }

  .wa-site-brand-mark {
    width: 70px;
    height: 36px;
  }

  .wa-site-nav-left {
    font-size: 13px;
  }

  .wa-player-footer {
    grid-template-columns: minmax(0, 1fr) auto;
    row-gap: 4px;
  }

  .wa-player-footer-center {
    grid-column: 1 / -1;
    justify-self: center;
    max-width: 100%;
  }

  .wa-control-dock > div,
  .wa-control-dock > .wrap {
    flex: 0 0 auto !important;
  }

  .wa-control-dock button {
    min-width: 0 !important;
    padding: 0 10px !important;
    font-size: 8.5px !important;
  }
}

.wa-dataframe.hide-container,
.wa-dataframe.hide-container *,
.wa-left-rail .wa-dataframe.hide-container,
.wa-left-rail .wa-dataframe.hide-container *,
.wa-right-rail .wa-dataframe.hide-container,
.wa-right-rail .wa-dataframe.hide-container * {
  display: none !important;
  visibility: hidden !important;
  opacity: 0 !important;
  pointer-events: none !important;
}

.wa-side-model-switcher {
  margin-top: 2px;
  padding: 10px 12px !important;
  border-radius: 18px !important;
  border: 1px solid var(--wa-line) !important;
  background: var(--wa-nav-bg) !important;
  box-shadow: 0 10px 28px rgba(15, 23, 42, 0.1);
}

.wa-side-model-switcher label {
  color: var(--wa-ink-strong) !important;
  font: 600 9px/1.2 "IBM Plex Mono", monospace;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.wa-side-model-switcher .wrap,
.wa-side-model-switcher .secondary-wrap,
.wa-side-model-switcher .single-select {
  border-radius: 14px !important;
}

.wa-side-model-switcher input,
.wa-side-model-switcher button,
.wa-side-model-switcher span {
  color: var(--wa-ink-strong) !important;
}

.wa-left-rail,
.wa-right-rail {
  overflow-x: hidden !important;
}

.wa-left-rail .wa-panel-copy,
.wa-right-rail .wa-panel-copy {
  color: var(--wa-ink-soft) !important;
}

.wa-player-footer {
  bottom: clamp(202px, 22vw, 220px);
  font-size: clamp(10px, 0.95vw, 11px);
}

.wa-player-footer-left {
  gap: 10px;
  font-weight: 600;
}

.wa-player-footer-center {
  max-width: min(42vw, 320px);
  color: var(--wa-ink-strong);
  font-size: clamp(13px, 1.15vw, 15px);
  font-style: normal;
  font-weight: 700;
  letter-spacing: -0.03em;
}

.wa-player-footer-right {
  color: var(--wa-ink-soft);
  font-weight: 600;
}

.wa-player-dot {
  width: 8px;
  height: 8px;
}

.wa-joystick-dock {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: clamp(10px, 1.4vw, 16px);
  min-height: 188px;
  height: auto;
  margin-top: 8px;
  padding: 8px clamp(14px, 2vw, 28px) 22px;
  overflow: hidden;
}

.wa-joystick-dock-copy {
  display: grid;
  gap: 8px;
  position: static;
  order: 2;
  flex: 0 0 clamp(216px, 24vw, 280px);
  transform: none;
  min-width: clamp(216px, 24vw, 280px);
  max-width: clamp(216px, 24vw, 280px);
  padding-inline: 6px;
  text-align: center;
}

.wa-joystick-dock .wa-floating-stick {
  position: relative;
  top: auto;
  flex: 1 1 0;
  width: clamp(148px, 17vw, 220px);
  min-width: clamp(148px, 17vw, 220px);
  max-width: none;
  justify-content: center;
  left: auto !important;
  right: auto !important;
}

.wa-joystick-dock .wa-floating-stick-left {
  order: 1;
  align-items: flex-start;
  text-align: left;
}

.wa-joystick-dock .wa-floating-stick-right {
  order: 3;
  align-items: flex-end;
  text-align: right;
}

.wa-joystick-dock .wa-stick-shell {
  width: 100%;
  gap: 8px;
  padding-top: 0;
}

.wa-joystick-dock .wa-stick-label {
  width: 100%;
  font-size: 10px;
  letter-spacing: 0.16em;
}

.wa-joystick-dock .wa-stick {
  width: clamp(102px, 8.1vw, 118px);
  height: clamp(102px, 8.1vw, 118px);
}

.wa-joystick-dock .wa-stick-ring {
  inset: 11px;
}

.wa-joystick-dock .wa-stick-thumb {
  width: 32px;
  height: 32px;
}

html[data-wa-theme="light"] .wa-side-model-switcher {
  background: rgba(255, 255, 255, 0.94) !important;
  border-color: rgba(15, 23, 42, 0.08) !important;
}

html[data-wa-theme="light"] .wa-left-rail .wa-panel-copy,
html[data-wa-theme="light"] .wa-right-rail .wa-panel-copy {
  color: #334155 !important;
}

@media (max-width: 900px) {
  .wa-side-model-switcher {
    width: 100%;
  }

  .wa-joystick-dock {
    min-height: 184px;
    padding-inline: 12px;
  }

  .wa-joystick-dock .wa-floating-stick {
    width: clamp(132px, 28vw, 172px);
  }

  .wa-joystick-dock .wa-stick-shell {
    gap: 7px;
  }

  .wa-joystick-dock-copy {
    flex-basis: min(240px, 44vw);
    min-width: min(240px, 44vw);
    max-width: min(240px, 44vw);
  }
}

@media (max-width: 760px) {
  .wa-world-tray-shell {
    max-width: calc(100% - 6px);
  }

  .wa-joystick-dock {
    min-height: 196px;
    flex-wrap: wrap;
    justify-content: center;
    align-items: center;
  }

  .wa-joystick-dock .wa-floating-stick {
    width: min(44vw, 164px);
  }

  .wa-joystick-dock-copy {
    order: 3;
    flex: 1 1 100%;
    min-width: 0;
    max-width: min(220px, 100%);
  }

  .wa-joystick-dock .wa-stick {
    width: 90px;
    height: 90px;
  }
}

/* Restore manual interactive generation controls. */
.wa-control-dock .wa-action-run,
.wa-control-dock .wa-action-run button,
.wa-control-dock .wa-action-step,
.wa-control-dock .wa-action-step button {
  display: inline-flex !important;
}

.wa-control-dock .hidden,
.wa-control-dock .hidden button,
.wa-control-dock .wa-action-run.hidden,
.wa-control-dock .wa-action-run.hidden button,
.wa-control-dock .wa-action-step.hidden,
.wa-control-dock .wa-action-step.hidden button {
  display: none !important;
}

.wa-control-dock .wa-action-run,
.wa-control-dock .wa-action-run button {
  background: linear-gradient(180deg, var(--wa-accent), var(--wa-accent-strong)) !important;
  color: #ffffff !important;
  border-color: transparent !important;
}

.wa-control-dock .wa-action-step,
.wa-control-dock .wa-action-step button {
  background: rgba(15, 23, 42, 0.88) !important;
  color: #f8fafc !important;
}

html[data-wa-theme="light"] .wa-control-dock .wa-action-step,
html[data-wa-theme="light"] .wa-control-dock .wa-action-step button {
  background: rgba(255, 255, 255, 0.92) !important;
  color: #111827 !important;
}

.wa-live-dock {
  position: static !important;
  left: auto !important;
  top: auto !important;
  width: auto !important;
  height: auto !important;
  overflow: visible !important;
  margin-top: 18px !important;
}

.wa-live-caption,
.wa-live-group-title,
.wa-run-overview {
  display: block !important;
}

.wa-live-grid {
  display: grid !important;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px !important;
}

.wa-live-grid > *,
.wa-live-grid .wrap {
  width: 100% !important;
}

.wa-live-grid button {
  width: 100% !important;
  min-height: 38px !important;
  font: 500 11px/1.25 "IBM Plex Mono", monospace !important;
}

html[data-wa-viewport="balanced"] {
  --wa-rail-width: clamp(272px, 28vw, 320px);
}

html[data-wa-viewport="compact"] {
  --wa-rail-width: clamp(248px, 30vw, 292px);
}

html[data-wa-viewport="stacked"] .wa-main-grid,
html[data-wa-viewport="phone"] .wa-main-grid,
html[data-wa-viewport="narrow"] .wa-main-grid {
  min-height: auto;
  flex-direction: column !important;
  align-items: stretch !important;
  gap: 16px;
}

html[data-wa-viewport="stacked"] .wa-stage-col,
html[data-wa-viewport="stacked"] .wa-left-rail,
html[data-wa-viewport="stacked"] .wa-right-rail,
html[data-wa-viewport="phone"] .wa-stage-col,
html[data-wa-viewport="phone"] .wa-left-rail,
html[data-wa-viewport="phone"] .wa-right-rail,
html[data-wa-viewport="narrow"] .wa-stage-col,
html[data-wa-viewport="narrow"] .wa-left-rail,
html[data-wa-viewport="narrow"] .wa-right-rail {
  width: 100% !important;
  min-width: 0 !important;
  max-width: none !important;
  margin: 0 auto !important;
}

html[data-wa-viewport="stacked"] .wa-stage-col,
html[data-wa-viewport="phone"] .wa-stage-col,
html[data-wa-viewport="narrow"] .wa-stage-col {
  order: 1;
  flex: 1 1 auto !important;
  transform: none !important;
}

html[data-wa-viewport="stacked"] .wa-left-rail,
html[data-wa-viewport="stacked"] .wa-right-rail,
html[data-wa-viewport="phone"] .wa-left-rail,
html[data-wa-viewport="phone"] .wa-right-rail,
html[data-wa-viewport="narrow"] .wa-left-rail,
html[data-wa-viewport="narrow"] .wa-right-rail {
  position: static !important;
  top: auto !important;
  left: auto !important;
  right: auto !important;
  transform: none !important;
  opacity: 1 !important;
  flex: 1 1 auto !important;
  max-height: none !important;
}

html[data-wa-viewport="stacked"] .wa-left-rail {
  order: 2;
}

html[data-wa-viewport="stacked"] .wa-right-rail {
  order: 3;
}

html[data-wa-viewport="phone"] .wa-left-rail {
  order: 2;
}

html[data-wa-viewport="phone"] .wa-right-rail {
  order: 3;
}

html[data-wa-viewport="narrow"] .wa-left-rail {
  order: 2;
}

html[data-wa-viewport="narrow"] .wa-right-rail {
  order: 3;
}

html[data-wa-viewport="stacked"] .wa-left-rail::after,
html[data-wa-viewport="stacked"] .wa-right-rail::before,
html[data-wa-viewport="phone"] .wa-left-rail::after,
html[data-wa-viewport="phone"] .wa-right-rail::before,
html[data-wa-viewport="narrow"] .wa-left-rail::after,
html[data-wa-viewport="narrow"] .wa-right-rail::before {
  display: none !important;
}

html[data-wa-viewport="stacked"] .wa-left-rail .wa-panel-block,
html[data-wa-viewport="stacked"] .wa-right-rail .wa-panel-block,
html[data-wa-viewport="phone"] .wa-left-rail .wa-panel-block,
html[data-wa-viewport="phone"] .wa-right-rail .wa-panel-block,
html[data-wa-viewport="narrow"] .wa-left-rail .wa-panel-block,
html[data-wa-viewport="narrow"] .wa-right-rail .wa-panel-block {
  max-height: none !important;
  opacity: 1 !important;
  pointer-events: auto !important;
}

html[data-wa-viewport="stacked"] .wa-right-rail:not(:hover):not(:focus-within) .wa-panel-block,
html[data-wa-viewport="phone"] .wa-right-rail:not(:hover):not(:focus-within) .wa-panel-block,
html[data-wa-viewport="narrow"] .wa-right-rail:not(:hover):not(:focus-within) .wa-panel-block {
  opacity: 1 !important;
  pointer-events: auto !important;
}

html[data-wa-viewport="stacked"] .wa-stage-col > .block,
html[data-wa-viewport="stacked"] .wa-stage-col > .form,
html[data-wa-viewport="stacked"] .wa-preview-panel,
html[data-wa-viewport="phone"] .wa-stage-col > .block,
html[data-wa-viewport="phone"] .wa-stage-col > .form,
html[data-wa-viewport="phone"] .wa-preview-panel,
html[data-wa-viewport="narrow"] .wa-stage-col > .block,
html[data-wa-viewport="narrow"] .wa-stage-col > .form,
html[data-wa-viewport="narrow"] .wa-preview-panel {
  width: min(100%, 920px) !important;
  max-width: min(100%, 920px) !important;
}

html[data-wa-viewport="stacked"] .wa-preview-panel,
html[data-wa-viewport="phone"] .wa-preview-panel,
html[data-wa-viewport="narrow"] .wa-preview-panel {
  min-height: clamp(560px, calc(100vh - 144px), 680px);
  border-radius: clamp(24px, 3vw, 32px) !important;
}

html[data-wa-viewport="stacked"] .wa-main-grid:has(.wa-left-rail:hover) .wa-stage-col,
html[data-wa-viewport="stacked"] .wa-main-grid:has(.wa-left-rail:focus-within) .wa-stage-col,
html[data-wa-viewport="stacked"] .wa-main-grid:has(.wa-right-rail:hover) .wa-stage-col,
html[data-wa-viewport="stacked"] .wa-main-grid:has(.wa-right-rail:focus-within) .wa-stage-col,
html[data-wa-viewport="phone"] .wa-main-grid:has(.wa-left-rail:hover) .wa-stage-col,
html[data-wa-viewport="phone"] .wa-main-grid:has(.wa-left-rail:focus-within) .wa-stage-col,
html[data-wa-viewport="phone"] .wa-main-grid:has(.wa-right-rail:hover) .wa-stage-col,
html[data-wa-viewport="phone"] .wa-main-grid:has(.wa-right-rail:focus-within) .wa-stage-col,
html[data-wa-viewport="narrow"] .wa-main-grid:has(.wa-left-rail:hover) .wa-stage-col,
html[data-wa-viewport="narrow"] .wa-main-grid:has(.wa-left-rail:focus-within) .wa-stage-col,
html[data-wa-viewport="narrow"] .wa-main-grid:has(.wa-right-rail:hover) .wa-stage-col,
html[data-wa-viewport="narrow"] .wa-main-grid:has(.wa-right-rail:focus-within) .wa-stage-col {
  transform: none !important;
}

html[data-wa-viewport="stacked"] .wa-main-grid:has(.wa-left-rail:hover) .wa-preview-panel,
html[data-wa-viewport="stacked"] .wa-main-grid:has(.wa-left-rail:focus-within) .wa-preview-panel,
html[data-wa-viewport="stacked"] .wa-main-grid:has(.wa-right-rail:hover) .wa-preview-panel,
html[data-wa-viewport="stacked"] .wa-main-grid:has(.wa-right-rail:focus-within) .wa-preview-panel,
html[data-wa-viewport="stacked"] .wa-main-grid:has(.wa-left-rail:hover) .wa-stage-col > .block,
html[data-wa-viewport="stacked"] .wa-main-grid:has(.wa-left-rail:focus-within) .wa-stage-col > .block,
html[data-wa-viewport="stacked"] .wa-main-grid:has(.wa-right-rail:hover) .wa-stage-col > .block,
html[data-wa-viewport="stacked"] .wa-main-grid:has(.wa-right-rail:focus-within) .wa-stage-col > .block,
html[data-wa-viewport="stacked"] .wa-main-grid:has(.wa-left-rail:hover) .wa-stage-col > .form,
html[data-wa-viewport="stacked"] .wa-main-grid:has(.wa-left-rail:focus-within) .wa-stage-col > .form,
html[data-wa-viewport="stacked"] .wa-main-grid:has(.wa-right-rail:hover) .wa-stage-col > .form,
html[data-wa-viewport="stacked"] .wa-main-grid:has(.wa-right-rail:focus-within) .wa-stage-col > .form,
html[data-wa-viewport="phone"] .wa-main-grid:has(.wa-left-rail:hover) .wa-preview-panel,
html[data-wa-viewport="phone"] .wa-main-grid:has(.wa-left-rail:focus-within) .wa-preview-panel,
html[data-wa-viewport="phone"] .wa-main-grid:has(.wa-right-rail:hover) .wa-preview-panel,
html[data-wa-viewport="phone"] .wa-main-grid:has(.wa-right-rail:focus-within) .wa-preview-panel,
html[data-wa-viewport="phone"] .wa-main-grid:has(.wa-left-rail:hover) .wa-stage-col > .block,
html[data-wa-viewport="phone"] .wa-main-grid:has(.wa-left-rail:focus-within) .wa-stage-col > .block,
html[data-wa-viewport="phone"] .wa-main-grid:has(.wa-right-rail:hover) .wa-stage-col > .block,
html[data-wa-viewport="phone"] .wa-main-grid:has(.wa-right-rail:focus-within) .wa-stage-col > .block,
html[data-wa-viewport="phone"] .wa-main-grid:has(.wa-left-rail:hover) .wa-stage-col > .form,
html[data-wa-viewport="phone"] .wa-main-grid:has(.wa-left-rail:focus-within) .wa-stage-col > .form,
html[data-wa-viewport="phone"] .wa-main-grid:has(.wa-right-rail:hover) .wa-stage-col > .form,
html[data-wa-viewport="phone"] .wa-main-grid:has(.wa-right-rail:focus-within) .wa-stage-col > .form,
html[data-wa-viewport="narrow"] .wa-main-grid:has(.wa-left-rail:hover) .wa-preview-panel,
html[data-wa-viewport="narrow"] .wa-main-grid:has(.wa-left-rail:focus-within) .wa-preview-panel,
html[data-wa-viewport="narrow"] .wa-main-grid:has(.wa-right-rail:hover) .wa-preview-panel,
html[data-wa-viewport="narrow"] .wa-main-grid:has(.wa-right-rail:focus-within) .wa-preview-panel,
html[data-wa-viewport="narrow"] .wa-main-grid:has(.wa-left-rail:hover) .wa-stage-col > .block,
html[data-wa-viewport="narrow"] .wa-main-grid:has(.wa-left-rail:focus-within) .wa-stage-col > .block,
html[data-wa-viewport="narrow"] .wa-main-grid:has(.wa-right-rail:hover) .wa-stage-col > .block,
html[data-wa-viewport="narrow"] .wa-main-grid:has(.wa-right-rail:focus-within) .wa-stage-col > .block,
html[data-wa-viewport="narrow"] .wa-main-grid:has(.wa-left-rail:hover) .wa-stage-col > .form,
html[data-wa-viewport="narrow"] .wa-main-grid:has(.wa-left-rail:focus-within) .wa-stage-col > .form,
html[data-wa-viewport="narrow"] .wa-main-grid:has(.wa-right-rail:hover) .wa-stage-col > .form,
html[data-wa-viewport="narrow"] .wa-main-grid:has(.wa-right-rail:focus-within) .wa-stage-col > .form {
  width: min(100%, 920px) !important;
  max-width: min(100%, 920px) !important;
}

html[data-wa-viewport="stacked"] .wa-control-dock,
html[data-wa-viewport="phone"] .wa-control-dock,
html[data-wa-viewport="narrow"] .wa-control-dock {
  left: auto;
  transform: none;
  width: 100% !important;
  flex-wrap: wrap;
  justify-content: center;
}

html[data-wa-viewport="phone"] .wa-world-tray-shell,
html[data-wa-viewport="narrow"] .wa-world-tray-shell {
  max-width: calc(100% - 6px);
}

html[data-wa-viewport="phone"] .wa-joystick-dock,
html[data-wa-viewport="narrow"] .wa-joystick-dock {
  min-height: 196px;
  flex-wrap: wrap;
  justify-content: center;
  align-items: center;
}

html[data-wa-viewport="phone"] .wa-joystick-dock .wa-floating-stick,
html[data-wa-viewport="narrow"] .wa-joystick-dock .wa-floating-stick {
  width: min(44vw, 164px);
}

html[data-wa-viewport="phone"] .wa-joystick-dock-copy,
html[data-wa-viewport="narrow"] .wa-joystick-dock-copy {
  order: 3;
  flex: 1 1 100%;
  min-width: 0;
  max-width: min(220px, 100%);
}

html[data-wa-viewport="phone"] .wa-joystick-dock .wa-stick,
html[data-wa-viewport="narrow"] .wa-joystick-dock .wa-stick {
  width: 90px;
  height: 90px;
}

html[data-wa-viewport="narrow"] .wa-site-brand {
  max-width: calc(100vw - 96px);
}

html[data-wa-viewport="narrow"] .wa-site-brand-mark {
  width: 70px;
  height: 36px;
}

html[data-wa-viewport="narrow"] .wa-site-nav-left {
  font-size: 13px;
}

html[data-wa-viewport="narrow"] .wa-player-footer {
  grid-template-columns: minmax(0, 1fr) auto;
  row-gap: 4px;
}

html[data-wa-viewport="narrow"] .wa-player-footer-center {
  grid-column: 1 / -1;
  justify-self: center;
  max-width: 100%;
}

html[data-wa-viewport="narrow"] .wa-control-dock > div,
html[data-wa-viewport="narrow"] .wa-control-dock > .wrap {
  flex: 0 0 auto !important;
}

html[data-wa-viewport="narrow"] .wa-control-dock button {
  min-width: 0 !important;
  padding: 0 10px !important;
  font-size: 8.5px !important;
}

.wa-main-grid[data-wa-viewport="balanced"] {
  --wa-rail-width: clamp(272px, 28vw, 320px);
}

.wa-main-grid[data-wa-viewport="compact"] {
  --wa-rail-width: clamp(248px, 30vw, 292px);
}

.wa-main-grid[data-wa-viewport="stacked"],
.wa-main-grid[data-wa-viewport="phone"],
.wa-main-grid[data-wa-viewport="narrow"] {
  min-height: auto;
  flex-direction: column !important;
  align-items: stretch !important;
  gap: 16px;
}

.wa-main-grid[data-wa-viewport="stacked"] .wa-stage-col,
.wa-main-grid[data-wa-viewport="stacked"] .wa-left-rail,
.wa-main-grid[data-wa-viewport="stacked"] .wa-right-rail,
.wa-main-grid[data-wa-viewport="phone"] .wa-stage-col,
.wa-main-grid[data-wa-viewport="phone"] .wa-left-rail,
.wa-main-grid[data-wa-viewport="phone"] .wa-right-rail,
.wa-main-grid[data-wa-viewport="narrow"] .wa-stage-col,
.wa-main-grid[data-wa-viewport="narrow"] .wa-left-rail,
.wa-main-grid[data-wa-viewport="narrow"] .wa-right-rail {
  width: 100% !important;
  min-width: 0 !important;
  max-width: none !important;
  margin: 0 auto !important;
}

.wa-main-grid[data-wa-viewport="stacked"] .wa-stage-col,
.wa-main-grid[data-wa-viewport="phone"] .wa-stage-col,
.wa-main-grid[data-wa-viewport="narrow"] .wa-stage-col {
  order: 1;
  flex: 1 1 auto !important;
  transform: none !important;
}

.wa-main-grid[data-wa-viewport="stacked"] .wa-left-rail,
.wa-main-grid[data-wa-viewport="stacked"] .wa-right-rail,
.wa-main-grid[data-wa-viewport="phone"] .wa-left-rail,
.wa-main-grid[data-wa-viewport="phone"] .wa-right-rail,
.wa-main-grid[data-wa-viewport="narrow"] .wa-left-rail,
.wa-main-grid[data-wa-viewport="narrow"] .wa-right-rail {
  position: static !important;
  top: auto !important;
  left: auto !important;
  right: auto !important;
  transform: none !important;
  opacity: 1 !important;
  flex: 1 1 auto !important;
  max-height: none !important;
}

.wa-main-grid[data-wa-viewport="stacked"] .wa-left-rail,
.wa-main-grid[data-wa-viewport="phone"] .wa-left-rail,
.wa-main-grid[data-wa-viewport="narrow"] .wa-left-rail {
  order: 2;
}

.wa-main-grid[data-wa-viewport="stacked"] .wa-right-rail,
.wa-main-grid[data-wa-viewport="phone"] .wa-right-rail,
.wa-main-grid[data-wa-viewport="narrow"] .wa-right-rail {
  order: 3;
}

.wa-main-grid[data-wa-viewport="stacked"] .wa-left-rail::after,
.wa-main-grid[data-wa-viewport="stacked"] .wa-right-rail::before,
.wa-main-grid[data-wa-viewport="phone"] .wa-left-rail::after,
.wa-main-grid[data-wa-viewport="phone"] .wa-right-rail::before,
.wa-main-grid[data-wa-viewport="narrow"] .wa-left-rail::after,
.wa-main-grid[data-wa-viewport="narrow"] .wa-right-rail::before {
  display: none !important;
}

.wa-main-grid[data-wa-viewport="stacked"] .wa-left-rail .wa-panel-block,
.wa-main-grid[data-wa-viewport="stacked"] .wa-right-rail .wa-panel-block,
.wa-main-grid[data-wa-viewport="phone"] .wa-left-rail .wa-panel-block,
.wa-main-grid[data-wa-viewport="phone"] .wa-right-rail .wa-panel-block,
.wa-main-grid[data-wa-viewport="narrow"] .wa-left-rail .wa-panel-block,
.wa-main-grid[data-wa-viewport="narrow"] .wa-right-rail .wa-panel-block {
  max-height: none !important;
  opacity: 1 !important;
  pointer-events: auto !important;
}

.wa-main-grid[data-wa-viewport="stacked"] .wa-right-rail:not(:hover):not(:focus-within) .wa-panel-block,
.wa-main-grid[data-wa-viewport="phone"] .wa-right-rail:not(:hover):not(:focus-within) .wa-panel-block,
.wa-main-grid[data-wa-viewport="narrow"] .wa-right-rail:not(:hover):not(:focus-within) .wa-panel-block {
  opacity: 1 !important;
  pointer-events: auto !important;
}

.wa-main-grid[data-wa-viewport="stacked"] .wa-stage-col > .block,
.wa-main-grid[data-wa-viewport="stacked"] .wa-stage-col > .form,
.wa-main-grid[data-wa-viewport="stacked"] .wa-preview-panel,
.wa-main-grid[data-wa-viewport="phone"] .wa-stage-col > .block,
.wa-main-grid[data-wa-viewport="phone"] .wa-stage-col > .form,
.wa-main-grid[data-wa-viewport="phone"] .wa-preview-panel,
.wa-main-grid[data-wa-viewport="narrow"] .wa-stage-col > .block,
.wa-main-grid[data-wa-viewport="narrow"] .wa-stage-col > .form,
.wa-main-grid[data-wa-viewport="narrow"] .wa-preview-panel {
  width: min(100%, 920px) !important;
  max-width: min(100%, 920px) !important;
}

.wa-main-grid[data-wa-viewport="stacked"] .wa-preview-panel,
.wa-main-grid[data-wa-viewport="phone"] .wa-preview-panel,
.wa-main-grid[data-wa-viewport="narrow"] .wa-preview-panel {
  min-height: clamp(560px, calc(100vh - 144px), 680px);
  border-radius: clamp(24px, 3vw, 32px) !important;
}

.wa-main-grid[data-wa-viewport="stacked"]:has(.wa-left-rail:hover) .wa-stage-col,
.wa-main-grid[data-wa-viewport="stacked"]:has(.wa-left-rail:focus-within) .wa-stage-col,
.wa-main-grid[data-wa-viewport="stacked"]:has(.wa-right-rail:hover) .wa-stage-col,
.wa-main-grid[data-wa-viewport="stacked"]:has(.wa-right-rail:focus-within) .wa-stage-col,
.wa-main-grid[data-wa-viewport="phone"]:has(.wa-left-rail:hover) .wa-stage-col,
.wa-main-grid[data-wa-viewport="phone"]:has(.wa-left-rail:focus-within) .wa-stage-col,
.wa-main-grid[data-wa-viewport="phone"]:has(.wa-right-rail:hover) .wa-stage-col,
.wa-main-grid[data-wa-viewport="phone"]:has(.wa-right-rail:focus-within) .wa-stage-col,
.wa-main-grid[data-wa-viewport="narrow"]:has(.wa-left-rail:hover) .wa-stage-col,
.wa-main-grid[data-wa-viewport="narrow"]:has(.wa-left-rail:focus-within) .wa-stage-col,
.wa-main-grid[data-wa-viewport="narrow"]:has(.wa-right-rail:hover) .wa-stage-col,
.wa-main-grid[data-wa-viewport="narrow"]:has(.wa-right-rail:focus-within) .wa-stage-col {
  transform: none !important;
}

.wa-main-grid[data-wa-viewport="stacked"]:has(.wa-left-rail:hover) .wa-preview-panel,
.wa-main-grid[data-wa-viewport="stacked"]:has(.wa-left-rail:focus-within) .wa-preview-panel,
.wa-main-grid[data-wa-viewport="stacked"]:has(.wa-right-rail:hover) .wa-preview-panel,
.wa-main-grid[data-wa-viewport="stacked"]:has(.wa-right-rail:focus-within) .wa-preview-panel,
.wa-main-grid[data-wa-viewport="stacked"]:has(.wa-left-rail:hover) .wa-stage-col > .block,
.wa-main-grid[data-wa-viewport="stacked"]:has(.wa-left-rail:focus-within) .wa-stage-col > .block,
.wa-main-grid[data-wa-viewport="stacked"]:has(.wa-right-rail:hover) .wa-stage-col > .block,
.wa-main-grid[data-wa-viewport="stacked"]:has(.wa-right-rail:focus-within) .wa-stage-col > .block,
.wa-main-grid[data-wa-viewport="stacked"]:has(.wa-left-rail:hover) .wa-stage-col > .form,
.wa-main-grid[data-wa-viewport="stacked"]:has(.wa-left-rail:focus-within) .wa-stage-col > .form,
.wa-main-grid[data-wa-viewport="stacked"]:has(.wa-right-rail:hover) .wa-stage-col > .form,
.wa-main-grid[data-wa-viewport="stacked"]:has(.wa-right-rail:focus-within) .wa-stage-col > .form,
.wa-main-grid[data-wa-viewport="phone"]:has(.wa-left-rail:hover) .wa-preview-panel,
.wa-main-grid[data-wa-viewport="phone"]:has(.wa-left-rail:focus-within) .wa-preview-panel,
.wa-main-grid[data-wa-viewport="phone"]:has(.wa-right-rail:hover) .wa-preview-panel,
.wa-main-grid[data-wa-viewport="phone"]:has(.wa-right-rail:focus-within) .wa-preview-panel,
.wa-main-grid[data-wa-viewport="phone"]:has(.wa-left-rail:hover) .wa-stage-col > .block,
.wa-main-grid[data-wa-viewport="phone"]:has(.wa-left-rail:focus-within) .wa-stage-col > .block,
.wa-main-grid[data-wa-viewport="phone"]:has(.wa-right-rail:hover) .wa-stage-col > .block,
.wa-main-grid[data-wa-viewport="phone"]:has(.wa-right-rail:focus-within) .wa-stage-col > .block,
.wa-main-grid[data-wa-viewport="phone"]:has(.wa-left-rail:hover) .wa-stage-col > .form,
.wa-main-grid[data-wa-viewport="phone"]:has(.wa-left-rail:focus-within) .wa-stage-col > .form,
.wa-main-grid[data-wa-viewport="phone"]:has(.wa-right-rail:hover) .wa-stage-col > .form,
.wa-main-grid[data-wa-viewport="phone"]:has(.wa-right-rail:focus-within) .wa-stage-col > .form,
.wa-main-grid[data-wa-viewport="narrow"]:has(.wa-left-rail:hover) .wa-preview-panel,
.wa-main-grid[data-wa-viewport="narrow"]:has(.wa-left-rail:focus-within) .wa-preview-panel,
.wa-main-grid[data-wa-viewport="narrow"]:has(.wa-right-rail:hover) .wa-preview-panel,
.wa-main-grid[data-wa-viewport="narrow"]:has(.wa-right-rail:focus-within) .wa-preview-panel,
.wa-main-grid[data-wa-viewport="narrow"]:has(.wa-left-rail:hover) .wa-stage-col > .block,
.wa-main-grid[data-wa-viewport="narrow"]:has(.wa-left-rail:focus-within) .wa-stage-col > .block,
.wa-main-grid[data-wa-viewport="narrow"]:has(.wa-right-rail:hover) .wa-stage-col > .block,
.wa-main-grid[data-wa-viewport="narrow"]:has(.wa-right-rail:focus-within) .wa-stage-col > .block,
.wa-main-grid[data-wa-viewport="narrow"]:has(.wa-left-rail:hover) .wa-stage-col > .form,
.wa-main-grid[data-wa-viewport="narrow"]:has(.wa-left-rail:focus-within) .wa-stage-col > .form,
.wa-main-grid[data-wa-viewport="narrow"]:has(.wa-right-rail:hover) .wa-stage-col > .form,
.wa-main-grid[data-wa-viewport="narrow"]:has(.wa-right-rail:focus-within) .wa-stage-col > .form {
  width: min(100%, 920px) !important;
  max-width: min(100%, 920px) !important;
}

.wa-main-grid[data-wa-viewport="stacked"] .wa-control-dock,
.wa-main-grid[data-wa-viewport="phone"] .wa-control-dock,
.wa-main-grid[data-wa-viewport="narrow"] .wa-control-dock {
  left: auto;
  transform: none;
  width: 100% !important;
  flex-wrap: wrap;
  justify-content: center;
}

.wa-main-grid[data-wa-viewport="phone"] .wa-world-tray-shell,
.wa-main-grid[data-wa-viewport="narrow"] .wa-world-tray-shell {
  max-width: calc(100% - 6px);
}

.wa-main-grid[data-wa-viewport="phone"] .wa-joystick-dock,
.wa-main-grid[data-wa-viewport="narrow"] .wa-joystick-dock {
  min-height: 196px;
  flex-wrap: wrap;
  justify-content: center;
  align-items: center;
}

.wa-main-grid[data-wa-viewport="phone"] .wa-joystick-dock .wa-floating-stick,
.wa-main-grid[data-wa-viewport="narrow"] .wa-joystick-dock .wa-floating-stick {
  width: min(44vw, 164px);
}

.wa-main-grid[data-wa-viewport="phone"] .wa-joystick-dock-copy,
.wa-main-grid[data-wa-viewport="narrow"] .wa-joystick-dock-copy {
  order: 3;
  flex: 1 1 100%;
  min-width: 0;
  max-width: min(220px, 100%);
}

.wa-main-grid[data-wa-viewport="phone"] .wa-joystick-dock .wa-stick,
.wa-main-grid[data-wa-viewport="narrow"] .wa-joystick-dock .wa-stick {
  width: 90px;
  height: 90px;
}

.wa-site-nav-shell[data-wa-viewport="narrow"] .wa-site-brand {
  max-width: calc(100vw - 96px);
}

.wa-site-nav-shell[data-wa-viewport="narrow"] .wa-site-brand-mark {
  width: 70px;
  height: 36px;
}

.wa-site-nav-shell[data-wa-viewport="narrow"] .wa-site-nav-left {
  font-size: 13px;
}

.wa-main-grid[data-wa-viewport="narrow"] .wa-player-footer {
  grid-template-columns: minmax(0, 1fr) auto;
  row-gap: 4px;
}

.wa-main-grid[data-wa-viewport="narrow"] .wa-player-footer-center {
  grid-column: 1 / -1;
  justify-self: center;
  max-width: 100%;
}

.wa-main-grid[data-wa-viewport="narrow"] .wa-control-dock > div,
.wa-main-grid[data-wa-viewport="narrow"] .wa-control-dock > .wrap {
  flex: 0 0 auto !important;
}

.wa-main-grid[data-wa-viewport="narrow"] .wa-control-dock button {
  min-width: 0 !important;
  padding: 0 10px !important;
  font-size: 8.5px !important;
}

.wa-player-footer-center.is-hidden,
.wa-player-footer-right.is-hidden {
  opacity: 0;
  visibility: hidden;
}

:root {
  --wa-media-frame-height: clamp(360px, calc(100vh - 304px), 680px);
}

#wa-main-preview-video,
#wa-main-preview-image,
.wa-preview-panel .tabitem,
#wa-main-preview-video .wrap,
#wa-main-preview-image .wrap,
#wa-main-preview-video .video-container,
#wa-main-preview-image .image-container,
.wa-preview-panel .model3D {
  width: 100% !important;
  min-height: 0 !important;
  height: auto !important;
  max-height: var(--wa-media-frame-height) !important;
  aspect-ratio: 16 / 9;
}

#wa-main-preview-video .wrap,
#wa-main-preview-image .wrap,
#wa-main-preview-video .video-container,
#wa-main-preview-image .image-container,
.wa-preview-panel .model3D {
  display: flex !important;
  align-items: stretch !important;
  justify-content: center !important;
  width: 100% !important;
  height: auto !important;
  overflow: hidden !important;
}

#wa-main-preview-video video,
#wa-main-preview-image img,
.wa-preview-panel model-viewer {
  width: 100% !important;
  height: 100% !important;
  max-height: none !important;
  aspect-ratio: 16 / 9;
  object-fit: contain !important;
}

@media (max-width: 980px) {
  :root {
    --wa-media-frame-height: clamp(320px, calc(100vh - 280px), 560px);
  }
}

@media (max-width: 760px) {
  :root {
    --wa-media-frame-height: clamp(240px, calc(100vh - 300px), 420px);
  }
}

.wa-main-grid.wa-main-grid-simple {
  justify-content: center !important;
  gap: clamp(14px, 1.8vw, 22px);
}

.wa-main-grid.wa-main-grid-simple:not([data-wa-viewport="phone"]):not([data-wa-viewport="narrow"]) {
  --wa-rail-width: clamp(248px, 25vw, 320px);
  flex-wrap: nowrap !important;
  align-items: flex-start !important;
}

.wa-main-grid.wa-main-grid-simple[data-wa-viewport="stacked"] {
  flex-direction: row !important;
}

.wa-main-grid.wa-main-grid-simple .wa-left-rail {
  display: none !important;
}

.wa-main-grid.wa-main-grid-simple:not([data-wa-viewport="phone"]):not([data-wa-viewport="narrow"]) .wa-right-rail {
  position: relative !important;
  top: auto !important;
  left: auto !important;
  right: auto !important;
  transform: none !important;
  opacity: 1 !important;
  flex: 0 0 var(--wa-rail-width) !important;
  width: var(--wa-rail-width) !important;
  min-width: var(--wa-rail-width) !important;
  max-width: var(--wa-rail-width) !important;
  max-height: none !important;
}

.wa-main-grid.wa-main-grid-simple:not([data-wa-viewport="phone"]):not([data-wa-viewport="narrow"]) .wa-right-rail::before {
  display: none !important;
}

.wa-main-grid.wa-main-grid-simple:not([data-wa-viewport="phone"]):not([data-wa-viewport="narrow"]) .wa-right-rail .wa-panel-block {
  max-height: none !important;
  opacity: 1 !important;
  pointer-events: auto !important;
  filter: none !important;
  transform: none !important;
  visibility: visible !important;
}

.wa-control-dock .wa-action-hidden,
.wa-control-dock .wa-action-hidden button,
.wa-control-dock button.wa-action-hidden {
  display: none !important;
}

.wa-mode-summary {
  margin-bottom: 12px !important;
}

.wa-mode-summary .wa-summary-card {
  border-color: rgba(130, 146, 170, 0.2);
  background: rgba(31, 34, 41, 0.62);
}

.wa-interface-contract .wa-summary-card {
  background: rgba(19, 22, 28, 0.72);
}

.wa-interface-contract .wa-summary-lines {
  max-height: 220px;
  overflow: auto;
  padding-right: 4px;
}

.wa-template-workbench-host {
  margin-bottom: 12px !important;
}

.wa-template-workbench {
  display: grid;
  gap: 10px;
  padding: 12px;
  border: 1px solid rgba(130, 146, 170, 0.18);
  background: rgba(16, 19, 24, 0.72);
}

.wa-workbench-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
}

.wa-workbench-head span {
  font: 700 9px/1 "IBM Plex Mono", monospace;
  color: rgba(155, 169, 190, 0.72);
  text-transform: uppercase;
}

.wa-workbench-head strong {
  font: 700 13px/1.2 "IBM Plex Sans", sans-serif;
  color: rgba(244, 247, 251, 0.92);
  text-align: right;
}

.wa-workbench-lanes {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
}

.wa-workbench-lane {
  min-width: 0;
  padding: 9px 8px;
  border: 1px solid rgba(130, 146, 170, 0.16);
  background: rgba(255, 255, 255, 0.035);
}

.wa-workbench-lane span,
.wa-workbench-lane strong,
.wa-workbench-action {
  display: block;
  overflow-wrap: anywhere;
}

.wa-workbench-lane span {
  margin-bottom: 4px;
  font: 700 8px/1 "IBM Plex Mono", monospace;
  color: rgba(155, 169, 190, 0.72);
  text-transform: uppercase;
}

.wa-workbench-lane strong {
  font: 600 11px/1.25 "IBM Plex Sans", sans-serif;
  color: rgba(244, 247, 251, 0.9);
}

.wa-workbench-action {
  justify-self: start;
  padding: 5px 8px;
  border: 1px solid rgba(130, 146, 170, 0.18);
  color: rgba(244, 247, 251, 0.86);
  font: 700 9px/1 "IBM Plex Mono", monospace;
  background: rgba(255, 255, 255, 0.045);
}

.wa-template-scene-3d .wa-template-workbench {
  border-color: rgba(52, 211, 153, 0.3);
}

.wa-template-depth-geometry .wa-template-workbench {
  border-color: rgba(96, 165, 250, 0.3);
}

.wa-template-embodied-policy .wa-template-workbench,
.wa-template-visual-action .wa-template-workbench {
  border-color: rgba(244, 114, 182, 0.28);
}

.wa-template-hosted-api .wa-template-workbench {
  border-color: rgba(250, 204, 21, 0.28);
}

.wa-template-depth-geometry,
.wa-template-embodied-policy,
.wa-template-visual-action,
.wa-template-hosted-api {
  --wa-media-frame-height: clamp(260px, calc(100vh - 420px), 480px);
}

.wa-template-depth-geometry .wa-joystick-dock-shell,
.wa-template-embodied-policy .wa-joystick-dock-shell,
.wa-template-visual-action .wa-joystick-dock-shell,
.wa-template-hosted-api .wa-joystick-dock-shell,
.wa-template-conditioned-video .wa-joystick-dock-shell,
.wa-template-text-video .wa-joystick-dock-shell {
  display: none !important;
}

.wa-template-depth-geometry .wa-input-tray-gallery,
.wa-template-embodied-policy .wa-input-tray-gallery,
.wa-template-visual-action .wa-input-tray-gallery,
.wa-template-hosted-api .wa-input-tray-gallery {
  display: none !important;
}

.wa-template-depth-geometry .wa-player-chrome,
.wa-template-embodied-policy .wa-player-chrome,
.wa-template-visual-action .wa-player-chrome,
.wa-template-hosted-api .wa-player-chrome {
  display: none !important;
}

.wa-template-depth-geometry .tabs [role="tablist"] button:nth-child(3),
.wa-template-embodied-policy .tabs [role="tablist"] button:nth-child(1),
.wa-template-embodied-policy .tabs [role="tablist"] button:nth-child(3),
.wa-template-visual-action .tabs [role="tablist"] button:nth-child(1),
.wa-template-visual-action .tabs [role="tablist"] button:nth-child(3),
.wa-template-hosted-api .tabs [role="tablist"] button:nth-child(3) {
  display: none !important;
}

@media (max-width: 760px) {
  .wa-workbench-lanes {
    grid-template-columns: 1fr;
  }
}

.wa-preview-panel:not(.wa-template-interactive-world) {
  width: min(100%, 1080px) !important;
  max-width: min(100%, 1080px) !important;
  min-height: 0 !important;
  max-height: none !important;
  padding: 16px !important;
  border-radius: 12px !important;
  border-color: rgba(148, 163, 184, 0.22) !important;
  background: rgba(21, 25, 33, 0.94) !important;
  box-shadow: 0 18px 52px rgba(0, 0, 0, 0.28) !important;
  overflow: visible !important;
}

.wa-preview-panel:not(.wa-template-interactive-world)::before {
  display: none !important;
}

.wa-preview-panel:not(.wa-template-interactive-world) .wa-player-chrome,
.wa-preview-panel:not(.wa-template-interactive-world) .wa-world-tray-shell,
.wa-preview-panel:not(.wa-template-interactive-world) .wa-input-tray-gallery,
.wa-preview-panel:not(.wa-template-interactive-world) .wa-joystick-dock-shell {
  display: none !important;
}

.wa-preview-panel:not(.wa-template-interactive-world) .wa-stage-empty {
  inset: 16px;
  border-radius: 8px;
  background: rgba(2, 6, 23, 0.74);
}

.wa-preview-panel:not(.wa-template-interactive-world) .wa-stage-empty-title {
  font: 700 13px/1.35 "IBM Plex Mono", monospace;
  letter-spacing: 0;
}

.wa-preview-panel:not(.wa-template-interactive-world) .wa-stage-empty-copy {
  font: 400 12px/1.55 "IBM Plex Mono", monospace;
}

.wa-preview-panel:not(.wa-template-interactive-world) .tabs [role="tablist"] {
  width: 100%;
  margin: 0 0 12px !important;
  padding: 0 0 10px !important;
  gap: 4px !important;
  border-bottom: 1px solid rgba(148, 163, 184, 0.16);
}

.wa-preview-panel:not(.wa-template-interactive-world) .tabs [role="tablist"] button {
  min-height: 32px !important;
  padding: 0 10px !important;
  border-radius: 6px !important;
  border-color: transparent !important;
  background: transparent !important;
  color: rgba(226, 232, 240, 0.78) !important;
  font: 600 12px/1.25 "IBM Plex Mono", monospace !important;
  letter-spacing: 0 !important;
}

.wa-preview-panel:not(.wa-template-interactive-world) .tabs [role="tablist"] button.selected,
.wa-preview-panel:not(.wa-template-interactive-world) .tabs [role="tablist"] button[aria-selected="true"] {
  border-color: rgba(148, 163, 184, 0.2) !important;
  background: rgba(148, 163, 184, 0.12) !important;
  color: #f8fafc !important;
}

.wa-preview-panel:not(.wa-template-interactive-world) .tabitem {
  min-height: var(--wa-media-frame-height) !important;
  height: auto !important;
}

.wa-preview-panel:not(.wa-template-interactive-world) #wa-main-preview-video .wrap,
.wa-preview-panel:not(.wa-template-interactive-world) #wa-main-preview-image .wrap,
.wa-preview-panel:not(.wa-template-interactive-world) .wa-stage-points-host,
.wa-preview-panel:not(.wa-template-interactive-world) .wa-stage-embodied-host {
  min-height: var(--wa-media-frame-height) !important;
}

.wa-preview-panel:not(.wa-template-interactive-world) #wa-main-preview-video video,
.wa-preview-panel:not(.wa-template-interactive-world) #wa-main-preview-image img,
.wa-preview-panel:not(.wa-template-interactive-world) model-viewer,
.wa-preview-panel:not(.wa-template-interactive-world) .wa-spatial-shell,
.wa-preview-panel:not(.wa-template-interactive-world) .wa-viser-frame,
.wa-preview-panel:not(.wa-template-interactive-world) .wa-embodied-viewport {
  border-radius: 8px !important;
}

.wa-preview-panel:not(.wa-template-interactive-world) .wa-spatial-shell,
.wa-preview-panel:not(.wa-template-interactive-world) .wa-viser-frame {
  min-height: clamp(420px, calc(100vh - 260px), 680px);
  height: clamp(420px, calc(100vh - 260px), 680px);
}

.wa-preview-panel:not(.wa-template-interactive-world) .wa-spatial-shell.is-ready .wa-spark-overlay {
  opacity: 0;
}

.wa-preview-panel:not(.wa-template-interactive-world) .wa-spatial-shell.is-empty .wa-spark-overlay,
.wa-preview-panel:not(.wa-template-interactive-world) .wa-spatial-shell.is-error .wa-spark-overlay {
  opacity: 1;
}

.wa-preview-panel:not(.wa-template-interactive-world) .wa-control-dock {
  position: static;
  left: auto;
  bottom: auto;
  transform: none;
  width: auto !important;
  margin-top: 14px !important;
  justify-content: flex-start;
}

.wa-preview-panel:not(.wa-template-interactive-world) .wa-control-dock button {
  min-height: 34px !important;
  min-width: 72px !important;
  border-radius: 6px !important;
  padding: 0 14px !important;
  font: 700 12px/1 "IBM Plex Mono", monospace !important;
  letter-spacing: 0 !important;
}

.wa-category-video-generation .wa-mode-summary .wa-summary-card,
.wa-category-video-generation.wa-preview-panel {
  border-color: rgba(255, 104, 43, 0.28) !important;
}

.wa-category-3d-scene .wa-mode-summary .wa-summary-card,
.wa-category-3d-scene.wa-preview-panel {
  border-color: rgba(52, 211, 153, 0.28) !important;
}

.wa-category-depth-geometry .wa-mode-summary .wa-summary-card,
.wa-category-depth-geometry.wa-preview-panel {
  border-color: rgba(96, 165, 250, 0.28) !important;
}

.wa-category-visual-action .wa-mode-summary .wa-summary-card,
.wa-category-embodied-action .wa-mode-summary .wa-summary-card,
.wa-category-visual-action.wa-preview-panel,
.wa-category-embodied-action.wa-preview-panel {
  border-color: rgba(244, 114, 182, 0.26) !important;
}

.wa-category-remote-api .wa-mode-summary .wa-summary-card,
.wa-category-remote-api.wa-preview-panel {
  border-color: rgba(250, 204, 21, 0.26) !important;
}

.wa-template-embodied-policy .wa-interface-contract .wa-summary-card,
.wa-template-visual-action .wa-interface-contract .wa-summary-card {
  border-color: rgba(244, 114, 182, 0.34) !important;
}

.wa-template-scene-3d .wa-interface-contract .wa-summary-card {
  border-color: rgba(52, 211, 153, 0.34) !important;
}

.wa-template-depth-geometry .wa-interface-contract .wa-summary-card {
  border-color: rgba(96, 165, 250, 0.34) !important;
}

.wa-template-hosted-api .wa-interface-contract .wa-summary-card {
  border-color: rgba(250, 204, 21, 0.34) !important;
}

.wa-main-grid.wa-main-grid-simple:not([data-wa-viewport="phone"]):not([data-wa-viewport="narrow"]) .wa-stage-col {
  transform: none !important;
  flex: 1 1 auto !important;
  width: auto !important;
  max-width: none !important;
  min-width: 0 !important;
}

.wa-main-grid.wa-main-grid-simple:not([data-wa-viewport="phone"]):not([data-wa-viewport="narrow"]) .wa-preview-panel,
.wa-main-grid.wa-main-grid-simple:not([data-wa-viewport="phone"]):not([data-wa-viewport="narrow"]) .wa-stage-col > .block,
.wa-main-grid.wa-main-grid-simple:not([data-wa-viewport="phone"]):not([data-wa-viewport="narrow"]) .wa-stage-col > .form {
  width: min(100%, 1080px) !important;
  max-width: min(100%, 1080px) !important;
}

.wa-main-grid.wa-main-grid-simple .wa-stage-col,
.wa-main-grid.wa-main-grid-simple .wa-preview-panel,
.wa-main-grid.wa-main-grid-simple .wa-stage-col > .block,
.wa-main-grid.wa-main-grid-simple .wa-stage-col > .form {
  transition: none !important;
}

.wa-main-grid.wa-main-grid-simple .wa-preview-panel {
  min-height: 0 !important;
  height: auto !important;
}

.wa-main-grid.wa-main-grid-simple #wa-main-preview-video,
.wa-main-grid.wa-main-grid-simple #wa-main-preview-image,
.wa-main-grid.wa-main-grid-simple .wa-preview-panel .tabitem,
.wa-main-grid.wa-main-grid-simple #wa-main-preview-video .wrap,
.wa-main-grid.wa-main-grid-simple #wa-main-preview-image .wrap,
.wa-main-grid.wa-main-grid-simple #wa-main-preview-video .video-container,
.wa-main-grid.wa-main-grid-simple #wa-main-preview-image .image-container,
.wa-main-grid.wa-main-grid-simple .wa-preview-panel .model3D {
  min-height: 0 !important;
  height: auto !important;
  aspect-ratio: 16 / 9;
}

.wa-main-grid.wa-main-grid-simple #wa-main-preview-video video,
.wa-main-grid.wa-main-grid-simple #wa-main-preview-image img,
.wa-main-grid.wa-main-grid-simple .wa-preview-panel model-viewer {
  aspect-ratio: 16 / 9;
}

html[data-wa-viewport="phone"] .wa-site-nav,
html[data-wa-viewport="narrow"] .wa-site-nav {
  flex-wrap: nowrap !important;
  align-items: center !important;
  gap: 8px !important;
  padding: 8px 12px 6px !important;
}

html[data-wa-viewport="phone"] .wa-site-brand,
html[data-wa-viewport="narrow"] .wa-site-brand {
  flex: 0 1 172px !important;
  max-width: 172px !important;
  gap: 8px !important;
  padding: 4px 10px 4px 4px !important;
}

html[data-wa-viewport="phone"] .wa-site-brand-mark,
html[data-wa-viewport="narrow"] .wa-site-brand-mark {
  width: 52px !important;
  height: 30px !important;
  border-radius: 10px !important;
}

html[data-wa-viewport="phone"] .wa-site-nav-left,
html[data-wa-viewport="narrow"] .wa-site-nav-left {
  min-width: 0 !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  font-size: 14px !important;
  letter-spacing: 0 !important;
}

html[data-wa-viewport="phone"] .wa-site-nav-right,
html[data-wa-viewport="narrow"] .wa-site-nav-right {
  flex: 0 0 auto !important;
  gap: 6px !important;
  margin-left: auto !important;
}

html[data-wa-viewport="phone"] .wa-site-nav-icon,
html[data-wa-viewport="narrow"] .wa-site-nav-icon {
  width: 32px !important;
  height: 32px !important;
}

html[data-wa-viewport="phone"] .wa-site-nav-icon svg,
html[data-wa-viewport="narrow"] .wa-site-nav-icon svg {
  width: 18px !important;
  height: 18px !important;
}

html[data-wa-viewport="phone"] .wa-main-grid,
html[data-wa-viewport="narrow"] .wa-main-grid {
  gap: 14px !important;
  padding-top: 4px !important;
}

html[data-wa-viewport="phone"] .wa-preview-panel,
html[data-wa-viewport="narrow"] .wa-preview-panel,
.wa-main-grid[data-wa-viewport="phone"] .wa-preview-panel,
.wa-main-grid[data-wa-viewport="narrow"] .wa-preview-panel {
  --wa-panel-pad-bottom: 232px;
  min-height: clamp(540px, calc(100vh - 116px), 640px) !important;
}

html[data-wa-viewport="phone"] .wa-player-footer,
html[data-wa-viewport="narrow"] .wa-player-footer,
.wa-main-grid[data-wa-viewport="phone"] .wa-player-footer,
.wa-main-grid[data-wa-viewport="narrow"] .wa-player-footer {
  display: none !important;
}

html[data-wa-viewport="phone"] .wa-control-dock,
html[data-wa-viewport="narrow"] .wa-control-dock,
.wa-main-grid[data-wa-viewport="phone"] .wa-control-dock,
.wa-main-grid[data-wa-viewport="narrow"] .wa-control-dock {
  position: absolute !important;
  left: var(--wa-panel-pad-x) !important;
  right: var(--wa-panel-pad-x) !important;
  bottom: 340px !important;
  transform: none !important;
  width: auto !important;
  gap: 8px !important;
  margin: 0 !important;
  z-index: 14 !important;
}

html[data-wa-viewport="phone"] .wa-control-dock button,
html[data-wa-viewport="narrow"] .wa-control-dock button,
.wa-main-grid[data-wa-viewport="phone"] .wa-control-dock button,
.wa-main-grid[data-wa-viewport="narrow"] .wa-control-dock button {
  min-height: 32px !important;
  min-width: 86px !important;
}

html[data-wa-viewport="phone"] .wa-preview-panel:not(.wa-joystick-open) .wa-joystick-dock,
html[data-wa-viewport="narrow"] .wa-preview-panel:not(.wa-joystick-open) .wa-joystick-dock,
.wa-main-grid[data-wa-viewport="phone"] .wa-preview-panel:not(.wa-joystick-open) .wa-joystick-dock,
.wa-main-grid[data-wa-viewport="narrow"] .wa-preview-panel:not(.wa-joystick-open) .wa-joystick-dock {
  display: none !important;
}

html[data-wa-viewport="phone"] .wa-preview-panel .wa-status,
html[data-wa-viewport="narrow"] .wa-preview-panel .wa-status {
  display: none !important;
}

/* Interactive focus and performance modes */
html[data-wa-focus="stage"] .wa-main-grid {
  justify-content: center !important;
  gap: 0 !important;
}

html[data-wa-focus="stage"] .wa-left-rail,
html[data-wa-focus="stage"] .wa-right-rail {
  display: none !important;
}

html[data-wa-focus="stage"] .wa-stage-col {
  flex: 0 1 min(1280px, calc(100vw - 48px)) !important;
  width: min(1280px, calc(100vw - 48px)) !important;
  max-width: min(1280px, calc(100vw - 48px)) !important;
  transform: none !important;
}

html[data-wa-focus="stage"] .wa-stage-col > .block,
html[data-wa-focus="stage"] .wa-stage-col > .form,
html[data-wa-focus="stage"] .wa-preview-panel {
  width: min(1280px, calc(100vw - 48px)) !important;
  max-width: min(1280px, calc(100vw - 48px)) !important;
}

html[data-wa-focus="stage"] .wa-preview-panel {
  --wa-media-frame-height: clamp(420px, calc(100vh - 294px), 760px);
}

html[data-wa-focus="stage"] .wa-site-brand {
  max-width: calc(100vw - 216px);
}

html[data-wa-performance="lite"] .wa-panel-block,
html[data-wa-performance="lite"] .wa-preview-panel,
html[data-wa-performance="lite"] .wa-site-brand,
html[data-wa-performance="lite"] .wa-site-nav-icon,
html[data-wa-performance="lite"] .wa-left-rail .wa-panel-block,
html[data-wa-performance="lite"] .wa-right-rail .wa-panel-block {
  backdrop-filter: none !important;
  box-shadow: none !important;
}

html[data-wa-performance="lite"] .wa-preview-panel,
html[data-wa-performance="lite"] .wa-stage-col,
html[data-wa-performance="lite"] .wa-left-rail,
html[data-wa-performance="lite"] .wa-right-rail,
html[data-wa-performance="lite"] .wa-site-nav-icon,
html[data-wa-performance="lite"] .wa-tray-item,
html[data-wa-performance="lite"] .wa-stage-empty,
html[data-wa-performance="lite"] .wa-joystick-dock {
  transition-duration: 90ms !important;
}

html[data-wa-performance="lite"] .wa-preview-panel::before {
  box-shadow: none !important;
}

html[data-wa-performance="lite"] .wa-stage-loader-cell {
  animation-duration: 1.8s;
}

html[data-wa-performance="lite"] body,
html[data-wa-performance="lite"] .gradio-container {
  background: linear-gradient(180deg, var(--wa-page-bg) 0%, var(--wa-page-bg-2) 100%) !important;
}

@media (prefers-reduced-motion: reduce) {
  .wa-stage-loader-cell,
  .wa-progress-bar.is-indeterminate {
    animation: none !important;
  }

  .wa-preview-panel,
  .wa-stage-col,
  .wa-left-rail,
  .wa-right-rail,
  .wa-tray-item,
  .wa-site-nav-icon {
    transition: none !important;
  }
}
"""

__all__ = ["CUSTOM_CSS"]
