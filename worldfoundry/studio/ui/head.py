from __future__ import annotations

from worldfoundry.studio.ui.vendor_assets import VENDOR_ASSET_INSTALL_COMMAND

from .assets import SPARK_MODULE_PATH, THREE_MODULE_PATH
from .assets import local_module_url as local_module_url

HEAD_HTML = """
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<script type="importmap">
{
  "imports": {
    "three": "__WA_THREE_MODULE_URL__",
    "@sparkjsdev/spark": "__WA_SPARK_MODULE_URL__"
  }
}
</script>
<script>
(() => {
  const WA_BUILD_ID = "20260513-studio-ui-tray-tabs-1";
  const BUILD_STORAGE_KEY = "worldfoundry-studio-build-id";
  const keyMap = {
    w: "wa-live-forward",
    a: "wa-live-left",
    s: "wa-live-backward",
    d: "wa-live-right",
    j: "wa-live-camera-left",
    l: "wa-live-camera-right",
    i: "wa-live-camera-up",
    k: "wa-live-camera-down",
  };
  const stickVisualMap = {
    w: { stickId: "wa-move-stick", x: 0, y: -1 },
    a: { stickId: "wa-move-stick", x: -1, y: 0 },
    s: { stickId: "wa-move-stick", x: 0, y: 1 },
    d: { stickId: "wa-move-stick", x: 1, y: 0 },
    i: { stickId: "wa-look-stick", x: 0, y: -1 },
    j: { stickId: "wa-look-stick", x: -1, y: 0 },
    k: { stickId: "wa-look-stick", x: 0, y: 1 },
    l: { stickId: "wa-look-stick", x: 1, y: 0 },
  };
  const keyboardStickState = {
    "wa-move-stick": new Set(),
    "wa-look-stick": new Set(),
  };
  const activeLiveKeys = new Set();
  let liveQueuedKey = "";
  let liveLastIntentKey = "";
  let liveRequestBusy = false;
  let liveDispatchCooldownUntil = 0;
  let trayThumbsDirty = true;
  let uiTranslationDirty = true;
  let resizeSyncTimer = null;
  let syncTimer = null;
  let syncQueued = false;
  const THEME_STORAGE_KEY = "worldfoundry-studio-theme";
  const JOYSTICK_STORAGE_KEY = "worldfoundry-studio-joystick";
  const FOCUS_STORAGE_KEY = "worldfoundry-studio-stage-focus";
  const PERFORMANCE_STORAGE_KEY = "worldfoundry-studio-performance-mode";

  try {
    const seenBuild = window.sessionStorage.getItem(BUILD_STORAGE_KEY);
    if (seenBuild !== WA_BUILD_ID) {
      window.sessionStorage.setItem(BUILD_STORAGE_KEY, WA_BUILD_ID);
      const url = new URL(window.location.href);
      if (url.searchParams.get("wa_build") !== WA_BUILD_ID) {
        url.searchParams.set("wa_build", WA_BUILD_ID);
        window.location.replace(url.toString());
        return;
      }
    }
  } catch (error) {
    // Ignore storage / navigation failures inside embedded browsers.
  }

  const readPreference = (key, fallback = "") => {
    try {
      const value = window.localStorage.getItem(key);
      return value == null ? fallback : value;
    } catch (error) {
      return fallback;
    }
  };

  const writePreference = (key, value) => {
    try {
      window.localStorage.setItem(key, value);
    } catch (error) {
      return;
    }
  };

  const syncViewportMode = () => {
    const width = window.innerWidth || document.documentElement.clientWidth || 0;
    let mode = "wide";
    if (width <= 460) {
      mode = "narrow";
    } else if (width <= 760) {
      mode = "phone";
    } else if (width <= 980) {
      mode = "stacked";
    } else if (width <= 1120) {
      mode = "compact";
    } else if (width <= 1280) {
      mode = "balanced";
    }
    document.documentElement.dataset.waViewport = mode;
    document.querySelectorAll(".wa-main-grid, .wa-site-nav-shell").forEach((node) => {
      if (node instanceof HTMLElement) {
        node.dataset.waViewport = mode;
      }
    });
  };

  document.documentElement.dataset.waTheme =
    readPreference(THEME_STORAGE_KEY, "dark") === "light" ? "light" : "dark";
  document.documentElement.dataset.waFocus =
    readPreference(FOCUS_STORAGE_KEY, "off") === "on" ? "stage" : "default";
  document.documentElement.dataset.waPerformance =
    readPreference(PERFORMANCE_STORAGE_KEY, "balanced") === "lite" ? "lite" : "balanced";
  syncViewportMode();

  const isEditableTarget = (node) => {
    if (!node) return false;
    const tag = (node.tagName || "").toLowerCase();
    return tag === "input" || tag === "textarea" || tag === "select" || node.isContentEditable;
  };

  const pulse = (button) => {
    if (typeof button.animate === "function") {
      button.animate(
        [
          { transform: "translateY(0) scale(1)", opacity: 1 },
          { transform: "translateY(-1px) scale(0.98)", opacity: 0.92 },
          { transform: "translateY(0) scale(1)", opacity: 1 },
        ],
        {
          duration: 180,
          easing: "cubic-bezier(0.22, 1, 0.36, 1)",
        },
      );
      return;
    }
    button.classList.add("wa-keyflash");
    window.setTimeout(() => button.classList.remove("wa-keyflash"), 180);
  };

  const elementIsRendered = (node) => {
    if (!(node instanceof HTMLElement)) return false;
    if (node.hidden || node.getAttribute("aria-hidden") === "true") return false;
    const style = window.getComputedStyle(node);
    return (
      style.display !== "none"
      && style.visibility !== "hidden"
      && style.opacity !== "0"
      && node.getClientRects().length > 0
    );
  };

  const controlButtonIsReady = (button) => {
    if (!button || button.disabled || !elementIsRendered(button)) return false;
    const liveDock = button.closest(".wa-live-dock, .wa-joystick-bridge");
    return !liveDock || elementIsRendered(liveDock);
  };

  const controlButtonForKey = (key) => {
    const host = document.getElementById(keyMap[key]);
    if (!host) return null;
    return host.querySelector("button") || host;
  };

  const triggerMappedControl = (key) => {
    const button = controlButtonForKey(key);
    if (!controlButtonIsReady(button)) return;
    pulse(button);
    button.click();
  };

  const setLiveKeyActive = (key, active) => {
    if (!keyMap[key]) return;
    if (active) {
      activeLiveKeys.add(key);
      liveLastIntentKey = key;
      return;
    }
    activeLiveKeys.delete(key);
    if (liveQueuedKey === key) {
      liveQueuedKey = "";
    }
    if (liveLastIntentKey === key) {
      liveLastIntentKey = Array.from(activeLiveKeys).pop() || "";
    }
  };

  const resetStickVisual = (stickId) => {
    const shell = document.getElementById(`${stickId}-shell`);
    const thumb = shell?.querySelector(".wa-stick-thumb");
    if (!(shell instanceof HTMLElement) || !(thumb instanceof HTMLElement)) return;
    shell.dataset.activeKey = "";
    shell.classList.remove("is-key-active");
    thumb.style.transform = "translate(-50%, -50%)";
  };

  const keyboardStickVector = (stickId) => {
    let x = 0;
    let y = 0;
    (keyboardStickState[stickId] || new Set()).forEach((activeKey) => {
      const mapping = stickVisualMap[activeKey];
      if (!mapping || mapping.stickId !== stickId) return;
      x += mapping.x;
      y += mapping.y;
    });
    const clampedX = Math.max(-1, Math.min(1, x));
    const clampedY = Math.max(-1, Math.min(1, y));
    const magnitude = Math.hypot(clampedX, clampedY) || 1;
    return {
      x: magnitude > 1 ? clampedX / magnitude : clampedX,
      y: magnitude > 1 ? clampedY / magnitude : clampedY,
    };
  };

  const applyKeyboardStickVisual = (stickId) => {
    mountVirtualControls();
    const shell = document.getElementById(`${stickId}-shell`);
    const stick = shell?.querySelector(`#${stickId}`);
    const thumb = shell?.querySelector(".wa-stick-thumb");
    if (!(shell instanceof HTMLElement) || !(stick instanceof HTMLElement) || !(thumb instanceof HTMLElement)) {
      return;
    }
    const activeKeys = keyboardStickState[stickId] || new Set();
    if (!activeKeys.size) {
      resetStickVisual(stickId);
      return;
    }
    const vector = keyboardStickVector(stickId);
    const maxRadius = Math.max(stick.clientWidth, 68) * 0.26;
    shell.dataset.activeKey = Array.from(activeKeys).sort().join(",");
    shell.classList.add("is-key-active");
    thumb.style.transform = `translate(calc(-50% + ${vector.x * maxRadius}px), calc(-50% + ${vector.y * maxRadius}px))`;
  };

  const syncKeyboardStickVisual = (key, active) => {
    const mapping = stickVisualMap[key];
    if (!mapping) return;
    const activeKeys = keyboardStickState[mapping.stickId];
    if (!activeKeys) return;
    if (active) {
      activeKeys.add(key);
    } else {
      activeKeys.delete(key);
    }
    applyKeyboardStickVisual(mapping.stickId);
  };

  const resetAllKeyboardStickVisuals = () => {
    Object.values(keyboardStickState).forEach((activeKeys) => activeKeys.clear());
    resetStickVisual("wa-move-stick");
    resetStickVisual("wa-look-stick");
  };

  const joystickControlsAvailable = () =>
    Object.keys(keyMap).some((key) => {
      const button = controlButtonForKey(key);
      return controlButtonIsReady(button);
    });

  const ensureVirtualStick = (stickId, label) => {
    const host =
      document.getElementById("wa-joystick-dock")
      || document.querySelector(".wa-preview-panel")
      || document.body;
    let shell = document.getElementById(`${stickId}-shell`);
    if (!shell) {
      shell = document.createElement("div");
      shell.id = `${stickId}-shell`;
      shell.className = `wa-stick-shell wa-floating-stick ${stickId === "wa-move-stick" ? "wa-floating-stick-left" : "wa-floating-stick-right"}`;
      shell.innerHTML = `
        <div class="wa-stick-label">${label}</div>
        <div id="${stickId}" class="wa-stick">
          <div class="wa-stick-ring"></div>
          <div class="wa-stick-thumb"></div>
        </div>
      `;
    }
    shell.className = `wa-stick-shell wa-floating-stick ${stickId === "wa-move-stick" ? "wa-floating-stick-left" : "wa-floating-stick-right"}`;
    shell.style.top = "";
    shell.style.bottom = "";
    shell.style.left = "";
    shell.style.right = "";
    if (shell.parentElement !== host) {
      host.appendChild(shell);
    }
    return shell;
  };

  const installVirtualStick = (stickId, keysByAxis) => {
    const shell = ensureVirtualStick(stickId, stickId === "wa-move-stick" ? "move" : "look");
    const stick = shell.querySelector(`#${stickId}`);
    if (!stick || stick.dataset.bound === "1") return;
    const thumb = stick.querySelector(".wa-stick-thumb");
    if (!thumb) return;
    stick.dataset.bound = "1";

    let pointerId = null;
    let activeKey = "";

    const setPointerKey = (key) => {
      const nextKey = key || "";
      if (activeKey === nextKey) return;
      if (activeKey) {
        setLiveKeyActive(activeKey, false);
      }
      activeKey = nextKey;
      if (!activeKey) return;
      setLiveKeyActive(activeKey, true);
      queueMappedControl(activeKey);
    };

    const resetStick = () => {
      pointerId = null;
      setPointerKey("");
      thumb.style.transform = "translate(-50%, -50%)";
    };

    const updateStick = (clientX, clientY) => {
      const rect = stick.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height / 2;
      const dx = clientX - centerX;
      const dy = clientY - centerY;
      const maxRadius = rect.width * 0.26;
      const distance = Math.hypot(dx, dy) || 1;
      const clamped = Math.min(distance, maxRadius);
      const offsetX = (dx / distance) * clamped;
      const offsetY = (dy / distance) * clamped;
      thumb.style.transform = `translate(calc(-50% + ${offsetX}px), calc(-50% + ${offsetY}px))`;

      const deadZone = rect.width * 0.12;
      if (Math.abs(dx) < deadZone && Math.abs(dy) < deadZone) {
        setPointerKey("");
        return;
      }
      if (Math.abs(dx) > Math.abs(dy)) {
        setPointerKey(dx > 0 ? keysByAxis.right : keysByAxis.left);
      } else {
        setPointerKey(dy > 0 ? keysByAxis.down : keysByAxis.up);
      }
    };

    stick.addEventListener("pointerdown", (event) => {
      pointerId = event.pointerId;
      stick.setPointerCapture(pointerId);
      updateStick(event.clientX, event.clientY);
    });
    stick.addEventListener("pointermove", (event) => {
      if (pointerId !== event.pointerId) return;
      updateStick(event.clientX, event.clientY);
    });
    stick.addEventListener("pointerup", (event) => {
      if (pointerId !== event.pointerId) return;
      resetStick();
    });
    stick.addEventListener("pointercancel", resetStick);
    stick.addEventListener("lostpointercapture", resetStick);
  };

  const mountVirtualControls = () => {
    const allowVirtualControls =
      !spatialWorldActive() && joystickDockOpen() && joystickControlsAvailable();
    if (!allowVirtualControls) {
      resetAllKeyboardStickVisuals();
      document.querySelectorAll(".wa-floating-stick").forEach((node) => node.remove());
      return;
    }
    installVirtualStick("wa-move-stick", { up: "w", down: "s", left: "a", right: "d" });
    installVirtualStick("wa-look-stick", { up: "i", down: "k", left: "j", right: "l" });
  };

  const normalizeText = (value) => (value || "").replace(/\\s+/g, " ").trim();

  const getStatusText = () =>
    normalizeText(document.querySelector(".wa-preview-panel .wa-status")?.textContent || "");

  const formatDuration = (seconds) => {
    const value = Math.max(0, Number.isFinite(seconds) ? Math.floor(seconds) : 0);
    const minutes = Math.floor(value / 60);
    const remainder = String(value % 60).padStart(2, "0");
    return `${minutes}:${remainder}`;
  };

  const truncateWorldLabel = (value) => {
    const normalized = normalizeText(value) || "WorldFoundry";
    return normalized.length > 20 ? `${normalized.slice(0, 19)}…` : normalized;
  };

  const inferStageState = (statusText) => {
    const lower = statusText.toLowerCase();
    if (lower.includes("queue")) {
      return { state: "queue", label: "IN QUEUE" };
    }
    if (lower.includes("input scene ready") || lower.includes("input source staged")) {
      return { state: "input", label: "INPUT READY" };
    }
    if (
      lower.includes("error") ||
      lower.includes("failed") ||
      lower.includes("exception") ||
      lower.includes("traceback")
    ) {
      return { state: "error", label: "ERROR" };
    }
    if (
      lower.includes("preparing") ||
      lower.includes("loading") ||
      lower.includes("initializing") ||
      lower.includes("running stream") ||
      lower.includes("running fresh") ||
      lower.includes("resolving runtime")
    ) {
      return { state: "loading", label: "RUNNING" };
    }
    if (lower === "idle" || !lower) {
      return { state: "idle", label: "READY" };
    }
    return { state: "flowing", label: "FLOWING" };
  };

  const liveStageBusy = () => {
    const state = inferStageState(getStatusText()).state;
    return state === "loading" || state === "queue";
  };

  const preferredLiveKey = () => {
    if (liveQueuedKey && activeLiveKeys.has(liveQueuedKey)) {
      return liveQueuedKey;
    }
    if (liveLastIntentKey && activeLiveKeys.has(liveLastIntentKey)) {
      return liveLastIntentKey;
    }
    return Array.from(activeLiveKeys).pop() || liveQueuedKey || "";
  };

  const drainMappedControlQueue = () => {
    if (liveStageBusy() || Date.now() < liveDispatchCooldownUntil) {
      liveRequestBusy = true;
      return;
    }
    const nextKey = preferredLiveKey();
    if (!nextKey) {
      liveRequestBusy = false;
      return;
    }
    liveQueuedKey = "";
    liveRequestBusy = true;
    liveDispatchCooldownUntil = Date.now() + 220;
    triggerMappedControl(nextKey);
  };

  const queueMappedControl = (key) => {
    if (!keyMap[key]) return;
    liveQueuedKey = key;
    liveLastIntentKey = key;
    drainMappedControlQueue();
  };

  const syncLiveRequestState = () => {
    const busy = liveStageBusy();
    if (busy) {
      liveRequestBusy = true;
      return;
    }
    if (liveRequestBusy && Date.now() < liveDispatchCooldownUntil) {
      return;
    }
    if (liveRequestBusy) {
      liveRequestBusy = false;
    }
    drainMappedControlQueue();
  };

  const previewTabList = () => document.querySelector('.wa-preview-panel .tabs [role="tablist"]');

  const previewTabButtons = () =>
    Array.from(previewTabList()?.querySelectorAll('button[role="tab"]') || []);

  const getActiveTabName = () =>
    previewTabList()?.querySelector('button[role="tab"][aria-selected="true"]')?.textContent.trim()
    || "Preview Video";

  const selectPreviewTab = (name) => {
    const buttons = previewTabButtons();
    const button = buttons.find((node) => node.textContent.trim() === name);
    if (button instanceof HTMLButtonElement) {
      button.click();
      return true;
    }
    return false;
  };

  const getSpatialShell = () => document.getElementById("wa-spatial-shell");

  const hasPointsViewportSurface = () =>
    Boolean(document.querySelector(".wa-stage-points-host section.wa-points-viewport"));

  const hasEmbodiedViewportSurface = () =>
    Boolean(document.querySelector(".wa-stage-embodied-host section.wa-embodied-viewport"));

  const spatialWorldActive = () => getActiveTabName() === "3D World";

  const spatialKeyboardActive = () =>
    spatialWorldActive() && Boolean(getSpatialShell()?.dataset.splatUrl);

  const getModelTitle = () =>
    truncateWorldLabel(document.querySelector(".wa-left-rail .wa-profile h3")?.textContent || "");

  const collectVisualSources = () => {
    const allImages = Array.from(document.querySelectorAll(".wa-preview-panel img"))
      .filter((img) => img.closest(".wa-world-tray") === null)
      .map((img) => img.currentSrc || img.getAttribute("src") || "")
      .filter(Boolean);
    const uniqueImages = allImages.filter((src, index) => allImages.indexOf(src) === index);
    const primaryImageNode = document.querySelector("#wa-main-preview-image img");
    const primaryImage = primaryImageNode
      ? (primaryImageNode.currentSrc || primaryImageNode.getAttribute("src") || "")
      : "";
    const gallery = uniqueImages.filter((src) => src !== primaryImage);
    const videoNode = document.querySelector("#wa-main-preview-video video");
    const videoPoster = videoNode ? (videoNode.getAttribute("poster") || "") : "";
    const spatialShell = getSpatialShell();
    const worldPoster = spatialShell?.dataset.posterUrl || primaryImage || gallery[0] || videoPoster || "";
    return {
      video: videoPoster || gallery[0] || primaryImage,
      image: primaryImage || gallery[0] || "",
      model: primaryImage || gallery[0] || "",
      world: worldPoster,
      gallery,
    };
  };

  const hasPreviewVideo = () => {
    const video = document.querySelector("#wa-main-preview-video video");
    return Boolean(video && (video.currentSrc || video.getAttribute("src") || video.getAttribute("poster")));
  };

  const hasPreviewImage = () =>
    Boolean(document.querySelector("#wa-main-preview-image img")?.getAttribute("src"));

  const hasModelPreview = () =>
    Boolean(
      document.querySelector(
        ".wa-stage-model model-viewer[src], .wa-stage-model canvas, .wa-stage-model img"
      )
    );

  const hasSpatialStage = () =>
    Boolean(document.querySelector(".wa-stage-splat-host #wa-spatial-shell"));

  const hasSpatialPreview = () =>
    Boolean(getSpatialShell()?.dataset.splatUrl) || hasModelPreview() || hasSpatialStage();

  const hasArtifacts = () =>
    Boolean(
      document.querySelector(
        ".wa-stage-artifacts .file-preview, .wa-stage-artifacts .file, .wa-stage-artifacts pre, .wa-stage-artifacts code"
      )
    );

  const currentTemplate = () => {
    const panel = document.querySelector(".wa-preview-panel");
    if (!(panel instanceof HTMLElement)) return "";
    const templateClass = Array.from(panel.classList).find((item) => item.startsWith("wa-template-"));
    return templateClass ? templateClass.replace("wa-template-", "") : "";
  };

  const templateDefaultTab = (template) => {
    if (template === "scene-3d") return "3D World";
    if (template === "depth-geometry") return "Preview Image";
    if (template === "embodied-policy") return "Embodied Sim";
    if (template === "visual-action" || template === "hosted-api") return "Artifacts";
    return "Preview Video";
  };

  const templateIdleCopy = (template) => {
    if (template === "scene-3d") return "Stage source media or import a 3DGS asset.";
    if (template === "depth-geometry") return "Stage image, video, folder, or data path.";
    if (template === "embodied-policy") return "Stage observation context for policy inference.";
    if (template === "visual-action") return "Stage visual context for action-token inference.";
    if (template === "hosted-api") return "Configure provider request inputs.";
    if (template === "conditioned-video" || template === "text-video") return "Stage prompt and media conditions.";
    return "Upload a scene and start generating.";
  };

  const syncTemplateStage = () => {
    const panel = document.querySelector(".wa-preview-panel");
    if (!(panel instanceof HTMLElement)) return;
    const template = currentTemplate();
    panel.dataset.waTemplate = template || "interactive-world";
    if (!template) return;
    const targetTab = templateDefaultTab(template);
    if (panel.dataset.waTemplateTabInit !== template && getActiveTabName() !== targetTab) {
      if (selectPreviewTab(targetTab)) {
        panel.dataset.waTemplateTabInit = template;
      }
    }
  };

  const activeTabHasVisualMedia = () => {
    const activeTab = getActiveTabName();
    if (activeTab === "Preview Video") {
      return hasPreviewVideo();
    }
    if (activeTab === "Preview Image") {
      return hasPreviewImage();
    }
    if (activeTab === "3D World") {
      return hasSpatialPreview();
    }
    if (activeTab === "Point Cloud (Viser)") {
      return hasPointsViewportSurface();
    }
    if (activeTab === "Embodied Sim") {
      return hasEmbodiedViewportSurface();
    }
    if (activeTab === "Gallery") {
      return collectVisualSources().gallery.length > 0;
    }
    if (activeTab === "Artifacts") {
      return hasArtifacts();
    }
    return false;
  };

  const setTrayThumb = (item, src) => {
    if (!item) return;
    if (src) {
      item.dataset.inputSource = src;
      item.style.setProperty("--wa-thumb-image", `url("${src.replace(/"/g, '\\"')}")`);
      item.classList.add("has-thumb");
      return;
    }
    item.dataset.inputSource = "";
    item.style.removeProperty("--wa-thumb-image");
    item.classList.remove("has-thumb");
  };

  const parseCssUrl = (value) => {
    const text = (value || "").trim();
    if (!text || text === "none") return "";
    const match = text.match(/^url\\((['"]?)(.*)\\1\\)$/);
    return match ? match[2] : "";
  };

  const trayImageSourceForItem = (item) => {
    if (!(item instanceof HTMLElement)) return "";
    const direct = item.dataset.inputSource || "";
    if (direct) return direct;
    const thumb = item.querySelector(".wa-tray-thumb");
    if (!(thumb instanceof HTMLElement)) return "";
    return parseCssUrl(window.getComputedStyle(thumb).backgroundImage);
  };

  const pushTrayImageToInput = (source) => {
    if (!source) return;
    const sourceHost = document.getElementById("wa-tray-image-source");
    const applyHost = document.getElementById("wa-tray-image-apply");
    const sourceInput =
      sourceHost instanceof HTMLInputElement || sourceHost instanceof HTMLTextAreaElement
        ? sourceHost
        : sourceHost?.querySelector("textarea, input");
    const applyButton =
      applyHost instanceof HTMLButtonElement ? applyHost : applyHost?.querySelector("button");
    if (!sourceInput || !(applyButton instanceof HTMLButtonElement)) return;
    sourceInput.value = source;
    sourceInput.dispatchEvent(new Event("input", { bubbles: true }));
    sourceInput.dispatchEvent(new Event("change", { bubbles: true }));
    window.setTimeout(() => applyButton.click(), 0);
  };

  window.waHandleTrayClick = (target) => {
    if (!(target instanceof HTMLElement)) return false;
    const tabName = target.getAttribute("data-tab-target");
    if (!tabName) return false;
    document.body.dataset.waActiveTray = target.getAttribute("data-thumb-source") || tabName;
    const buttons = previewTabButtons();
    const button = buttons.find((node) => node.textContent.trim() === tabName);
    if (button instanceof HTMLButtonElement) {
      button.click();
      window.setTimeout(requestSync, 20);
    }
    const trayImageSource = trayImageSourceForItem(target);
    if (
      trayImageSource
      && (
        tabName === "Preview Video"
        || tabName === "Preview Image"
        || tabName === "3D World"
        || tabName === "Point Cloud (Viser)"
        || tabName === "Embodied Sim"
        || tabName === "Gallery"
      )
    ) {
      pushTrayImageToInput(trayImageSource);
    }
    return false;
  };

  const syncTrayThumbs = () => {
    const sources = collectVisualSources();
    document.querySelectorAll(".wa-tray-item").forEach((item) => {
      const thumbSource = item.dataset.thumbSource || "";
      let src = "";
      if (thumbSource === "video") {
        src = sources.video;
      } else if (thumbSource === "image") {
        src = sources.image;
      } else if (thumbSource === "world") {
        src = sources.world || sources.image || sources.video;
      } else if (thumbSource === "points") {
        src = sources.world || sources.image || sources.video;
      } else if (thumbSource === "embodied") {
        src = sources.video || sources.image || sources.world;
      } else if (thumbSource === "model") {
        src = sources.model;
      } else if (thumbSource.startsWith("gallery-")) {
        const galleryIndex = Number(thumbSource.split("-")[1] || "0");
        src = sources.gallery[galleryIndex] || sources.image || sources.video;
      } else if (thumbSource === "artifacts") {
        src = sources.gallery[0] || sources.image;
      }
      setTrayThumb(item, src);
    });
  };

  const syncTraySelection = () => {
    const activeName = getActiveTabName();
    const panel = document.querySelector(".wa-preview-panel");
    if (panel instanceof HTMLElement) {
      panel.dataset.waActiveTab = activeName;
    }
    const items = Array.from(document.querySelectorAll(".wa-tray-item"));
    const remembered = document.body.dataset.waActiveTray || "";
    let activeItem = null;
    if (remembered) {
      activeItem = items.find(
        (item) => item.dataset.thumbSource === remembered && item.dataset.tabTarget === activeName
      );
    }
    if (!activeItem) {
      activeItem = items.find((item) => item.dataset.tabTarget === activeName) || null;
    }
    items.forEach((item) => item.classList.toggle("is-active", item === activeItem));
  };

  const installTrayBinding = () => {
    if (document.body.dataset.waTrayBound === "1") return;
    document.body.dataset.waTrayBound = "1";
  };

  const bindNoticeClose = () => {
    if (document.body.dataset.waNoticeBound === "1") return;
    document.body.dataset.waNoticeBound = "1";
    document.addEventListener("click", (event) => {
      const target = event.target instanceof Element ? event.target.closest(".wa-notice-close") : null;
      if (!target) return;
      document.getElementById("wa-notice-bar")?.classList.add("is-hidden");
    });
  };

  const currentTheme = () =>
    document.documentElement.dataset.waTheme === "light" ? "light" : "dark";

  const joystickDockOpen = () => {
    const viewport = document.documentElement.dataset.waViewport || "wide";
    const fallback = viewport === "phone" || viewport === "narrow" ? "closed" : "open";
    return readPreference(JOYSTICK_STORAGE_KEY, fallback) === "open";
  };

  const stageFocusActive = () => readPreference(FOCUS_STORAGE_KEY, "off") === "on";

  const performanceMode = () => readPreference(PERFORMANCE_STORAGE_KEY, "balanced") === "lite" ? "lite" : "balanced";

  const syncChromeState = () => {
    const focus = stageFocusActive();
    const performance = performanceMode();
    document.documentElement.dataset.waFocus = focus ? "stage" : "default";
    document.documentElement.dataset.waPerformance = performance;
    document.body.classList.toggle("wa-stage-focus", focus);
    document.body.classList.toggle("wa-performance-lite", performance === "lite");
    document.querySelectorAll(".wa-main-grid, .wa-preview-panel").forEach((node) => {
      if (!(node instanceof HTMLElement)) return;
      node.dataset.waFocus = focus ? "stage" : "default";
      node.dataset.waPerformance = performance;
    });
  };

  const setTheme = (theme) => {
    const next = theme === "light" ? "light" : "dark";
    document.documentElement.dataset.waTheme = next;
    writePreference(THEME_STORAGE_KEY, next);
    syncNavState();
  };

  const toggleTheme = () => {
    setTheme(currentTheme() === "light" ? "dark" : "light");
  };

  const setJoystickDockOpen = (open) => {
    writePreference(JOYSTICK_STORAGE_KEY, open ? "open" : "closed");
    const panel = document.querySelector(".wa-preview-panel");
    if (panel) {
      panel.classList.toggle("wa-joystick-open", !!open);
    }
    syncNavState();
    requestSync();
  };

  const setStageFocus = (enabled) => {
    writePreference(FOCUS_STORAGE_KEY, enabled ? "on" : "off");
    syncChromeState();
    syncNavState();
    requestSync();
  };

  const toggleStageFocus = () => {
    setStageFocus(!stageFocusActive());
  };

  const setPerformanceMode = (mode) => {
    writePreference(PERFORMANCE_STORAGE_KEY, mode === "lite" ? "lite" : "balanced");
    syncChromeState();
    syncNavState();
    window.dispatchEvent(new CustomEvent("wa-performance-change"));
    requestSync();
  };

  const togglePerformanceMode = () => {
    setPerformanceMode(performanceMode() === "lite" ? "balanced" : "lite");
  };

  const syncNavState = () => {
    const joystickButton = document.querySelector('[data-wa-nav="joystick"]');
    const themeButton = document.querySelector('[data-wa-nav="theme"]');
    const focusButton = document.querySelector('[data-wa-nav="focus"]');
    const performanceButton = document.querySelector('[data-wa-nav="performance"]');
    const joystickOpen = !spatialWorldActive() && joystickDockOpen() && joystickControlsAvailable();
    const lightTheme = currentTheme() === "light";
    const focus = stageFocusActive();
    const performance = performanceMode();

    if (joystickButton) {
      joystickButton.classList.toggle("is-active", joystickOpen);
      joystickButton.setAttribute("aria-pressed", joystickOpen ? "true" : "false");
      joystickButton.setAttribute("aria-label", joystickOpen ? "Hide joystick" : "Show joystick");
      joystickButton.setAttribute("title", joystickOpen ? "Hide joystick" : "Show joystick");
    }

    if (themeButton) {
      themeButton.classList.toggle("is-active", lightTheme);
      themeButton.setAttribute("aria-pressed", lightTheme ? "true" : "false");
      themeButton.setAttribute("aria-label", lightTheme ? "Switch to dark" : "Switch to light");
      themeButton.setAttribute("title", lightTheme ? "Switch to dark" : "Switch to light");
    }

    if (focusButton) {
      focusButton.classList.toggle("is-active", focus);
      focusButton.setAttribute("aria-pressed", focus ? "true" : "false");
      focusButton.setAttribute("aria-label", focus ? "Show rails" : "Focus stage");
      focusButton.setAttribute("title", focus ? "Show rails" : "Focus stage");
    }

    if (performanceButton) {
      performanceButton.classList.toggle("is-active", performance === "lite");
      performanceButton.setAttribute("aria-pressed", performance === "lite" ? "true" : "false");
      performanceButton.setAttribute(
        "aria-label",
        performance === "lite" ? "Use balanced rendering" : "Use lighter rendering"
      );
      performanceButton.setAttribute(
        "title",
        performance === "lite" ? "Use balanced rendering" : "Use lighter rendering"
      );
    }
  };

  const syncJoystickDock = () => {
    const panel = document.querySelector(".wa-preview-panel");
    const dock = document.getElementById("wa-joystick-dock");
    const note = document.getElementById("wa-joystick-note");
    if (!panel || !dock) return;

    const enabledControls = Object.keys(keyMap).filter((key) => {
      const button = controlButtonForKey(key);
      return controlButtonIsReady(button);
    }).length;
    const open = !spatialWorldActive() && joystickDockOpen() && enabledControls > 0;
    panel.classList.toggle("wa-joystick-open", open);
    dock.setAttribute("aria-hidden", open ? "false" : "true");
    dock.classList.toggle("is-disabled", enabledControls === 0);

    if (note) {
      note.textContent = spatialWorldActive()
        ? "3D world mode uses drag, wheel, and WASD directly inside the Spark viewer."
        : enabledControls
        ? "Use the lower move and look sticks, or hold keyboard directions to autoroll step by step."
        : "Pick a live navigation model to unlock move and camera sticks.";
    }
  };

  const bindNavActions = () => {
    if (document.body.dataset.waNavBound === "1") return;
    document.body.dataset.waNavBound = "1";
    document.addEventListener("click", (event) => {
      const target = event.target instanceof Element ? event.target.closest("[data-wa-nav]") : null;
      if (!target) return;
      const action = target.getAttribute("data-wa-nav");
      if (action === "joystick") {
        event.preventDefault();
        setJoystickDockOpen(!joystickDockOpen());
      }
      if (action === "theme") {
        event.preventDefault();
        toggleTheme();
      }
      if (action === "focus") {
        event.preventDefault();
        toggleStageFocus();
      }
      if (action === "performance") {
        event.preventDefault();
        togglePerformanceMode();
      }
    });
  };

  const UI_TEXT_REPLACEMENTS = new Map([
    ["将图像拖放到此处 - 或 - 点击上传", "Drop image here or click to upload"],
    ["将文件拖放到此处 - 或 - 点击上传", "Drop file here or click to upload"],
    ["将视频拖放到此处 - 或 - 点击上传", "Drop video here or click to upload"],
    ["通过 API 使用", "Use via API"],
  ]);

  const translateUiToEnglish = () => {
    document.querySelectorAll(".gradio-container *").forEach((node) => {
      if (!(node instanceof HTMLElement)) return;
      const ariaLabel = node.getAttribute("aria-label");
      const title = node.getAttribute("title");
      if (ariaLabel === "清除") {
        node.setAttribute("aria-label", "Clear");
      }
      if (title === "清除") {
        node.setAttribute("title", "Clear");
      }
      if (node.children.length) return;
      const text = normalizeText(node.textContent || "");
      const replacement = UI_TEXT_REPLACEMENTS.get(text);
      if (replacement && node.textContent !== replacement) {
        node.textContent = replacement;
      }
    });
  };

  const bindMediaListeners = () => {
    const video = document.querySelector("#wa-main-preview-video video");
    if (video && video.dataset.waBound !== "1") {
      video.dataset.waBound = "1";
      [
        "loadedmetadata",
        "durationchange",
        "emptied",
        "loadeddata",
      ].forEach((name) => {
        video.addEventListener(name, () => {
          trayThumbsDirty = true;
          requestSync();
        });
      });
      [
        "timeupdate",
        "play",
        "pause",
        "ended",
        "seeking",
        "seeked",
      ].forEach((name) => {
        video.addEventListener(name, requestSync);
      });
    }
    Array.from(document.querySelectorAll(".wa-preview-panel img")).forEach((img) => {
      if (img.dataset.waBound === "1") return;
      img.dataset.waBound = "1";
      img.addEventListener("load", () => {
        trayThumbsDirty = true;
        requestSync();
      });
      img.addEventListener("error", () => {
        trayThumbsDirty = true;
        requestSync();
      });
    });
  };

  const syncFooterMeta = () => {
    const panel = document.querySelector(".wa-preview-panel");
    if (!panel) return;
    const info = inferStageState(getStatusText());
    panel.dataset.state = info.state;
    const showWorldLabel = !activeTabHasVisualMedia() && info.state !== "input";
    const showTime = getActiveTabName() === "Preview Video" && hasPreviewVideo();

    const statusNode = document.getElementById("wa-player-footer-status");
    const worldNode = document.getElementById("wa-player-footer-world");
    const timeNode = document.getElementById("wa-player-footer-time");
    if (statusNode) {
      statusNode.textContent = info.label;
    }
    if (worldNode) {
      worldNode.textContent = showWorldLabel ? getModelTitle() : "";
      worldNode.classList.toggle("is-hidden", !showWorldLabel);
    }
    if (timeNode) {
      const video = document.querySelector("#wa-main-preview-video video");
      if (!showTime) {
        timeNode.textContent = "";
        timeNode.classList.add("is-hidden");
        return;
      }
      const currentTime = video && Number.isFinite(video.currentTime) && video.currentTime >= 0
        ? formatDuration(video.currentTime)
        : "0:00";
      const duration = video && Number.isFinite(video.duration) && video.duration > 0
        ? formatDuration(video.duration)
        : "";
      timeNode.textContent = duration ? `${currentTime} / ${duration}` : currentTime;
      timeNode.classList.toggle("is-hidden", !showTime);
    }
  };

  const syncEmptyState = () => {
    const panel = document.querySelector(".wa-preview-panel");
    const emptyShell = document.getElementById("wa-stage-empty");
    const emptyTitle = document.getElementById("wa-stage-empty-title");
    const emptyCopy = document.getElementById("wa-stage-empty-copy");
    if (!panel || !emptyShell || !emptyTitle || !emptyCopy) return;

    const statusText = getStatusText();
    const info = inferStageState(statusText);
    const activeTab = getActiveTabName();
    let hasMedia = false;

    if (activeTab === "Preview Video") {
      hasMedia = hasPreviewVideo();
    } else if (activeTab === "Preview Image") {
      hasMedia = hasPreviewImage();
    } else if (activeTab === "3D World") {
      hasMedia = hasSpatialPreview();
    } else if (activeTab === "Point Cloud (Viser)") {
      hasMedia = hasPointsViewportSurface();
    } else if (activeTab === "Embodied Sim") {
      hasMedia = hasEmbodiedViewportSurface();
    } else if (activeTab === "Gallery") {
      hasMedia = collectVisualSources().gallery.length > 0;
    } else if (activeTab === "Artifacts") {
      hasMedia = hasArtifacts();
    }

    panel.classList.toggle("wa-has-media", hasMedia);
    panel.dataset.state = info.state;
    if (info.state === "loading" || info.state === "queue") {
      emptyTitle.textContent = "";
      emptyCopy.textContent = "";
      return;
    }
    emptyTitle.textContent = info.state === "error"
      ? "ERROR"
      : info.state === "input"
        ? "INPUT READY"
        : info.state === "idle"
          ? "READY"
          : "";
    if (info.state === "error") {
      emptyCopy.textContent = statusText || "The selected run could not be completed.";
      return;
    }
    if (info.state === "queue") {
      emptyCopy.textContent = statusText || "In queue…";
      return;
    }
    if (info.state === "input") {
      emptyCopy.textContent = statusText || templateIdleCopy(currentTemplate());
      return;
    }
    emptyCopy.textContent = statusText && statusText.toLowerCase() !== "idle"
      ? statusText
      : templateIdleCopy(currentTemplate());
  };

  const syncPreviewFallback = () => {
    const info = inferStageState(getStatusText());
    const activeTab = getActiveTabName();
    const hasVideo = hasPreviewVideo();
    const hasImage = hasPreviewImage();
    const fallbackActive = document.body.dataset.waPreviewFallbackActive === "1";

    if (info.state === "input" && hasImage && activeTab !== "Preview Image") {
      if (selectPreviewTab("Preview Image")) {
        document.body.dataset.waPreviewFallbackActive = "1";
      }
      return;
    }

    if ((info.state === "loading" || info.state === "queue") && activeTab === "Preview Video" && !hasVideo && hasImage) {
      if (selectPreviewTab("Preview Image")) {
        document.body.dataset.waPreviewFallbackActive = "1";
      }
      return;
    }

    if (fallbackActive && hasVideo && activeTab === "Preview Image") {
      if (selectPreviewTab("Preview Video")) {
        document.body.dataset.waPreviewFallbackActive = "0";
      }
      return;
    }

    if (info.state === "error" || (!hasImage && !hasVideo)) {
      document.body.dataset.waPreviewFallbackActive = "0";
    }
  };

  const requestSync = (options = {}) => {
    if (options.ui) uiTranslationDirty = true;
    if (options.tray) trayThumbsDirty = true;
    if (syncQueued) return;
    syncQueued = true;
    window.clearTimeout(syncTimer);
    syncTimer = window.setTimeout(() => {
      syncQueued = false;
      syncTimer = null;
      syncViewportMode();
      syncChromeState();
      mountVirtualControls();
      bindNoticeClose();
      bindNavActions();
      if (uiTranslationDirty) {
        translateUiToEnglish();
        uiTranslationDirty = false;
      }
      installTrayBinding();
      bindMediaListeners();
      syncTemplateStage();
      syncPreviewFallback();
      if (trayThumbsDirty) {
        syncTrayThumbs();
        trayThumbsDirty = false;
      }
      syncTraySelection();
      syncFooterMeta();
      syncEmptyState();
      syncJoystickDock();
      syncNavState();
      syncLiveRequestState();
    }, performanceMode() === "lite" ? 84 : 48);
  };

  document.addEventListener("keydown", (event) => {
    const key = event.key.toLowerCase();
    if (event.repeat || !keyMap[key] || isEditableTarget(document.activeElement)) return;
    if (spatialKeyboardActive()) return;
    event.preventDefault();
    setLiveKeyActive(key, true);
    syncKeyboardStickVisual(key, true);
    queueMappedControl(key);
  });

  document.addEventListener("keyup", (event) => {
    const key = event.key.toLowerCase();
    if (!keyMap[key]) return;
    setLiveKeyActive(key, false);
    syncKeyboardStickVisual(key, false);
  });

  window.addEventListener("blur", () => {
    activeLiveKeys.clear();
    liveQueuedKey = "";
    liveLastIntentKey = "";
    liveRequestBusy = false;
    liveDispatchCooldownUntil = 0;
    resetAllKeyboardStickVisuals();
  });

  const observer = new MutationObserver((records) => {
    const shouldRefreshDomText = records.some((record) => (
      record.type === "childList"
      || record.attributeName === "aria-label"
      || record.attributeName === "title"
    ));
    const shouldRefreshTray = records.some((record) => (
      record.type === "childList"
      || record.attributeName === "src"
      || record.attributeName === "poster"
    ));
    requestSync({ ui: shouldRefreshDomText, tray: shouldRefreshTray });
  });
  observer.observe(document.querySelector(".gradio-container") || document.body || document.documentElement, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["class", "src", "poster", "aria-label", "title"],
  });
  window.addEventListener("load", () => {
    syncViewportMode();
    syncChromeState();
    syncNavState();
    requestSync({ ui: true, tray: true });
  });
  window.setInterval(() => {
    syncLiveRequestState();
  }, 140);
  window.addEventListener("resize", () => {
    window.clearTimeout(resizeSyncTimer);
    resizeSyncTimer = window.setTimeout(() => {
      syncViewportMode();
      requestSync({ tray: true });
    }, 90);
  });
  window.setTimeout(() => {
    requestSync({ ui: true, tray: true });
  }, 240);
})();
</script>
<script type="module">
(() => {
  let syncQueued = false;
  let runtimePromise = null;
  let viewerOnscreen = true;
  let viewerIntersectionObserver = null;
  const VIEWER_KEY = "__waSparkViewer";
  const PERFORMANCE_STORAGE_KEY = "worldfoundry-studio-performance-mode";

  const getPanel = () => document.querySelector(".wa-preview-panel");
  const sparkTabList = () => document.querySelector('.wa-preview-panel .tabs [role="tablist"]');
  const getActiveTabName = () =>
    sparkTabList()?.querySelector('button[role="tab"][aria-selected="true"]')?.textContent.trim()
    || "Preview Video";
  const getShell = () => document.getElementById("wa-spatial-shell");
  const spatialWorldActive = () => getActiveTabName() === "3D World";

  const readPreference = (key, fallback = "") => {
    try {
      const value = window.localStorage.getItem(key);
      return value == null ? fallback : value;
    } catch (error) {
      return fallback;
    }
  };

  const performanceMode = () => readPreference(PERFORMANCE_STORAGE_KEY, "balanced") === "lite" ? "lite" : "balanced";

  const resolveViewerPixelRatio = () => {
    const devicePixelRatio = Math.max(1, window.devicePixelRatio || 1);
    return Math.min(devicePixelRatio, performanceMode() === "lite" ? 1 : 1.5);
  };

  const viewerShouldRun = () =>
    spatialWorldActive() && document.visibilityState !== "hidden" && viewerOnscreen;

  const setPanelSpatial = (value) => {
    const panel = getPanel();
    if (panel) {
      panel.dataset.waSpatial = value || "empty";
    }
  };

  const updatePoster = (shell) => {
    if (!(shell instanceof HTMLElement)) return;
    const poster = shell.dataset.posterUrl || "";
    if (poster) {
      shell.style.setProperty("--wa-spatial-poster", `url("${poster.replace(/"/g, '\\"')}")`);
      return;
    }
    shell.style.removeProperty("--wa-spatial-poster");
  };

  const updateShellState = (shell, mode, message) => {
    if (!(shell instanceof HTMLElement)) return;
    shell.classList.toggle("is-loading", mode === "loading");
    shell.classList.toggle("is-ready", mode === "ready");
    shell.classList.toggle("is-empty", mode === "empty" || mode === "mesh");
    shell.classList.toggle("is-error", mode === "error");
    const copy = shell.querySelector("#wa-spark-copy");
    const loading = shell.querySelector("#wa-spark-loading");
    if (copy && message) {
      copy.textContent = message;
    }
    if (loading) {
      loading.textContent = mode === "loading"
        ? (message || "Loading 3DGS…")
        : mode === "error"
          ? "3DGS Error"
          : "3DGS Ready";
    }
  };

  const loadRuntime = () => {
    if (!runtimePromise) {
      runtimePromise = Promise.all([
        import("three"),
        import("@sparkjsdev/spark"),
      ]).then(([THREE, Spark]) => ({ THREE, Spark })).catch((error) => {
        runtimePromise = null;
        const detail = error instanceof Error ? error.message : String(error);
        throw new Error(
          `3DGS browser modules unavailable. Run __WA_VENDOR_INSTALL_COMMAND__, then refresh this page. ${detail}`
        );
      });
    }
    return runtimePromise;
  };

  const resizeViewer = (viewer = window[VIEWER_KEY]) => {
    if (!viewer) return false;
    const rect = viewer.canvas.getBoundingClientRect();
    const width = Math.max(1, Math.round(rect.width));
    const height = Math.max(1, Math.round(rect.height));
    if (!width || !height) return false;
    const pixelRatio = resolveViewerPixelRatio();
    if (viewer.pixelRatio !== pixelRatio) {
      viewer.renderer.setPixelRatio(pixelRatio);
      viewer.pixelRatio = pixelRatio;
    }
    if (viewer.width !== width || viewer.height !== height) {
      viewer.renderer.setSize(width, height, false);
      viewer.camera.aspect = width / height;
      viewer.camera.updateProjectionMatrix();
      viewer.width = width;
      viewer.height = height;
    }
    return true;
  };

  const pauseViewer = (viewer = window[VIEWER_KEY]) => {
    if (!viewer || !viewer.running) return;
    viewer.running = false;
    viewer.controls.fpsMovement.enable = false;
    viewer.renderer.setAnimationLoop(null);
  };

  const disposeViewer = () => {
    const viewer = window[VIEWER_KEY];
    if (!viewer) return;
    pauseViewer(viewer);
    try {
      viewer.scene.remove(viewer.splat);
      viewer.splat?.dispose?.();
    } catch (error) {
      console.error("Failed to dispose Spark splat:", error);
    }
    try {
      viewer.spark?.dispose?.();
    } catch (error) {
      console.error("Failed to dispose Spark renderer:", error);
    }
    try {
      viewer.renderer?.dispose?.();
    } catch (error) {
      console.error("Failed to dispose WebGL renderer:", error);
    }
    if (viewerIntersectionObserver) {
      viewerIntersectionObserver.disconnect();
      viewerIntersectionObserver = null;
    }
    window[VIEWER_KEY] = null;
  };

  const resumeViewer = (viewer = window[VIEWER_KEY]) => {
    if (!viewer || viewer.running || !viewerShouldRun()) return;
    viewer.running = true;
    viewer.controls.fpsMovement.enable = true;
    viewer.renderer.setAnimationLoop(() => {
      if (!viewerShouldRun()) {
        pauseViewer(viewer);
        return;
      }
      if (!resizeViewer(viewer)) return;
      viewer.controls.update(viewer.camera);
      viewer.renderer.render(viewer.scene, viewer.camera);
    });
  };

  const observeViewerCanvas = (canvas) => {
    if (viewerIntersectionObserver) {
      viewerIntersectionObserver.disconnect();
      viewerIntersectionObserver = null;
    }
    viewerOnscreen = true;
    if (!("IntersectionObserver" in window)) return;
    viewerIntersectionObserver = new IntersectionObserver((entries) => {
      viewerOnscreen = entries.some((entry) => entry.isIntersecting);
      if (!viewerOnscreen) {
        pauseViewer();
      }
      requestSync();
    }, { threshold: 0.01 });
    viewerIntersectionObserver.observe(canvas);
  };

  const syncViewer = async () => {
    const shell = getShell();
    if (!shell) {
      setPanelSpatial("empty");
      disposeViewer();
      return;
    }

    updatePoster(shell);
    const splatUrl = shell.dataset.splatUrl || "";
    const resolvedSplatUrl = splatUrl
      ? new URL(splatUrl, window.location.href).toString()
      : "";
    const note = shell.dataset.note || "";
    const kind = shell.dataset.kind || "empty";
    const canvas = shell.querySelector("#wa-spark-canvas");

    if (!splatUrl || !(canvas instanceof HTMLCanvasElement)) {
      delete shell.dataset.sparkFailure;
      setPanelSpatial(kind === "mesh" ? "mesh" : "empty");
      disposeViewer();
      updateShellState(
        shell,
        kind === "mesh" ? "mesh" : "empty",
        note || (kind === "mesh"
          ? "This run has a mesh preview but no Gaussian Splat export yet."
          : "Import a Gaussian Splat to continue the world in 3D.")
      );
      return;
    }

    let viewer = window[VIEWER_KEY];
    if (viewer && viewer.assetUrl === resolvedSplatUrl && viewer.canvas === canvas) {
      if (viewer.loaded) {
        delete shell.dataset.sparkFailure;
      }
      setPanelSpatial("splat");
      updateShellState(shell, viewer.loaded ? "ready" : "loading", note);
      resizeViewer(viewer);
      if (viewerShouldRun()) {
        resumeViewer(viewer);
      } else {
        pauseViewer(viewer);
      }
      return;
    }

    if (!spatialWorldActive()) {
      setPanelSpatial("splat");
      updateShellState(
        shell,
        "loading",
        note || "3DGS asset is attached. Open the 3D World tray tile to initialize Spark."
      );
      return;
    }

    if (shell.dataset.sparkFailure === resolvedSplatUrl) {
      setPanelSpatial("error");
      updateShellState(
        shell,
        "error",
        note || "The last Spark load failed for this asset. Re-import it after fixing the runtime."
      );
      return;
    }

    disposeViewer();
    setPanelSpatial("splat");
    updateShellState(shell, "loading", "Loading Gaussian Splat…");

    try {
      const { THREE, Spark } = await loadRuntime();
      if (!document.body.contains(canvas)) return;

      const scene = new THREE.Scene();

      const camera = new THREE.PerspectiveCamera(75, 1, 0.01, 1000);
      const renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: performanceMode() !== "lite",
        powerPreference: performanceMode() === "lite" ? "low-power" : "high-performance",
      });
      renderer.setPixelRatio(resolveViewerPixelRatio());

      camera.position.set(0, 0, 1);

      const spark = new Spark.SparkRenderer({ renderer });
      scene.add(spark);

      const controls = new Spark.SparkControls({ canvas });

      viewer = {
        assetUrl: resolvedSplatUrl,
        canvas,
        shell,
        scene,
        camera,
        renderer,
        spark,
        controls,
        loaded: false,
        running: false,
        width: 0,
        height: 0,
        pixelRatio: resolveViewerPixelRatio(),
      };
      window[VIEWER_KEY] = viewer;
      observeViewerCanvas(canvas);
      resizeViewer(viewer);
      if (viewerShouldRun()) {
        resumeViewer(viewer);
      }

      const fileName = splatUrl.split("/").pop()?.split("?")[0] || "world.splat";
      const splat = new Spark.SplatMesh({
        url: resolvedSplatUrl,
        fileName,
        onProgress: (event) => {
          if (window[VIEWER_KEY] !== viewer) return;
          const loaded = Number(event?.loaded || 0);
          const total = Number(event?.total || 0);
          const message = total > 0
            ? `Loading Gaussian Splat… ${Math.round((loaded / total) * 100)}%`
            : "Loading Gaussian Splat…";
          updateShellState(shell, "loading", message);
        },
      });
      splat.quaternion.set(1, 0, 0, 0);
      viewer.splat = splat;
      scene.add(splat);

      void splat.initialized.then(() => {
        if (window[VIEWER_KEY] !== viewer) {
          splat.dispose?.();
          renderer.dispose?.();
          return;
        }

        viewer.camera.position.set(0, 0, 1);
        viewer.camera.lookAt(0, 0, 0);
        viewer.controls.lastTime = 0;
        viewer.loaded = true;
        delete shell.dataset.sparkFailure;
        updateShellState(
          shell,
          "ready",
          note || "Drag to look, wheel to dolly, and use WASD to move through the world."
        );
        if (viewerShouldRun()) {
          resumeViewer(viewer);
        }
      }).catch((error) => {
        if (window[VIEWER_KEY] !== viewer) {
          return;
        }
        console.error("WorldFoundry Spark viewer failed:", error);
        disposeViewer();
        shell.dataset.sparkFailure = resolvedSplatUrl;
        setPanelSpatial("error");
        updateShellState(
          shell,
          "error",
          error instanceof Error ? error.message : "Could not load the Gaussian Splat asset."
        );
      });
    } catch (error) {
      console.error("WorldFoundry Spark viewer failed:", error);
      disposeViewer();
      shell.dataset.sparkFailure = resolvedSplatUrl;
      setPanelSpatial("error");
      updateShellState(
        shell,
        "error",
        error instanceof Error ? error.message : "Could not load the Gaussian Splat asset."
      );
    }
  };

  const requestSync = () => {
    if (syncQueued) return;
    syncQueued = true;
    window.requestAnimationFrame(() => {
      syncQueued = false;
      void syncViewer();
    });
  };

  const observer = new MutationObserver(requestSync);
  observer.observe(document.querySelector(".gradio-container") || document.body || document.documentElement, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["class", "src", "poster", "data-splat-url"],
  });

  window.addEventListener("load", requestSync);
  window.addEventListener("resize", requestSync);
  window.addEventListener("wa-performance-change", requestSync);
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") {
      pauseViewer();
      return;
    }
    requestSync();
  });
  window.setTimeout(requestSync, 320);
})();
</script>
""".replace("__WA_THREE_MODULE_URL__", local_module_url(THREE_MODULE_PATH)).replace(
    "__WA_SPARK_MODULE_URL__", local_module_url(SPARK_MODULE_PATH)
).replace(
    "__WA_VENDOR_INSTALL_COMMAND__", VENDOR_ASSET_INSTALL_COMMAND
)

__all__ = ["HEAD_HTML"]
