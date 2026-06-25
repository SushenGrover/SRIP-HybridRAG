/**
 * HybridRAG Frontend — app.js
 * ============================
 * Handles file upload, document management, query submission,
 * rendering of Vector RAG + Graph RAG results, and
 * interactive Knowledge Graph visualisation (vis.js).
 */

const API_BASE = "http://localhost:8000"; // VectorRAG
const GRAPH_API_BASE = "http://localhost:8001"; // GraphRAG
const HYBRID_API_BASE = "http://localhost:8002"; // HybridRAG

// ── State ──────────────────────────────────────────────
let documents = [];
let selectedDocId = null;
let kgNetwork = null; // vis.js Network instance
let kgPhysicsOn = true;

// ── DOM refs ───────────────────────────────────────────
const uploadZone = document.getElementById("uploadZone");
const fileInput = document.getElementById("fileInput");
const uploadProgress = document.getElementById("uploadProgress");
const progressFill = document.getElementById("progressFill");
const uploadStatus = document.getElementById("uploadStatus");
const documentsBar = document.getElementById("documentsBar");
const noDocs = document.getElementById("noDocs");
const queryInput = document.getElementById("queryInput");
const queryBtn = document.getElementById("queryBtn");
const loadingSpinner = document.getElementById("loadingSpinner");
const loadingText = document.getElementById("loadingText");
const vectorBody = document.getElementById("vectorBody");
const vectorSources = document.getElementById("vectorSources");
const vectorTime = document.getElementById("vectorTime");
const graphBody = document.getElementById("graphBody");
const graphTriplets = document.getElementById("graphTriplets");
const graphTime = document.getElementById("graphTime");
const combinedBody = document.getElementById("combinedBody");
const combinedSources = document.getElementById("combinedSources");
const combinedTime = document.getElementById("combinedTime");
const toastContainer = document.getElementById("toastContainer");

// KG Explorer refs
const kgDocSelect = document.getElementById("kgDocSelect");
const kgLoadBtn = document.getElementById("kgLoadBtn");
const kgCanvas = document.getElementById("kgCanvas");
const kgPlaceholder = document.getElementById("kgPlaceholder");
const kgLoading = document.getElementById("kgLoading");
const kgNodeCount = document.getElementById("kgNodeCount");
const kgEdgeCount = document.getElementById("kgEdgeCount");
const kgPhysicsBtn = document.getElementById("kgPhysicsBtn");
const kgFitBtn = document.getElementById("kgFitBtn");
const kgLegend = document.getElementById("kgLegend");

// ── Toast notifications ────────────────────────────────
function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast toast--${type}`;
  toast.textContent = message;
  toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.animation = "toastOut 0.3s ease forwards";
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// ── Upload ─────────────────────────────────────────────
uploadZone.addEventListener("click", () => fileInput.click());

uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadZone.classList.add("dragover");
});

uploadZone.addEventListener("dragleave", () => {
  uploadZone.classList.remove("dragover");
});

uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) uploadFile(fileInput.files[0]);
});

async function uploadFile(file) {
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    showToast("Only PDF files are supported", "error");
    return;
  }

  // Show progress
  uploadProgress.classList.add("active");
  progressFill.style.width = "10%";
  uploadStatus.textContent = `Uploading "${file.name}"...`;

  const formData = new FormData();
  formData.append("file", file);

  // We also need a separate FormData for GraphRAG (can't reuse after read)
  const graphFormData = new FormData();
  graphFormData.append("file", file);

  try {
    // ── Phase 1: VectorRAG upload ──────────────────
    progressFill.style.width = "20%";
    uploadStatus.textContent = "Building vector index...";

    const vectorPromise = fetch(`${API_BASE}/api/upload`, {
      method: "POST",
      body: formData,
    });

    // ── Phase 2: GraphRAG upload (parallel) ────────
    progressFill.style.width = "30%";
    uploadStatus.textContent =
      "Building vector index & extracting knowledge graph...";

    const graphPromise = fetch(`${GRAPH_API_BASE}/api/graph/upload`, {
      method: "POST",
      body: graphFormData,
    }).catch((err) => {
      console.warn(
        "GraphRAG upload failed (server may not be running):",
        err.message,
      );
      return null;
    });

    // Wait for VectorRAG (required)
    const vectorRes = await vectorPromise;
    progressFill.style.width = "60%";

    if (!vectorRes.ok) {
      const err = await vectorRes.json();
      throw new Error(err.detail || "Upload failed");
    }

    const vectorData = await vectorRes.json();

    if (vectorData.status === "already_indexed") {
      showToast(`"${file.name}" is already indexed`, "info");
    } else {
      showToast(
        `"${file.name}" processed — ${vectorData.num_chunks} chunks indexed`,
        "success",
      );
    }

    // Wait for GraphRAG (optional — might fail if server not running)
    progressFill.style.width = "75%";
    uploadStatus.textContent = "Finalizing knowledge graph...";

    const graphRes = await graphPromise;
    if (graphRes && graphRes.ok) {
      const graphData = await graphRes.json();
      if (graphData.status === "already_indexed") {
        showToast(`Knowledge graph already exists for "${file.name}"`, "info");
      } else {
        showToast(
          `Knowledge graph built — ${graphData.num_triplets} triplets extracted`,
          "success",
        );
      }
    } else if (graphRes) {
      const err = await graphRes.json().catch(() => ({}));
      showToast(
        `Graph extraction issue: ${err.detail || "unknown error"}`,
        "error",
      );
    }

    progressFill.style.width = "100%";
    uploadStatus.textContent = "Done!";

    // Refresh document list and select the new doc
    await loadDocuments();
    selectDocument(vectorData.document_id);

    // Refresh KG document list
    await loadKgDocuments();
  } catch (err) {
    showToast(`Upload failed: ${err.message}`, "error");
  } finally {
    setTimeout(() => {
      uploadProgress.classList.remove("active");
      progressFill.style.width = "0%";
    }, 1200);
    fileInput.value = "";
  }
}

// ── Documents ──────────────────────────────────────────
async function loadDocuments() {
  try {
    const res = await fetch(`${API_BASE}/api/documents`);
    if (!res.ok) throw new Error("Failed to load documents");
    const data = await res.json();
    documents = data.documents || [];
    renderDocuments();
  } catch (err) {
    console.warn("Could not load documents:", err.message);
  }
}

function renderDocuments() {
  documentsBar.innerHTML = "";

  if (documents.length === 0) {
    const span = document.createElement("span");
    span.className = "no-docs";
    span.textContent = "No documents uploaded yet";
    documentsBar.appendChild(span);
    queryInput.disabled = true;
    queryBtn.disabled = true;
    return;
  }

  documents.forEach((doc) => {
    const chip = document.createElement("div");
    chip.className =
      "doc-chip" + (doc.document_id === selectedDocId ? " active" : "");
    chip.innerHTML = `
            <span>${doc.filename}</span>
            <span class="doc-pages">${doc.page_count}p · ${doc.num_chunks} chunks</span>
            <span class="doc-delete" title="Delete document">✕</span>
        `;

    chip.addEventListener("click", (e) => {
      if (e.target.classList.contains("doc-delete")) return;
      selectDocument(doc.document_id);
    });

    chip.querySelector(".doc-delete").addEventListener("click", async (e) => {
      e.stopPropagation();
      await deleteDocument(doc.document_id, doc.filename);
    });

    documentsBar.appendChild(chip);
  });
}

function selectDocument(docId) {
  selectedDocId = docId;
  queryInput.disabled = false;
  queryBtn.disabled = false;
  queryInput.focus();
  renderDocuments();
}

async function deleteDocument(docId, filename) {
  try {
    const res = await fetch(`${API_BASE}/api/documents/${docId}`, {
      method: "DELETE",
    });
    if (!res.ok) throw new Error("Delete failed");
    showToast(`"${filename}" deleted`, "info");

    // Also delete from GraphRAG (best effort)
    fetch(`${GRAPH_API_BASE}/api/graph/documents/${docId}`, {
      method: "DELETE",
    }).catch(() => {});

    if (selectedDocId === docId) {
      selectedDocId = null;
      queryInput.disabled = true;
      queryBtn.disabled = true;
      clearResults();
    }
    await loadDocuments();
    await loadKgDocuments();
  } catch (err) {
    showToast(`Delete failed: ${err.message}`, "error");
  }
}

function clearResults() {
  vectorBody.innerHTML =
    '<div class="result-placeholder">Vector RAG results</div>';
  vectorSources.textContent = "—";
  vectorTime.textContent = "—";
  graphBody.innerHTML =
    '<div class="result-placeholder">Graph RAG results</div>';
  graphTriplets.textContent = "—";
  graphTime.textContent = "—";
  combinedBody.innerHTML =
    '<div class="result-placeholder">Combined Vector + Graph RAG</div>';
  combinedSources.textContent = "—";
  combinedTime.textContent = "—";
}

// ── Query ──────────────────────────────────────────────
queryBtn.addEventListener("click", submitQuery);
queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !queryBtn.disabled) submitQuery();
});

async function submitQuery() {
  const query = queryInput.value.trim();
  if (!query || !selectedDocId) return;

  queryBtn.disabled = true;
  loadingSpinner.classList.add("active");
  loadingText.textContent = "Searching documents and generating answers...";

  // Clear previous results
  vectorBody.innerHTML = "";
  vectorSources.textContent = "—";
  vectorTime.textContent = "—";
  graphBody.innerHTML = "";
  graphTriplets.textContent = "—";
  graphTime.textContent = "—";
  combinedBody.innerHTML = "";
  combinedSources.textContent = "—";
  combinedTime.textContent = "—";

  // Fire both queries in parallel
  const vectorPromise = fetch(`${API_BASE}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ document_id: selectedDocId, query, top_k: 5 }),
  }).catch((err) => {
    console.error("VectorRAG query failed:", err);
    return null;
  });

  const graphPromise = fetch(`${GRAPH_API_BASE}/api/graph/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ document_id: selectedDocId, query }),
  }).catch((err) => {
    console.warn("GraphRAG query failed:", err);
    return null;
  });

  let vectorData = null;
  let graphData = null;

  // Handle VectorRAG result
  try {
    const vectorRes = await vectorPromise;
    if (vectorRes && vectorRes.ok) {
      vectorData = await vectorRes.json();
      renderVectorResult(vectorData);
    } else if (vectorRes) {
      const err = await vectorRes.json().catch(() => ({}));
      vectorBody.innerHTML = `<div class="result-placeholder" style="color:var(--accent-danger)">❌ ${err.detail || "Query failed"}</div>`;
    }
  } catch (err) {
    vectorBody.innerHTML = `<div class="result-placeholder" style="color:var(--accent-danger)">❌ ${err.message}</div>`;
  }

  // Handle GraphRAG result
  try {
    const graphRes = await graphPromise;
    if (graphRes && graphRes.ok) {
      graphData = await graphRes.json();
      renderGraphResult(graphData);
    } else if (graphRes) {
      const err = await graphRes.json().catch(() => ({}));
      graphBody.innerHTML = `<div class="result-placeholder" style="color:var(--accent-danger)">❌ ${err.detail || "GraphRAG query failed"}</div>`;
    } else {
      graphBody.innerHTML =
        '<div class="result-placeholder">GraphRAG server not available</div>';
    }
  } catch (err) {
    graphBody.innerHTML = `<div class="result-placeholder" style="color:var(--accent-danger)">❌ ${err.message}</div>`;
  }

  let hybridData = null;

  if (vectorData || graphData) {
    try {
      const hybridRes = await fetch(`${HYBRID_API_BASE}/api/hybrid/compose`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          vector_answer: vectorData?.answer || "",
          graph_answer: graphData?.answer || "",
        }),
      });

      if (hybridRes.ok) {
        hybridData = await hybridRes.json();
      } else {
        const err = await hybridRes.json().catch(() => ({}));
        console.warn("HybridRAG failed:", err.detail || "unknown error");
      }
    } catch (err) {
      console.warn("HybridRAG request failed:", err.message);
    }
  }

  renderHybridResult(vectorData, graphData, hybridData);

  loadingSpinner.classList.remove("active");
  queryBtn.disabled = false;
}

// ── Render vector RAG result ───────────────────────────
function renderVectorResult(data) {
  let html = `<div class="result-answer">${renderBasicMarkdown(data.answer)}</div>`;

  if (data.sources && data.sources.length > 0) {
    html += `<div class="result-sources">
            <div class="result-sources__title">📎 Retrieved Sources</div>`;
    data.sources.forEach((src, i) => {
      const page = src.metadata?.page_number || "?";
      const score = (src.score * 100).toFixed(1);
      const text =
        src.text.length > 250 ? src.text.slice(0, 250) + "..." : src.text;
      html += `<div class="source-chunk">
                <div class="source-chunk__meta">Page ${page} · Relevance: ${score}%</div>
                ${escapeHtml(text)}
            </div>`;
    });
    html += `</div>`;
  }

  vectorBody.innerHTML = html;
  vectorSources.textContent = `${data.num_sources} sources`;
  vectorTime.textContent = `${data.time_taken_ms}ms`;
}

// ── Render Graph RAG result ────────────────────────────
function renderGraphResult(data) {
  let html = `<div class="result-answer">${renderBasicMarkdown(data.answer)}</div>`;

  // Show entities searched
  if (data.entities_searched && data.entities_searched.length > 0) {
    html += `<div class="graph-entities">
            <div class="graph-entities__title">🔎 Entities Searched</div>
            <div class="graph-entities__list">`;
    data.entities_searched.forEach((e) => {
      html += `<span class="entity-tag">${escapeHtml(e)}</span>`;
    });
    html += `</div></div>`;
  }

  // Show triplets used
  if (data.triplets_used && data.triplets_used.length > 0) {
    html += `<div class="result-sources">
            <div class="result-sources__title">🕸️ Graph Triplets Used</div>`;
    data.triplets_used.slice(0, 8).forEach((t) => {
      const subj = t.subject || "?";
      const pred = (t.predicate || "?").replace(/_/g, " ").toLowerCase();
      const obj = t.object || "?";
      const page = t.source_page || "?";
      html += `<div class="source-chunk source-chunk--graph">
                <div class="source-chunk__meta source-chunk__meta--graph">Page ${page}</div>
                <span class="triplet-subject">${escapeHtml(subj)}</span>
                <span class="triplet-arrow">→</span>
                <span class="triplet-predicate">${escapeHtml(pred)}</span>
                <span class="triplet-arrow">→</span>
                <span class="triplet-object">${escapeHtml(obj)}</span>
            </div>`;
    });
    if (data.triplets_used.length > 8) {
      html += `<div class="source-chunk source-chunk--graph" style="text-align:center;color:var(--text-muted)">
                + ${data.triplets_used.length - 8} more triplets
            </div>`;
    }
    html += `</div>`;
  }

  graphBody.innerHTML = html;
  graphTriplets.textContent = `${data.num_triplets} triplets`;
  graphTime.textContent = `${data.time_taken_ms}ms`;
}

// ── Render Hybrid RAG result ──────────────────────────
function renderHybridResult(vectorData, graphData, hybridData) {
  if (!vectorData && !graphData) {
    combinedBody.innerHTML =
      '<div class="result-placeholder">Combined Vector + Graph RAG</div>';
    combinedSources.textContent = "—";
    combinedTime.textContent = "—";
    return;
  }

  if (hybridData && hybridData.answer) {
    combinedBody.innerHTML = `<div class="result-answer">${renderBasicMarkdown(hybridData.answer)}</div>`;
    combinedSources.textContent = "Final answer";
    combinedTime.textContent = `${hybridData.time_taken_ms}ms`;
    return;
  }

  const vectorAnswer = vectorData?.answer || "";
  const graphAnswer = graphData?.answer || "";

  const graphIsUseful =
    graphAnswer &&
    !graphAnswer.startsWith(
      "⚠️ The requested information was not found in the knowledge graph",
    );

  const finalAnswer = graphIsUseful ? graphAnswer : vectorAnswer;
  combinedBody.innerHTML = finalAnswer
    ? `<div class="result-answer">${renderBasicMarkdown(finalAnswer)}</div>`
    : '<div class="result-placeholder">Combined Vector + Graph RAG</div>';

  combinedSources.textContent = "Final answer";
  combinedTime.textContent = "—";
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function renderBasicMarkdown(text) {
  const safe = escapeHtml(text || "");
  const bolded = safe.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  return bolded.replace(/\n/g, "<br>");
}

// ═══════════════════════════════════════════════════════
//  Knowledge Graph Explorer
// ═══════════════════════════════════════════════════════

// Entity type → color mapping (matching the dark theme)
const ENTITY_COLORS = {
  COMPANY: { background: "#4d8eff", border: "#3a6fd8", font: "#ffffff" },
  PERSON: { background: "#a855f7", border: "#8b3ed4", font: "#ffffff" },
  METRIC: { background: "#34d399", border: "#22b07a", font: "#000000" },
  VALUE: { background: "#fbbf24", border: "#d4a31f", font: "#000000" },
  DATE: { background: "#f87171", border: "#d85a5a", font: "#ffffff" },
  PRODUCT: { background: "#38bdf8", border: "#2da0d4", font: "#000000" },
  LOCATION: { background: "#fb923c", border: "#d47a33", font: "#000000" },
  EVENT: { background: "#e879f9", border: "#c55dd6", font: "#000000" },
  STRATEGY: { background: "#818cf8", border: "#6b74d4", font: "#ffffff" },
  ENTITY: { background: "#94a3b8", border: "#6b7b8d", font: "#000000" },
};

function getNodeColor(type) {
  return ENTITY_COLORS[type] || ENTITY_COLORS.ENTITY;
}

// Load KG document list
async function loadKgDocuments() {
  try {
    // Try GraphRAG documents first, fall back to VectorRAG docs
    let docs = [];

    try {
      const res = await fetch(`${GRAPH_API_BASE}/api/graph/documents`);
      if (res.ok) {
        const data = await res.json();
        docs = data.documents || [];
      }
    } catch (e) {
      // GraphRAG server not available
    }

    // If no graph docs, use vector docs as reference
    if (docs.length === 0 && documents.length > 0) {
      docs = documents.map((d) => ({
        document_id: d.document_id,
        filename: d.filename,
      }));
    }

    // Populate the select dropdown
    kgDocSelect.innerHTML = '<option value="">— Select a document —</option>';
    docs.forEach((doc) => {
      const opt = document.createElement("option");
      opt.value = doc.document_id;
      opt.textContent = doc.filename || doc.document_id;
      if (doc.num_triplets) {
        opt.textContent += ` (${doc.num_triplets} triplets)`;
      }
      kgDocSelect.appendChild(opt);
    });
  } catch (err) {
    console.warn("Could not load KG documents:", err.message);
  }
}

// Enable/disable load button based on selection
kgDocSelect.addEventListener("change", () => {
  kgLoadBtn.disabled = !kgDocSelect.value;
});

// Load and render knowledge graph
kgLoadBtn.addEventListener("click", async () => {
  const docId = kgDocSelect.value;
  if (!docId) return;

  kgPlaceholder.style.display = "none";
  kgLoading.style.display = "flex";

  try {
    const res = await fetch(`${GRAPH_API_BASE}/api/graph/visualize/${docId}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Failed to load graph");
    }

    const data = await res.json();
    renderKnowledgeGraph(data);
    showToast(
      `Knowledge graph loaded: ${data.stats.node_count} nodes, ${data.stats.edge_count} edges`,
      "success",
    );
  } catch (err) {
    kgPlaceholder.style.display = "flex";
    showToast(`Failed to load graph: ${err.message}`, "error");
  } finally {
    kgLoading.style.display = "none";
  }
});

function renderKnowledgeGraph(data) {
  // Destroy previous network
  if (kgNetwork) {
    kgNetwork.destroy();
    kgNetwork = null;
  }

  if (!data.nodes || data.nodes.length === 0) {
    kgPlaceholder.innerHTML = `
            <div class="kg-placeholder__icon">📭</div>
            <p>No knowledge graph data found for this document.<br>
            Upload the document first to extract triplets.</p>`;
    kgPlaceholder.style.display = "flex";
    return;
  }

  kgPlaceholder.style.display = "none";

  // Prepare vis.js data
  const nodes = new vis.DataSet(
    data.nodes.map((n) => {
      const colors = getNodeColor(n.type || n.group);
      return {
        id: n.id,
        label: n.label,
        group: n.type || "ENTITY",
        title: `<strong>${n.label}</strong><br>Type: ${n.type || "ENTITY"}`,
        color: {
          background: colors.background,
          border: colors.border,
          highlight: { background: colors.background, border: "#ffffff" },
          hover: { background: colors.background, border: "#ffffff" },
        },
        font: {
          color: colors.font,
          size: 13,
          face: "Inter, sans-serif",
          bold: true,
        },
        shape: "dot",
        size: 18,
        borderWidth: 2,
        shadow: { enabled: true, color: colors.background + "40", size: 10 },
      };
    }),
  );

  const edges = new vis.DataSet(
    data.edges.map((e, i) => ({
      id: i,
      from: e.from,
      to: e.to,
      label: e.label,
      title: `${e.label}${e.source_page ? " (Page " + e.source_page + ")" : ""}`,
      font: {
        size: 10,
        color: "#8888a8",
        strokeWidth: 0,
        face: "Inter, sans-serif",
      },
      color: { color: "#555570", highlight: "#a855f7", hover: "#a855f7" },
      arrows: { to: { enabled: true, scaleFactor: 0.6 } },
      width: 1.5,
      smooth: { type: "continuous", roundness: 0.3 },
    })),
  );

  // vis.js options
  const options = {
    physics: {
      enabled: true,
      barnesHut: {
        gravitationalConstant: -3000,
        centralGravity: 0.3,
        springLength: 150,
        springConstant: 0.04,
        damping: 0.09,
      },
      stabilization: { iterations: 200, fit: true },
    },
    interaction: {
      hover: true,
      tooltipDelay: 200,
      zoomView: true,
      dragView: true,
      navigationButtons: false,
      keyboard: { enabled: true },
    },
    layout: {
      improvedLayout: true,
    },
    nodes: {
      borderWidthSelected: 3,
    },
    edges: {
      selectionWidth: 2,
    },
  };

  // Render
  kgNetwork = new vis.Network(kgCanvas, { nodes, edges }, options);

  // Update stats
  kgNodeCount.querySelector(".kg-stat__value").textContent =
    data.stats.node_count;
  kgEdgeCount.querySelector(".kg-stat__value").textContent =
    data.stats.edge_count;

  // Build legend from unique types in the data
  const types = [
    ...new Set(data.nodes.map((n) => n.type || n.group || "ENTITY")),
  ];
  renderLegend(types);

  kgPhysicsOn = true;
}

function renderLegend(types) {
  kgLegend.innerHTML = "";
  types.forEach((type) => {
    const colors = getNodeColor(type);
    const item = document.createElement("div");
    item.className = "kg-legend__item";
    item.innerHTML = `
            <span class="kg-legend__dot" style="background:${colors.background};border-color:${colors.border}"></span>
            <span class="kg-legend__label">${type}</span>
        `;
    kgLegend.appendChild(item);
  });
}

// Physics toggle
kgPhysicsBtn.addEventListener("click", () => {
  if (!kgNetwork) return;
  kgPhysicsOn = !kgPhysicsOn;
  kgNetwork.setOptions({ physics: { enabled: kgPhysicsOn } });
  kgPhysicsBtn.classList.toggle("active", kgPhysicsOn);
  showToast(`Physics ${kgPhysicsOn ? "enabled" : "disabled"}`, "info");
});

// Fit to view
kgFitBtn.addEventListener("click", () => {
  if (!kgNetwork) return;
  kgNetwork.fit({
    animation: { duration: 500, easingFunction: "easeInOutQuad" },
  });
});

// ── Init ───────────────────────────────────────────────
loadDocuments();
loadKgDocuments();
