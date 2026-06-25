import React, { useEffect, useMemo, useRef, useState } from "react";

const API_BASE =
  import.meta.env.VITE_API_BASE?.replace(/\/+$/, "") || "http://localhost:8000";

const API_KEY = import.meta.env.VITE_API_KEY || "";

function authHeaders(extra = {}) {
  return API_KEY ? { ...extra, "X-API-KEY": API_KEY } : extra;
}

function confidenceFromScore(score) {
  if (score >= 0.03) return { label: "High", tone: "text-green-700 dark:text-green-400" };
  if (score >= 0.02) return { label: "Medium", tone: "text-yellow-700 dark:text-yellow-400" };
  return { label: "Low", tone: "text-red-700 dark:text-red-400" };
}

export default function App() {
  const [backendStatus, setBackendStatus] = useState("Connecting...");
  const [docs, setDocs] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState("");

  const [uploading, setUploading] = useState(false);
  const [summarizing, setSummarizing] = useState(false);
  const [sending, setSending] = useState(false);

  const [retrievalMode, setRetrievalMode] = useState("hybrid");
  const [fileToUpload, setFileToUpload] = useState(null);
  const [input, setInput] = useState("");

  const [messages, setMessages] = useState([]);

  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem("secrag_dark");
    return saved === "1";
  });

  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState("");

  const chatEndRef = useRef(null);
  const hasSelectedDoc = useMemo(() => !!selectedDoc, [selectedDoc]);

  useEffect(() => {
    localStorage.setItem("secrag_dark", darkMode ? "1" : "0");
    document.documentElement.classList.toggle("dark", darkMode);
  }, [darkMode]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    checkHealth();
    refreshDocs(true);
  }, []);

  function pushSystem(text) {
    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), role: "system", text },
    ]);
  }

  async function checkHealth() {
  console.log("Checking health at:", `${API_BASE}/health`);
  try {
    const res = await fetch(`${API_BASE}/health`, { headers: authHeaders() });
    if (!res.ok) throw new Error();
    setBackendStatus("Connected");
  } catch (e) {
    console.error("Health check failed:", e);
    setBackendStatus("Disconnected — start backend");
  }
}

  async function refreshDocs(selectFirst = false) {
    try {
      const res = await fetch(`${API_BASE}/list_docs`, {
        headers: authHeaders(),
      });
      const data = await res.json().catch(() => ({}));
      const list = Array.isArray(data?.documents) ? data.documents : [];
      setDocs(list);

      if (selectFirst && !selectedDoc && list.length > 0) {
        setSelectedDoc(list[0]);
      }
    } catch {
      // ignore
    }
  }

  async function handleUpload() {
    if (!fileToUpload) return;
    setUploading(true);

    try {
      const form = new FormData();
      form.append("file", fileToUpload);

      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        headers: authHeaders(),
        body: form,
      });

      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.detail || "Upload failed");

      await refreshDocs(false);
      setSelectedDoc(data?.filename || "");

      pushSystem(
        `Uploaded ${data.filename} • ${data.chunks_inserted_to_chroma ?? data.total_chunks} chunks • ${data.embedding_dim} dim`
      );
    } catch (e) {
      pushSystem(`Upload error: ${e.message}`);
    } finally {
      setUploading(false);
      setFileToUpload(null);
    }
  }

  async function handleSend() {
    const q = input.trim();
    if (!q) return;

    if (!selectedDoc) {
      pushSystem("Select a document first.");
      return;
    }

    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), role: "user", text: q },
    ]);

    setInput("");
    setSending(true);

    try {
      const res = await fetch(`${API_BASE}/answer`, {
        method: "POST",
        headers: authHeaders({
          "Content-Type": "application/json",
        }),
        body: JSON.stringify({
          filename: selectedDoc,
          query: q,
          top_k: 5,
          mode: retrievalMode,
          alpha: 0.7,
        }),
      });

      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.detail || "Answer failed");

      const topScore =
        Array.isArray(data?.citations) && data.citations.length > 0
          ? Number(data.citations[0].rerank_score ?? data.citations[0].score ?? 0)
          : 0;

      const conf = confidenceFromScore(topScore);

      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: data.answer || "",
          confidence: conf,
          citations: data.citations || [],
        },
      ]);
    } catch (e) {
      pushSystem(`Answer error: ${e.message}`);
    } finally {
      setSending(false);
    }
  }

  async function handleSummarize() {
    if (!selectedDoc) {
      pushSystem("Select a document first.");
      return;
    }

    setSummarizing(true);

    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), role: "user", text: "Summarize this document." },
    ]);

    try {
      const res = await fetch(`${API_BASE}/summarize`, {
        method: "POST",
        headers: authHeaders({
          "Content-Type": "application/json",
        }),
        body: JSON.stringify({
          filename: selectedDoc,
          intro_chunks: 3,
          top_k: 5,
          max_output_tokens: 350,
          mode: retrievalMode,
          alpha: 0.7,
        }),
      });

      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.detail || "Summarize failed");

      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: data.summary || "",
          citations: data.citations || [],
        },
      ]);
    } catch (e) {
      pushSystem(`Summarize error: ${e.message}`);
    } finally {
      setSummarizing(false);
    }
  }

  async function handleDeleteSelectedConfirmed() {
    if (!selectedDoc) return;

    setDeleting(true);
    setDeleteError("");

    try {
      const res = await fetch(
        `${API_BASE}/documents/${encodeURIComponent(selectedDoc)}`,
        {
          method: "DELETE",
          headers: authHeaders(),
        }
      );

      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.detail || "Delete failed");

      pushSystem(
        `Deleted "${selectedDoc}" • removed ${data?.deleted?.length || 0} file(s)`
      );

      await refreshDocs(false);
      setSelectedDoc("");
      setShowDeleteModal(false);
    } catch (e) {
      setDeleteError(e.message);
    } finally {
      setDeleting(false);
    }
  }

  function clearChat() {
    setMessages([]);
  }

  return (
    <div className="h-screen flex bg-gray-50 text-gray-900 dark:bg-zinc-950 dark:text-zinc-100">
      <aside className="w-[320px] border-r bg-white p-4 space-y-4 dark:bg-zinc-900 dark:border-zinc-800">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h1 className="text-xl font-semibold">SecRAG</h1>
            <p className="text-xs text-gray-500 dark:text-zinc-400">
              PDF Q&A with verified citations
            </p>
            <p className="text-xs text-gray-600 dark:text-zinc-300 mt-2">
              Backend: <span className="font-medium">{backendStatus}</span>
            </p>
          </div>

          <button
            className="text-xs px-3 py-2 rounded border bg-gray-50 hover:bg-gray-100 dark:bg-zinc-800 dark:border-zinc-700 dark:hover:bg-zinc-700"
            onClick={() => setDarkMode((v) => !v)}
            title="Toggle dark mode"
          >
            {darkMode ? "Light" : "Dark"}
          </button>
        </div>

        <div>
          <label className="text-sm font-medium">Retrieval Mode</label>
          <select
            value={retrievalMode}
            onChange={(e) => setRetrievalMode(e.target.value)}
            className="w-full mt-1 border rounded p-2 text-sm bg-white dark:bg-zinc-900 dark:border-zinc-700"
          >
            <option value="hybrid">Hybrid</option>
            <option value="semantic">Semantic</option>
            <option value="bm25">BM25</option>
          </select>
        </div>

        <div className="border rounded p-3 bg-gray-50 dark:bg-zinc-800 dark:border-zinc-700">
          <div className="text-sm font-medium mb-2">Upload PDF</div>
          <input
            type="file"
            accept="application/pdf"
            onChange={(e) => setFileToUpload(e.target.files?.[0] || null)}
            className="text-sm"
          />
          <button
            onClick={handleUpload}
            disabled={!fileToUpload || uploading}
            className="mt-2 w-full bg-black text-white py-2 rounded disabled:opacity-50 dark:bg-white dark:text-black"
          >
            {uploading ? "Uploading..." : "Upload"}
          </button>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-medium">Documents</h2>

            <span className="text-xs px-2 py-1 rounded-full border bg-gray-50 dark:bg-zinc-800 dark:border-zinc-700">
              {docs.length}
            </span>
          </div>

          <div className="max-h-[240px] overflow-auto border rounded bg-white dark:bg-zinc-900 dark:border-zinc-700">
            {docs.length === 0 ? (
              <div className="p-3 text-sm text-gray-500 dark:text-zinc-400">
                No documents yet. Upload a PDF.
              </div>
            ) : (
              docs.map((d) => (
                <button
                  key={d}
                  onClick={() => setSelectedDoc(d)}
                  className={`block w-full text-left px-3 py-2 text-sm border-b last:border-b-0
                    dark:border-zinc-800 hover:bg-gray-50 dark:hover:bg-zinc-800
                    ${selectedDoc === d ? "bg-gray-100 dark:bg-zinc-800 font-medium" : ""}`}
                >
                  {d}
                </button>
              ))
            )}
          </div>

          <div className="text-xs text-gray-500 dark:text-zinc-400">
            Selected: <span className="font-medium">{selectedDoc || "None"}</span>
          </div>
        </div>

        <button
          onClick={handleSummarize}
          disabled={!hasSelectedDoc || summarizing}
          className="w-full border rounded py-2 text-sm bg-white disabled:opacity-50 hover:bg-gray-50
                     dark:bg-zinc-900 dark:border-zinc-700 dark:hover:bg-zinc-800"
        >
          {summarizing ? "Summarizing..." : "Summarize Document"}
        </button>

        <button
          onClick={() => {
            if (!selectedDoc) return;
            setDeleteError("");
            setShowDeleteModal(true);
          }}
          disabled={!selectedDoc}
          className="w-full border rounded py-2 text-sm bg-white disabled:opacity-50 hover:bg-gray-50
                     dark:bg-zinc-900 dark:border-zinc-700 dark:hover:bg-zinc-800"
        >
          Delete Selected Document
        </button>

        <div className="flex gap-2">
          <button
            onClick={clearChat}
            className="flex-1 border rounded py-2 text-sm bg-white hover:bg-gray-50
                       dark:bg-zinc-900 dark:border-zinc-700 dark:hover:bg-zinc-800"
          >
            Clear Chat
          </button>

          <button
            onClick={() => refreshDocs(false)}
            className="flex-1 border rounded py-2 text-sm bg-white hover:bg-gray-50
                       dark:bg-zinc-900 dark:border-zinc-700 dark:hover:bg-zinc-800"
          >
            Refresh
          </button>
        </div>
      </aside>

      <main className="flex-1 flex flex-col">
        <header className="h-14 border-b bg-white px-6 flex items-center justify-between dark:bg-zinc-900 dark:border-zinc-800">
          <div className="font-medium">Chat</div>
          <div className="text-sm text-gray-500 dark:text-zinc-400">
            {sending ? "Thinking..." : "Ready"}
          </div>
        </header>

        <section className="flex-1 overflow-auto p-6 space-y-4">
          {messages.length === 0 ? (
            <div className="text-sm text-gray-500 dark:text-zinc-400">
              Upload/select a document, then ask a question or click “Summarize Document”.
            </div>
          ) : (
            messages.map((m) => (
              <div
                key={m.id}
                className={`max-w-3xl ${m.role === "user" ? "ml-auto" : "mr-auto"}`}
              >
                <div className="text-xs text-gray-500 dark:text-zinc-400 mb-1">
                  {m.role === "user" ? "You" : m.role === "assistant" ? "SecRAG" : "System"}
                  {m.confidence && (
                    <span className={`ml-2 ${m.confidence.tone}`}>
                      Confidence: {m.confidence.label}
                    </span>
                  )}
                </div>

                <div className="bg-white p-3 rounded border whitespace-pre-wrap dark:bg-zinc-900 dark:border-zinc-800">
                  {m.text}
                </div>
              </div>
            ))
          )}
          <div ref={chatEndRef} />
        </section>

        <footer className="border-t p-4 flex gap-2 bg-white dark:bg-zinc-900 dark:border-zinc-800">
          <textarea
            className="flex-1 border rounded p-2 resize-none bg-white dark:bg-zinc-950 dark:border-zinc-700"
            placeholder="Ask a question..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                if (!sending) handleSend();
              }
            }}
          />
          <button
            onClick={handleSend}
            disabled={!selectedDoc || sending}
            className="bg-black text-white px-5 rounded disabled:opacity-50 dark:bg-white dark:text-black"
          >
            Send
          </button>
        </footer>
      </main>
      {showDeleteModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => !deleting && setShowDeleteModal(false)}
          />

          <div className="relative w-[92%] max-w-md rounded-xl border bg-white p-5 shadow-lg dark:bg-zinc-900 dark:border-zinc-700">
            <div className="text-lg font-semibold">Delete document?</div>

            <div className="mt-2 text-sm text-gray-600 dark:text-zinc-300">
              You are about to delete:
              <div className="mt-2 rounded-lg border bg-gray-50 px-3 py-2 font-medium dark:bg-zinc-800 dark:border-zinc-700">
                {selectedDoc}
              </div>
              <div className="mt-2 text-xs text-gray-500 dark:text-zinc-400">
                This removes the PDF and generated artifacts (chunks/embeddings/text). This cannot be undone.
              </div>
            </div>

            {deleteError && (
              <div className="mt-3 rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-700 dark:border-red-700 dark:bg-red-950 dark:text-red-300">
                {deleteError}
              </div>
            )}

            <div className="mt-5 flex gap-2 justify-end">
              <button
                className="rounded-lg border px-4 py-2 text-sm bg-white hover:bg-gray-50 disabled:opacity-50
                           dark:bg-zinc-900 dark:border-zinc-700 dark:hover:bg-zinc-800"
                onClick={() => setShowDeleteModal(false)}
                disabled={deleting}
              >
                Cancel
              </button>

              <button
                className="rounded-lg bg-red-600 px-4 py-2 text-sm text-white hover:bg-red-700 disabled:opacity-50"
                onClick={handleDeleteSelectedConfirmed}
                disabled={deleting}
              >
                {deleting ? "Deleting..." : "Delete"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}