import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const EXAMPLE_QUESTION = "What does this person think about remote work?";

// In production (e.g. Vercel) set VITE_API_BASE to the backend URL, e.g.
// https://api.your-domain.com. Left empty for local dev so requests stay
// relative and Vite's dev proxy forwards /chat and /health to localhost:8000.
const API_BASE = (import.meta.env.VITE_API_BASE || "").replace(/\/+$/, "");

function App() {
  const [question, setQuestion] = useState(EXAMPLE_QUESTION);
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [health, setHealth] = useState("checking");
  const sourceRefs = useRef({});

  useEffect(() => {
    let ignore = false;
    fetch(`${API_BASE}/health`)
      .then((response) => {
        if (!response.ok) {
          throw new Error("unavailable");
        }
        return response.json();
      })
      .then(() => {
        if (!ignore) setHealth("ok");
      })
      .catch(() => {
        if (!ignore) setHealth("down");
      });
    return () => {
      ignore = true;
    };
  }, []);

  const citedSources = useMemo(
    () => new Set(sources.filter((source) => source.cited).map((source) => source.index)),
    [sources],
  );

  async function ask(event) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || status === "loading") {
      return;
    }

    setStatus("loading");
    setError("");
    setAnswer("");
    setSources([]);

    try {
      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || "Request failed");
      }
      if (!response.body) {
        throw new Error("Streaming is unavailable in this browser");
      }

      let streamError = "";
      await readSse(response, {
        token(data) {
          setAnswer((current) => current + (data.text || ""));
        },
        sources(data) {
          setAnswer(data.answer || "");
          setSources(data.sources || []);
        },
        error(data) {
          streamError = data.detail || "Streaming failed";
        },
      });
      if (streamError) {
        throw new Error(streamError);
      }
      setStatus("done");
    } catch (err) {
      setError(err.message || "Request failed");
      setStatus("error");
    }
  }

  function jumpToSource(index) {
    const node = sourceRefs.current[index];
    if (node) {
      node.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }

  return (
    <main className="app-shell">
      <section className="workspace">
        <header className="topbar">
          <div>
            <h1>Cortex</h1>
            <p>Grounded answers from indexed social exports.</p>
          </div>
          <span className={`health health-${health}`}>
            {health === "ok" ? "API online" : health === "down" ? "API offline" : "Checking API"}
          </span>
        </header>

        <form className="ask-form" onSubmit={ask}>
          <label htmlFor="question">Question</label>
          <div className="ask-row">
            <textarea
              id="question"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask about the indexed posts"
              rows={3}
            />
            <button type="submit" disabled={status === "loading" || !question.trim()}>
              {status === "loading" ? "Asking..." : "Ask"}
            </button>
          </div>
        </form>

        {error && <div className="notice notice-error">{error}</div>}

        <section className="result-grid" aria-live="polite">
          <article className="answer-panel">
            <div className="section-head">
              <h2>Answer</h2>
              {status === "done" && (
                <span className={citedSources.size ? "pill pill-grounded" : "pill pill-ungrounded"}>
                  {citedSources.size ? "Grounded" : "Uncited"}
                </span>
              )}
            </div>
            {status === "loading" && !answer ? (
              <p className="muted">Retrieving sources and drafting an answer.</p>
            ) : answer ? (
              <p className="answer-text">
                {renderAnswer(answer, sources, jumpToSource)}
              </p>
            ) : (
              <p className="muted">Ask a question to retrieve context and generate a cited answer.</p>
            )}
          </article>

          <aside className="sources-panel">
            <div className="section-head">
              <h2>Sources</h2>
              <span className="source-count">{sources.length}</span>
            </div>
            {sources.length ? (
              <ol className="source-list">
                {sources.map((source) => (
                  <li
                    key={`${source.platform}-${source.external_id}-${source.index}`}
                    ref={(node) => {
                      sourceRefs.current[source.index] = node;
                    }}
                    className={source.cited ? "source-item source-cited" : "source-item"}
                  >
                    <div className="source-meta">
                      <span>[{source.index}] {source.platform} · {source.date || "undated"}</span>
                      <span>{source.content_type}</span>
                    </div>
                    <p>{source.snippet}</p>
                    <div className="source-actions">
                      {source.url ? (
                        <a href={source.url} target="_blank" rel="noreferrer">
                          Open source
                        </a>
                      ) : (
                        <span>No source URL</span>
                      )}
                      {source.cited && <strong>Cited</strong>}
                    </div>
                  </li>
                ))}
              </ol>
            ) : (
              <p className="muted">Retrieved sources will appear here.</p>
            )}
          </aside>
        </section>
      </section>
    </main>
  );
}

async function readSse(response, handlers) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const records = buffer.split("\n\n");
    buffer = records.pop() || "";
    for (const record of records) {
      handleSseRecord(record, handlers);
    }
    if (done) {
      break;
    }
  }

  if (buffer.trim()) {
    handleSseRecord(buffer, handlers);
  }
}

function handleSseRecord(record, handlers) {
  let event = "message";
  const data = [];
  for (const line of record.split("\n")) {
    if (line.startsWith("event: ")) {
      event = line.slice(7).trim();
    } else if (line.startsWith("data: ")) {
      data.push(line.slice(6));
    }
  }

  const payload = data.length ? JSON.parse(data.join("\n")) : {};
  if (handlers[event]) {
    handlers[event](payload);
  }
}

function renderAnswer(answer, sources, onCitationClick) {
  const sourceIndexes = new Set(sources.map((source) => source.index));
  const parts = answer.split(/(\[\d+\])/g);
  return parts.map((part, index) => {
    const match = part.match(/^\[(\d+)\]$/);
    if (!match) {
      return <React.Fragment key={`${part}-${index}`}>{part}</React.Fragment>;
    }

    const sourceIndex = Number(match[1]);
    if (!sourceIndexes.has(sourceIndex)) {
      return <React.Fragment key={`${part}-${index}`}>{part}</React.Fragment>;
    }

    return (
      <button
        key={`${part}-${index}`}
        type="button"
        className="cite-link"
        onClick={() => onCitationClick(sourceIndex)}
        aria-label={`Jump to source ${sourceIndex}`}
      >
        {part}
      </button>
    );
  });
}

createRoot(document.getElementById("root")).render(<App />);
