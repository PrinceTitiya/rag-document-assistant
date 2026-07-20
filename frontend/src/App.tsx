import { useEffect, useRef, useState } from 'react'
import './App.css'

const API_BASE = 'http://localhost:8000'

interface Source {
  source: string
  page: number
}

interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
}

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeSource, setActiveSource] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    fetch(`${API_BASE}/status`)
      .then((res) => res.json())
      .then((data: { active_source: string | null }) =>
        setActiveSource(data.active_source),
      )
      .catch(() => {})
  }, [])

  const scrollToBottom = () => {
    requestAnimationFrame(() => {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    })
  }

  const sendQuery = async () => {
    const question = input.trim()
    if (!question || loading || uploading) return

    setMessages((prev) => [...prev, { role: 'user', content: question }])
    setInput('')
    setError(null)
    setLoading(true)
    scrollToBottom()

    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })

      if (!res.ok) {
        const body = await res.json().catch(() => null)
        throw new Error(body?.detail || `Request failed (${res.status})`)
      }

      const data: { answer: string; sources: Source[] } = await res.json()

      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: data.answer, sources: data.sources },
      ])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong')
    } finally {
      setLoading(false)
      scrollToBottom()
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendQuery()
    }
  }

  const handleUploadClick = () => {
    if (uploading || loading) return
    fileInputRef.current?.click()
  }

  const handleFileSelected = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (!file) return

    setError(null)
    setUploading(true)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const body = await res.json().catch(() => null)
        throw new Error(body?.detail || `Upload failed (${res.status})`)
      }

      const data: { filename: string; chunks_indexed: number } = await res.json()
      setActiveSource(data.filename)
      setMessages([])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  const handleResetSource = async () => {
    if (uploading || loading) return

    try {
      await fetch(`${API_BASE}/reset`, { method: 'POST' })
    } catch {
      // ignored — worst case the badge just stays until next successful call
    }

    setActiveSource(null)
    setMessages([])
    setError(null)
  }

  return (
    <div className="chat-page">
      <header className="chat-header">
        <h1 className="title">RAG Document Assistant</h1>
        <p className="subtitle">Ask questions about your documents</p>
      </header>

      <div className="source-bar">
        <span className="source-badge">
          {activeSource ? `📄 ${activeSource}` : '📁 Using default documents'}
        </span>
        {activeSource && (
          <button className="source-clear" onClick={handleResetSource}>
            Reset to default
          </button>
        )}
      </div>

      <div className="chat-window">
        {messages.length === 0 && !loading && (
          <div className="empty-state">Ask a question to get started.</div>
        )}

        {messages.map((message, i) => (
          <div key={i} className={`message ${message.role}`}>
            <div className="bubble">
              <p>{message.content}</p>
              {message.sources && message.sources.length > 0 && (
                <div className="sources">
                  {message.sources.map((s, j) => (
                    <span key={j} className="source-tag">
                      {s.source.split('/').pop()} · p.{s.page}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="message assistant">
            <div className="bubble bubble-loading">Thinking…</div>
          </div>
        )}

        {uploading && (
          <div className="message assistant">
            <div className="bubble bubble-loading">
              Indexing your PDF (chunking, embedding, building index)…
            </div>
          </div>
        )}

        {error && <div className="error-banner">{error}</div>}

        <div ref={bottomRef} />
      </div>

      <div className="input-bar">
        <input
          ref={fileInputRef}
          type="file"
          accept="application/pdf"
          className="file-input"
          onChange={handleFileSelected}
        />
        <button
          className="upload-button"
          onClick={handleUploadClick}
          disabled={uploading || loading}
          title="Upload a PDF to chat with"
        >
          📎
        </button>
        <textarea
          className="input-box"
          placeholder="Type your question..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
        />
        <button
          className="send-button"
          onClick={sendQuery}
          disabled={loading || uploading || !input.trim()}
        >
          Send
        </button>
      </div>
    </div>
  )
}

export default App
