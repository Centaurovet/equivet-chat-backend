import { useState, useRef, useEffect } from "react";

// ── Configure a URL do seu backend aqui ──────────────────────────────────────
// Após o deploy no Railway/Render, substitua pela URL gerada.
// Exemplo: "https://equivet-chat.railway.app"
const BACKEND_URL = "http://localhost:8000";

// ── Perfis ────────────────────────────────────────────────────────────────────
const PROFILES = [
  { id: "vet",     label: "Médico-Veterinário", icon: "⚕️", color: "#C9A84C", description: "Diagnóstico, farmacologia, protocolos clínicos" },
  { id: "owner",   label: "Proprietário",        icon: "🏡", color: "#7EB8A4", description: "Manejo, saúde, nutrição do seu cavalo" },
  { id: "trainer", label: "Treinador",           icon: "🏇", color: "#A07CB5", description: "Performance, condicionamento, bem-estar atlético" },
  { id: "farrier", label: "Ferrador",            icon: "🔨", color: "#D4845A", description: "Casco, ferrageamento, claudicação" },
];

const WELCOME_MESSAGES = {
  vet:     "Olá, colega. Como posso ajudar no seu atendimento hoje?",
  owner:   "Olá! Sou o EquiVet IA. Pode me contar o que está acontecendo com seu cavalo?",
  trainer: "Olá! Aqui é o EquiVet IA. Qual desafio de condicionamento ou saúde posso ajudar a resolver hoje?",
  farrier: "Olá! Sou o EquiVet IA. Qual questão de casco ou ferrageamento está na sua bancada hoje?",
};

// ── Renderizador simples de Markdown ─────────────────────────────────────────
function MarkdownText({ text }) {
  const linhas = text.split("\n");
  const elementos = [];
  let i = 0;

  while (i < linhas.length) {
    const linha = linhas[i];

    // Linha em branco
    if (linha.trim() === "") { elementos.push(<br key={i} />); i++; continue; }

    // Título ##
    if (linha.startsWith("## ")) {
      elementos.push(<strong key={i} style={{ display: "block", marginTop: 10, marginBottom: 4, color: "#E8E0D0" }}>{linha.slice(3)}</strong>);
      i++; continue;
    }

    // Bullet - ou *
    if (/^[-*]\s/.test(linha)) {
      const itens = [];
      while (i < linhas.length && /^[-*]\s/.test(linhas[i])) {
        itens.push(<li key={i} style={{ marginBottom: 3 }}>{renderInline(linhas[i].slice(2))}</li>);
        i++;
      }
      elementos.push(<ul key={`ul-${i}`} style={{ margin: "6px 0", paddingLeft: 20 }}>{itens}</ul>);
      continue;
    }

    // Parágrafo normal
    elementos.push(<p key={i} style={{ margin: "4px 0", lineHeight: 1.65 }}>{renderInline(linha)}</p>);
    i++;
  }

  return <div>{elementos}</div>;
}

function renderInline(texto) {
  // **negrito** e *itálico*
  const partes = texto.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g);
  return partes.map((parte, i) => {
    if (parte.startsWith("**") && parte.endsWith("**"))
      return <strong key={i}>{parte.slice(2, -2)}</strong>;
    if (parte.startsWith("*") && parte.endsWith("*"))
      return <em key={i}>{parte.slice(1, -1)}</em>;
    return parte;
  });
}

// ── Componente principal ──────────────────────────────────────────────────────
export default function EquiVetChat() {
  const [selectedProfile, setSelectedProfile] = useState(null);
  const [messages, setMessages]               = useState([]);
  const [input, setInput]                     = useState("");
  const [loading, setLoading]                 = useState(false);
  const [error, setError]                     = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef       = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const selectProfile = (profile) => {
    setSelectedProfile(profile);
    setMessages([{ role: "assistant", content: WELCOME_MESSAGES[profile.id] }]);
    setError(null);
    setTimeout(() => inputRef.current?.focus(), 100);
  };

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage  = { role: "user", content: input.trim() };
    const newMessages  = [...messages, userMessage];
    setMessages(newMessages);
    setInput("");
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          perfil:    selectedProfile.id,
          mensagens: newMessages,
        }),
      });

      if (res.status === 429) {
        setError("Muitas mensagens em pouco tempo. Aguarde um momento.");
        setLoading(false);
        return;
      }
      if (!res.ok) throw new Error(`Erro ${res.status}`);

      const data = await res.json();
      setMessages([...newMessages, { role: "assistant", content: data.resposta }]);
    } catch {
      setError("Falha na conexão com o EquiVet IA. Tente novamente.");
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  };

  const reset = () => {
    setSelectedProfile(null);
    setMessages([]);
    setInput("");
    setError(null);
  };

  return (
    <div style={styles.root}>
      <div style={styles.bgTexture} />

      {/* Header */}
      <header style={styles.header}>
        <div style={styles.headerInner}>
          <div style={styles.logo}>
            <span style={styles.logoIcon}>⚕</span>
            <div>
              <div style={styles.logoTitle}>EquiVet IA</div>
              <div style={styles.logoSub}>Centaurovet · Scientia · Cura · Equus</div>
            </div>
          </div>
          {selectedProfile && (
            <button onClick={reset} style={styles.resetBtn}>← Trocar perfil</button>
          )}
        </div>
      </header>

      <main style={styles.main}>
        {!selectedProfile ? (
          /* Seleção de perfil */
          <div style={styles.profileScreen}>
            <div style={styles.profileHeading}>
              <h2 style={styles.profileTitle}>Quem está consultando hoje?</h2>
              <p style={styles.profileSubtitle}>Seu perfil calibra o nível técnico e o foco da conversa.</p>
            </div>
            <div style={styles.profileGrid}>
              {PROFILES.map((p) => (
                <button key={p.id} onClick={() => selectProfile(p)}
                  style={{ ...styles.profileCard, "--accent": p.color }}>
                  <div style={{ ...styles.profileCardAccent, background: p.color }} />
                  <span style={styles.profileCardIcon}>{p.icon}</span>
                  <div style={styles.profileCardLabel}>{p.label}</div>
                  <div style={styles.profileCardDesc}>{p.description}</div>
                </button>
              ))}
            </div>
            <div style={styles.disclaimer}>
              O EquiVet IA é uma ferramenta de apoio. Não substitui avaliação clínica presencial.
            </div>
          </div>
        ) : (
          /* Chat */
          <div style={styles.chatWrapper}>
            <div style={{ ...styles.profileBadge, borderColor: selectedProfile.color }}>
              <span>{selectedProfile.icon}</span>
              <span style={{ color: selectedProfile.color }}>{selectedProfile.label}</span>
            </div>

            <div style={styles.messageList}>
              {messages.map((msg, i) => (
                <div key={i} style={{ ...styles.messageBubble, ...(msg.role === "user" ? styles.userBubble : styles.aiBubble) }}>
                  {msg.role === "assistant" && (
                    <div style={{ ...styles.aiLabel, color: selectedProfile.color }}>EquiVet IA</div>
                  )}
                  {msg.role === "assistant"
                    ? <MarkdownText text={msg.content} />
                    : <div style={styles.messageText}>{msg.content}</div>
                  }
                </div>
              ))}

              {loading && (
                <div style={{ ...styles.messageBubble, ...styles.aiBubble }}>
                  <div style={{ ...styles.aiLabel, color: selectedProfile.color }}>EquiVet IA</div>
                  <div style={styles.typingDots}>
                    <span style={{ ...styles.dot, animationDelay: "0s" }} />
                    <span style={{ ...styles.dot, animationDelay: "0.2s" }} />
                    <span style={{ ...styles.dot, animationDelay: "0.4s" }} />
                  </div>
                </div>
              )}

              {error && <div style={styles.errorMsg}>{error}</div>}
              <div ref={messagesEndRef} />
            </div>

            <div style={styles.inputArea}>
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Digite sua dúvida... (Enter para enviar)"
                style={styles.textarea}
                rows={2}
              />
              <button onClick={sendMessage} disabled={loading || !input.trim()}
                style={{ ...styles.sendBtn, background: selectedProfile.color, opacity: loading || !input.trim() ? 0.5 : 1 }}>
                {loading ? "..." : "→"}
              </button>
            </div>
            <div style={styles.inputHint}>Enter envia · Shift+Enter quebra linha</div>
          </div>
        )}
      </main>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');
        @keyframes blink {
          0%, 80%, 100% { opacity: 0.2; transform: scale(0.8); }
          40% { opacity: 1; transform: scale(1); }
        }
        textarea:focus { outline: none; }
        button:focus { outline: none; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #3a3a3a; border-radius: 2px; }
      `}</style>
    </div>
  );
}

// ── Estilos ───────────────────────────────────────────────────────────────────
const styles = {
  root: { minHeight: "100vh", background: "#0E0E0E", color: "#E8E0D0", fontFamily: "'DM Sans', sans-serif", display: "flex", flexDirection: "column", position: "relative", overflow: "hidden" },
  bgTexture: { position: "fixed", inset: 0, backgroundImage: `radial-gradient(ellipse at 20% 20%, rgba(201,168,76,0.06) 0%, transparent 60%), radial-gradient(ellipse at 80% 80%, rgba(126,184,164,0.04) 0%, transparent 60%)`, pointerEvents: "none", zIndex: 0 },
  header: { borderBottom: "1px solid #2A2A2A", padding: "16px 24px", position: "relative", zIndex: 10, backdropFilter: "blur(8px)", background: "rgba(14,14,14,0.9)" },
  headerInner: { maxWidth: 720, margin: "0 auto", display: "flex", alignItems: "center", justifyContent: "space-between" },
  logo: { display: "flex", alignItems: "center", gap: 12 },
  logoIcon: { fontSize: 28, color: "#C9A84C", lineHeight: 1 },
  logoTitle: { fontFamily: "'Cormorant Garamond', serif", fontSize: 22, fontWeight: 600, color: "#E8E0D0", letterSpacing: "0.04em" },
  logoSub: { fontSize: 10, color: "#666", letterSpacing: "0.12em", fontFamily: "'DM Mono', monospace", marginTop: 2 },
  resetBtn: { background: "transparent", border: "1px solid #333", color: "#888", padding: "6px 14px", borderRadius: 4, fontSize: 12, cursor: "pointer", fontFamily: "'DM Sans', sans-serif" },
  main: { flex: 1, display: "flex", flexDirection: "column", position: "relative", zIndex: 1 },
  profileScreen: { flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "48px 24px", maxWidth: 720, margin: "0 auto", width: "100%" },
  profileHeading: { textAlign: "center", marginBottom: 48 },
  profileTitle: { fontFamily: "'Cormorant Garamond', serif", fontSize: 36, fontWeight: 500, color: "#E8E0D0", margin: "0 0 12px 0" },
  profileSubtitle: { color: "#888", fontSize: 15, margin: 0 },
  profileGrid: { display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 16, width: "100%", maxWidth: 560 },
  profileCard: { background: "#161616", border: "1px solid #2A2A2A", borderRadius: 8, padding: "28px 24px", cursor: "pointer", textAlign: "left", position: "relative", overflow: "hidden", transition: "border-color 0.2s, transform 0.15s", fontFamily: "'DM Sans', sans-serif" },
  profileCardAccent: { position: "absolute", top: 0, left: 0, right: 0, height: 3, borderRadius: "8px 8px 0 0" },
  profileCardIcon: { fontSize: 28, display: "block", marginBottom: 12 },
  profileCardLabel: { fontSize: 15, fontWeight: 500, color: "#E8E0D0", marginBottom: 6 },
  profileCardDesc: { fontSize: 12, color: "#666", lineHeight: 1.5 },
  disclaimer: { marginTop: 40, fontSize: 11, color: "#444", textAlign: "center", maxWidth: 400, lineHeight: 1.6, fontFamily: "'DM Mono', monospace" },
  chatWrapper: { flex: 1, display: "flex", flexDirection: "column", maxWidth: 720, margin: "0 auto", width: "100%", padding: "0 24px 24px", height: "calc(100vh - 73px)" },
  profileBadge: { display: "flex", alignItems: "center", gap: 8, padding: "10px 0", fontSize: 13, borderBottom: "1px solid #2A2A2A", marginBottom: 16 },
  messageList: { flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: 16, paddingBottom: 8 },
  messageBubble: { maxWidth: "85%", padding: "14px 18px", borderRadius: 8, lineHeight: 1.65, fontSize: 14 },
  aiBubble: { background: "#161616", border: "1px solid #2A2A2A", alignSelf: "flex-start", borderRadius: "2px 8px 8px 8px" },
  userBubble: { background: "#1E1E1E", border: "1px solid #333", alignSelf: "flex-end", color: "#D4CFC5", borderRadius: "8px 2px 8px 8px" },
  aiLabel: { fontSize: 11, fontFamily: "'DM Mono', monospace", letterSpacing: "0.08em", marginBottom: 6, fontWeight: 500 },
  messageText: { whiteSpace: "pre-wrap", color: "#D4CFC5" },
  typingDots: { display: "flex", gap: 6, padding: "4px 0" },
  dot: { width: 7, height: 7, borderRadius: "50%", background: "#555", display: "inline-block", animation: "blink 1.2s infinite" },
  errorMsg: { background: "#1A0A0A", border: "1px solid #4A2020", color: "#D48080", padding: "12px 16px", borderRadius: 6, fontSize: 13, alignSelf: "center" },
  inputArea: { display: "flex", gap: 10, alignItems: "flex-end", marginTop: 12, padding: "12px", background: "#141414", border: "1px solid #2A2A2A", borderRadius: 8 },
  textarea: { flex: 1, background: "transparent", border: "none", color: "#E8E0D0", fontSize: 14, fontFamily: "'DM Sans', sans-serif", resize: "none", lineHeight: 1.5, padding: 0 },
  sendBtn: { width: 40, height: 40, borderRadius: 6, border: "none", color: "#0E0E0E", fontSize: 20, fontWeight: 700, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, transition: "opacity 0.2s" },
  inputHint: { fontSize: 10, color: "#444", textAlign: "center", marginTop: 6, fontFamily: "'DM Mono', monospace" },
};
