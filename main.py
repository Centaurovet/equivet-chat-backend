"""
EquiVet IA — Backend Proxy com RAG
Protege a API Key da Anthropic, adiciona rate limiting e busca literatura.
Deploy: Railway / Render / qualquer servidor Python.
"""

import os
import re
import time
import unicodedata
from collections import defaultdict
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic

# ── Configuração ──────────────────────────────────────────────────────────────
API_KEY        = os.environ.get("ANTHROPIC_API_KEY", "").strip()
API_SECRET     = os.environ.get("API_SECRET", "").strip()   # token obrigatório nos headers do frontend
SUPABASE_URL   = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY   = os.environ.get("SUPABASE_KEY", "")
MODELO_SONNET  = "claude-sonnet-4-6"
MODELO_HAIKU   = "claude-haiku-4-5-20251001"
MAX_TOKENS     = 2000
RAG_CHUNKS     = 4   # número de trechos da literatura a incluir por resposta

# Perfis que usam Sonnet (raciocínio clínico profundo)
PERFIS_SONNET  = {"vet", "farrier"}

# Domínios autorizados a usar o chat
ORIGENS_PERMITIDAS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

# Rate limiting simples: máx. requisições por janela de tempo por IP
RATE_LIMITE      = int(os.environ.get("RATE_LIMIT", "15"))
RATE_JANELA_SEG  = int(os.environ.get("RATE_WINDOW_SEC", "60"))

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="EquiVet IA Chat API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGENS_PERMITIDAS,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# ── Rate limiter em memória ───────────────────────────────────────────────────
_contadores: dict = defaultdict(list)

def verificar_rate_limit(ip: str):
    agora = time.time()
    janela = agora - RATE_JANELA_SEG
    _contadores[ip] = [t for t in _contadores[ip] if t > janela]
    if len(_contadores[ip]) >= RATE_LIMITE:
        raise HTTPException(
            status_code=429,
            detail="Muitas requisições. Aguarde um momento e tente novamente."
        )
    _contadores[ip].append(agora)

# ── RAG: busca de literatura no Supabase ─────────────────────────────────────
# Stopwords PT/EN/ES a ignorar na extração de keywords
_STOPWORDS = {
    "a","o","as","os","um","uma","de","do","da","dos","das","em","no","na",
    "nos","nas","e","é","são","foi","que","se","por","para","com","como",
    "não","tem","ter","ser","this","the","and","for","with","can","you",
    "your","que","del","los","las","una","con","por","para","como","es",
    "en","se","vc","eu","ele","ela","meu","seu","seu","está","isso","isto",
    "qual","quais","quando","onde","quem","mais","muito","bem","mas","ou",
}

def extrair_keywords(texto: str) -> List[str]:
    """Extrai palavras ≥4 letras de texto misto PT/EN/ES, ignorando stopwords."""
    # Captura todo o alfabeto latino (incluindo acentuados)
    palavras = re.findall(r"[a-zA-ZÀ-ɏ]+", texto.lower())
    resultado: List[str] = []
    vistos: set = set()
    for p in palavras:
        if len(p) >= 4 and p not in _STOPWORDS and p not in vistos:
            resultado.append(p)
            vistos.add(p)
    return resultado

def traduzir_para_busca(pergunta: str) -> str:
    """
    Usa Claude Haiku para traduzir a pergunta PT → EN + ES.
    Retorna as traduções concatenadas ao texto original para ampliar os keywords.
    Latência típica: ~0.5s. Falha silenciosamente.
    """
    if not API_KEY:
        return ""
    try:
        client = anthropic.Anthropic(api_key=API_KEY)
        r = client.messages.create(
            model=MODELO_HAIKU,
            max_tokens=120,
            messages=[{
                "role": "user",
                "content": (
                    "Translate the following equine medicine question to English and Spanish. "
                    "Return ONLY the two translations on separate lines. No labels, no punctuation changes.\n\n"
                    f"{pergunta}"
                )
            }]
        )
        return r.content[0].text.strip()
    except Exception:
        return ""

def buscar_literatura(pergunta: str) -> str:
    """
    Busca chunks relevantes no Supabase por keyword (ilike).
    Traduz PT→EN/ES antes da busca para encontrar termos nos livros em inglês/espanhol.
    Retorna string formatada para injetar no system prompt, ou "" se vazio.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return ""
    try:
        from supabase import create_client
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)

        # Combina PT + EN + ES para maximizar cobertura de keywords
        traducoes = traduzir_para_busca(pergunta)
        texto_busca = pergunta + " " + traducoes if traducoes else pergunta

        keywords = extrair_keywords(texto_busca)
        if not keywords:
            return ""

        vistos: set = set()
        chunks: list = []

        # Tenta cada keyword até ter RAG_CHUNKS trechos únicos
        for kw in keywords[:8]:
            if len(chunks) >= RAG_CHUNKS:
                break
            try:
                res = (
                    sb.table("literatura")
                    .select("chunk_id,book,author,chapter,page_start,text")
                    .ilike("text", f"%{kw}%")
                    .limit(3)
                    .execute()
                )
                for row in res.data:
                    cid = row.get("chunk_id") or row["text"][:60]
                    if cid not in vistos:
                        vistos.add(cid)
                        chunks.append(row)
            except Exception:
                continue

        if not chunks:
            return ""

        linhas = []
        for r in chunks[:RAG_CHUNKS]:
            ref = r.get("book", "Referência")
            cap = r.get("chapter") or ""
            pag = r.get("page_start") or ""
            ref_str = f"{ref}"
            if cap:
                ref_str += f" — {cap}"
            if pag:
                ref_str += f", p.{pag}"
            linhas.append(f"[{ref_str}]\n{r['text']}")

        return "\n\n---\n\n".join(linhas)

    except Exception:
        return ""

# ── System prompts ────────────────────────────────────────────────────────────
_BASE_RAG = (
    "\n\nQuando relevante, cite as referências bibliográficas fornecidas acima "
    "indicando o livro e a página. Se os trechos não forem pertinentes à pergunta, ignore-os."
)

SYSTEM_PROMPTS = {
    "vet": (
        "Você é o EquiVet IA, assistente clínico de medicina equina desenvolvido pela Centaurovet. "
        "Você está conversando com um MÉDICO-VETERINÁRIO EQUINO. Use linguagem técnica precisa: "
        "termos latinos, nomenclatura farmacológica, protocolos clínicos, doses, vias de administração. "
        "Seja direto e denso como um colega consultando outro. Sem condescendência. "
        "Quando a dúvida envolver diagnóstico ou tratamento, pergunte: espécie/raça, idade, peso estimado, "
        "sinais clínicos específicos, achados de exame físico relevantes. "
        "Pode discutir diagnósticos diferenciais, indicações cirúrgicas, exames complementares. "
        "Responda em português do Brasil. Seja conciso. "
        "Use bullet points quando listar diagnósticos diferenciais ou protocolos."
    ),
    "owner": (
        "Você é o EquiVet IA, assistente de saúde equina desenvolvido pela Centaurovet. "
        "Você está conversando com um PROPRIETÁRIO DE CAVALO. "
        "Use linguagem clara, acolhedora e precisa — sem ser condescendente, sem jargão desnecessário. "
        "Quando alguém descrever um problema, pergunte com cuidado: raça, idade, peso aproximado, "
        "sintomas observados (o que viram, quando começou), se já viu veterinário. "
        "Sempre que houver risco de emergência (cólica, dificuldade respiratória, trauma), "
        "sinalize com clareza e oriente a chamar veterinário imediatamente. "
        "Responda em português do Brasil. "
        "Tom: como um veterinário experiente que explica para um amigo que ama o animal."
    ),
    "trainer": (
        "Você é o EquiVet IA, assistente técnico de performance equina desenvolvido pela Centaurovet. "
        "Você está conversando com um TREINADOR EQUINO. "
        "Foco: condicionamento físico, cargas de trabalho, recuperação, sinais de overtraining, "
        "nutrição esportiva, prevenção de lesões musculoesqueléticas. "
        "Quando a dúvida envolver performance ou claudicação, pergunte: modalidade esportiva, "
        "frequência e intensidade de treinos, histórico de lesões, ferrageamento atual. "
        "Use linguagem técnica mas acessível. "
        "Pode referenciar parâmetros fisiológicos (FC de recuperação, lactato, VO2) quando pertinente. "
        "Responda em português do Brasil. Seja prático e objetivo."
    ),
    "farrier": (
        "Você é o EquiVet IA, assistente técnico de podologia equina desenvolvido pela Centaurovet. "
        "Você está conversando com um FERRADOR. "
        "Foco: anatomia do casco, mecânica do passo, desequilíbrios, defeitos de aprumos, "
        "tipos de ferradura, materiais, patologias do casco (laminite, murça, abscessos). "
        "Quando a dúvida envolver ferrageamento ou claudicação, pergunte: membro acometido, "
        "tipo de piso predominante, modalidade esportiva, histórico de ferrageamento anterior. "
        "Respeite o conhecimento prático do ferrador. Seja técnico e colaborativo. "
        "Responda em português do Brasil. Seja direto."
    ),
}

# ── Schemas ───────────────────────────────────────────────────────────────────
class Mensagem(BaseModel):
    role: str   # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    perfil: str
    mensagens: List[Mensagem]

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def raiz():
    return {"status": "EquiVet IA Chat API online", "version": "2.0", "rag": bool(SUPABASE_URL)}

@app.get("/health")
def health():
    return {"ok": True, "sonnet": MODELO_SONNET, "haiku": MODELO_HAIKU, "rag": bool(SUPABASE_URL)}

@app.get("/debug-auth")
async def debug_auth(request: Request):
    """Diagnóstico seguro: compara tamanhos sem expor valores."""
    token_recebido = request.headers.get("X-API-Key", "").strip()
    secret_configurado = API_SECRET
    return {
        "api_secret_configurado": bool(secret_configurado),
        "api_secret_len": len(secret_configurado),
        "token_recebido_len": len(token_recebido),
        "tokens_iguais": token_recebido == secret_configurado,
        "token_primeiros_4": token_recebido[:4] if token_recebido else "(vazio)",
        "secret_primeiros_4": secret_configurado[:4] if secret_configurado else "(vazio)",
    }

@app.get("/test-api")
async def test_api():
    """Diagnóstico: testa conexão com Anthropic e Supabase."""
    resultado: dict = {}
    if not API_KEY:
        resultado["anthropic"] = "API_KEY não configurada"
    else:
        try:
            client = anthropic.Anthropic(api_key=API_KEY)
            r = client.messages.create(
                model=MODELO_SONNET,
                max_tokens=10,
                messages=[{"role": "user", "content": "Olá"}],
            )
            resultado["anthropic"] = {"ok": True, "resposta": r.content[0].text}
        except Exception as e:
            resultado["anthropic"] = {"erro": str(e)}

    if not SUPABASE_URL:
        resultado["supabase"] = "SUPABASE_URL não configurada"
    else:
        try:
            from supabase import create_client
            sb = create_client(SUPABASE_URL, SUPABASE_KEY)
            c = sb.table("literatura").select("chunk_id", count="exact").limit(1).execute()
            resultado["supabase"] = {"ok": True, "total_chunks": c.count}
        except Exception as e:
            resultado["supabase"] = {"erro": str(e)}

    return resultado

@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    # Verifica token secreto
    if API_SECRET:
        token = request.headers.get("X-API-Key", "").strip()
        if token != API_SECRET:
            raise HTTPException(status_code=401, detail="Não autorizado.")

    # Rate limiting
    ip = request.client.host
    verificar_rate_limit(ip)

    # Valida perfil
    if req.perfil not in SYSTEM_PROMPTS:
        raise HTTPException(status_code=400, detail="Perfil inválido.")

    if not req.mensagens:
        raise HTTPException(status_code=400, detail="Nenhuma mensagem enviada.")

    if not API_KEY:
        raise HTTPException(status_code=500, detail="API Key não configurada no servidor.")

    # Filtra mensagens (Anthropic exige que o primeiro item seja "user")
    msgs = [{"role": m.role, "content": m.content} for m in req.mensagens]
    primeiro_user = next((i for i, m in enumerate(msgs) if m["role"] == "user"), None)
    if primeiro_user is None:
        raise HTTPException(status_code=400, detail="Nenhuma mensagem do usuário.")
    msgs = msgs[primeiro_user:]

    # Pega a última mensagem do usuário para busca RAG
    ultima_msg_user = next(
        (m["content"] for m in reversed(msgs) if m["role"] == "user"), ""
    )

    # Busca literatura relevante
    contexto_literatura = buscar_literatura(ultima_msg_user)

    # Monta system prompt: base + literatura (se encontrou)
    system = SYSTEM_PROMPTS[req.perfil]
    if contexto_literatura:
        system = (
            system
            + "\n\n══════════════════════════════\n"
            + "REFERÊNCIAS BIBLIOGRÁFICAS RELEVANTES (use quando pertinente):\n\n"
            + contexto_literatura
            + "\n══════════════════════════════"
            + _BASE_RAG
        )

    # Escolhe o modelo pelo perfil
    modelo = MODELO_SONNET if req.perfil in PERFIS_SONNET else MODELO_HAIKU

    try:
        client = anthropic.Anthropic(api_key=API_KEY)
        resposta = client.messages.create(
            model=modelo,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=msgs,
        )
        texto = resposta.content[0].text
        return {"resposta": texto}

    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e.message))
    except Exception:
        raise HTTPException(status_code=500, detail="Erro interno. Tente novamente.")
