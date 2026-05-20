"""
EquiVet IA — Backend Proxy
Protege a API Key da Anthropic e adiciona rate limiting.
Deploy: Railway / Render / qualquer servidor Python.
"""

import os
import time
from collections import defaultdict
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic

# ── Configuração ──────────────────────────────────────────────────────────────
API_KEY        = os.environ.get("ANTHROPIC_API_KEY", "")
MODELO_SONNET  = "claude-sonnet-4-6"
MODELO_HAIKU   = "claude-haiku-4-5-20251001"
MAX_TOKENS     = 2000

# Perfis que usam Sonnet (raciocínio clínico profundo)
PERFIS_SONNET  = {"vet", "farrier"}

# Domínios autorizados a usar o chat (adicione o seu domínio aqui)
ORIGENS_PERMITIDAS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

# Rate limiting simples: máx. requisições por janela de tempo por IP
RATE_LIMITE      = int(os.environ.get("RATE_LIMIT", "15"))       # requisições
RATE_JANELA_SEG  = int(os.environ.get("RATE_WINDOW_SEC", "60"))  # por minuto

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="EquiVet IA Chat API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGENS_PERMITIDAS,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
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
            detail=f"Muitas requisições. Aguarde um momento e tente novamente."
        )
    _contadores[ip].append(agora)

# ── System prompts (espelho do frontend) ─────────────────────────────────────
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
    perfil: str          # "vet" | "owner" | "trainer" | "farrier"
    mensagens: List[Mensagem]

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def raiz():
    return {"status": "EquiVet IA Chat API online"}

@app.get("/health")
def health():
    return {"ok": True, "modelo": MODELO}

@app.get("/test-api")
async def test_api():
    """Endpoint de diagnóstico — testa a conexão com a Anthropic."""
    if not API_KEY:
        return {"erro": "API_KEY não configurada"}
    try:
        client = anthropic.Anthropic(api_key=API_KEY)
        r = client.messages.create(
            model=MODELO_SONNET,
            max_tokens=10,
            messages=[{"role": "user", "content": "Olá"}],
        )
        return {"ok": True, "sonnet": MODELO_SONNET, "haiku": MODELO_HAIKU, "resposta": r.content[0].text}
    except Exception as e:
        return {"erro": str(e), "tipo": type(e).__name__, "modelo": MODELO}

@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
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

    # Escolhe o modelo pelo perfil
    modelo = MODELO_SONNET if req.perfil in PERFIS_SONNET else MODELO_HAIKU

    # Chama o Claude
    try:
        client = anthropic.Anthropic(api_key=API_KEY)
        # A Anthropic exige que o primeiro item seja sempre "user"
        # Remove mensagens iniciais do assistente (boas-vindas do frontend)
        msgs = [{"role": m.role, "content": m.content} for m in req.mensagens]
        primeiro_user = next((i for i, m in enumerate(msgs) if m["role"] == "user"), None)
        if primeiro_user is None:
            raise HTTPException(status_code=400, detail="Nenhuma mensagem do usuário.")
        msgs = msgs[primeiro_user:]

        resposta = client.messages.create(
            model=modelo,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPTS[req.perfil],
            messages=msgs,
        )
        texto = resposta.content[0].text
        return {"resposta": texto}

    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e.message))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Erro interno. Tente novamente.")
