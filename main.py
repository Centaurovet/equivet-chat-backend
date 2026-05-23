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
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "").strip()   # busca vetorial semântica
MODELO_SONNET  = "claude-sonnet-4-6"
MODELO_HAIKU   = "claude-haiku-4-5-20251001"
MAX_TOKENS     = 2000
RAG_CHUNKS     = 6   # número de trechos da literatura a incluir por resposta
RAG_SIM_MIN    = 0.45   # threshold de similaridade (Cohere cosine) — chunks abaixo são descartados

# Perfis que usam Sonnet (raciocínio clínico profundo)
PERFIS_SONNET  = {"vet", "farrier"}

# Web search nativo do Claude (server-side tool — Anthropic executa a busca).
# Habilita "consulta complementar à internet" mantendo literatura como fonte primária.
#
# IMPORTANTE: web_search injeta CADA resultado no contexto da próxima rodada do modelo.
# Uma única busca em site com PDFs gigantes (ex.: regulamentos de entidades equestres
# de 200+ páginas) já estoura o limite de 1M tokens. Mitigamos com 2 alavancas:
# 1. max_uses=1 — uma única busca por pergunta.
# 2. allowed_domains — restringe a sites institucionais leves (HTML, não PDFs gigantes).
#
# Se ainda assim estourar (ou se WEB_SEARCH=0), o fallback no /chat re-tenta sem tools.
WEB_SEARCH_HABILITADO = os.environ.get("WEB_SEARCH", "1").strip() not in {"0", "false", "False", ""}
WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 1,
    "allowed_domains": [
        # Entidades equestres brasileiras (institucionais, HTML leve)
        "abvaq.com.br",
        "abqm.com.br",
        "abccmm.com.br",
        "abccc.com.br",
        "cbh.org.br",
        "anpa.com.br",
        "abcccampolina.com.br",
        # Órgãos públicos brasileiros
        "gov.br",
        # Veterinária internacional (institucional / consensos clínicos)
        "aaep.org",
        "fei.org",
        # Publicações científicas (snippets curtos, sem PDFs gigantes)
        "pubmed.ncbi.nlm.nih.gov",
        "scielo.br",
        # Referência geral (pode ajudar em raças, eventos, definições)
        "wikipedia.org",
    ],
}

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

def _formatar_chunks(chunks: list) -> str:
    """Formata lista de chunks para injeção no system prompt."""
    linhas = []
    for r in chunks[:RAG_CHUNKS]:
        ref     = r.get("book", "Referência")
        cap     = r.get("chapter") or ""
        pag     = r.get("page_start") or ""
        ref_str = ref
        if cap:
            ref_str += f" — {cap}"
        if pag:
            ref_str += f", p.{pag}"
        linhas.append(f"[{ref_str}]\n{r['text']}")
    return "\n\n---\n\n".join(linhas)


def buscar_literatura(pergunta: str) -> str:
    """
    Busca chunks relevantes no Supabase.
    Modo A (preferido): busca vetorial semântica via Voyage AI + pgvector.
    Modo B (fallback):  busca por keyword AND+OR com tradução PT→EN/ES.
    Retorna string formatada para injetar no system prompt, ou "" se vazio.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return ""

    try:
        from supabase import create_client
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)

        # ── Modo A: busca vetorial (Cohere disponível) ────────────────────────
        if COHERE_API_KEY:
            try:
                import cohere
                co     = cohere.Client(api_key=COHERE_API_KEY)
                result = co.embed(texts=[pergunta], model="embed-multilingual-v3.0", input_type="search_query")
                query_embedding = result.embeddings[0]

                # Pede mais que o necessário para filtrar por similaridade depois.
                res = sb.rpc("match_documents", {
                    "query_embedding": query_embedding,
                    "match_count": RAG_CHUNKS * 2,
                }).execute()

                if res.data:
                    # Filtra chunks com similaridade abaixo do threshold —
                    # evita injetar trechos irrelevantes que confundem o modelo.
                    relevantes = [
                        r for r in res.data
                        if (r.get("similarity") or 0) >= RAG_SIM_MIN
                    ][:RAG_CHUNKS]
                    if relevantes:
                        return _formatar_chunks(relevantes)
                    # Se nenhum passa do threshold, cai para o fallback keyword
                    # (pode pegar algo útil por correspondência literal).
            except Exception:
                pass   # cai no fallback keyword

        # ── Modo B: busca por keyword (fallback) ──────────────────────────────
        traducoes   = traduzir_para_busca(pergunta)
        texto_busca = pergunta + " " + traducoes if traducoes else pergunta
        keywords    = extrair_keywords(texto_busca)
        if not keywords:
            return ""

        vistos: set = set()
        chunks: list = []

        kws_especificos = [kw for kw in keywords if len(kw) >= 6]

        # Fase 1: AND — exige 2 termos específicos no mesmo chunk
        if len(kws_especificos) >= 2:
            for i in range(len(kws_especificos) - 1):
                if len(chunks) >= RAG_CHUNKS:
                    break
                kw1, kw2 = kws_especificos[i], kws_especificos[i + 1]
                try:
                    res = (
                        sb.table("literatura")
                        .select("chunk_id,book,author,chapter,page_start,text")
                        .ilike("text", f"%{kw1}%")
                        .ilike("text", f"%{kw2}%")
                        .limit(4)
                        .execute()
                    )
                    for row in res.data:
                        cid = row.get("chunk_id") or row["text"][:60]
                        if cid not in vistos:
                            vistos.add(cid)
                            chunks.append(row)
                except Exception:
                    continue

        # Fase 2: OR — keyword individual para complementar
        if len(chunks) < RAG_CHUNKS:
            for kw in keywords[:8]:
                if len(chunks) >= RAG_CHUNKS:
                    break
                try:
                    res = (
                        sb.table("literatura")
                        .select("chunk_id,book,author,chapter,page_start,text")
                        .ilike("text", f"%{kw}%")
                        .limit(2)
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
        return _formatar_chunks(chunks)

    except Exception:
        return ""

# ── System prompts ────────────────────────────────────────────────────────────
# Bloco compartilhado: domínio (mercado equino BR) + hierarquia obrigatória de fontes.
# Vai ANTES da persona de cada perfil para enquadrar o modelo desde o início.
_CONTEXTO_DOMINIO = (
    "DOMÍNIO: ECOSSISTEMA EQUINO BRASILEIRO COMPLETO. "
    "Público: veterinários, treinadores, ferradores, criadores, proprietários e amantes do "
    "cavalo no Brasil.\n\n"
    "O EquiVet IA cobre TODO o universo equestre brasileiro, incluindo:\n"
    "• MEDICINA E CIRURGIA EQUINA — diagnóstico, tratamento, farmacologia, exames complementares.\n"
    "• PERFORMANCE E TREINAMENTO — condicionamento, cargas de trabalho, recuperação, nutrição esportiva.\n"
    "• PODOLOGIA E FERRAGEAMENTO — anatomia do casco, mecânica, ferraduras, patologias podais.\n"
    "• MANEJO E NUTRIÇÃO — alimentação, suplementação, manejo sanitário, pastagens.\n"
    "• REPRODUÇÃO E CRIAÇÃO — manejo reprodutivo, neonatologia, genética, registro genealógico.\n"
    "• BEM-ESTAR ANIMAL — etologia, condições de alojamento, transporte.\n"
    "• MERCADO E EVENTOS — calendários de provas, leilões, regulamentos esportivos, raças, "
    "criatórios, valores de mercado, profissionalização.\n"
    "• ENTIDADES E LEGISLAÇÃO — ABQM, ABCCMM (Mangalarga Marchador), ABCCC (Crioulo), "
    "ABVAQ (Vaquejada), CBH (Hipismo), FEI, AAEP, IBGE, MAPA, normas sanitárias, GTA, exames "
    "obrigatórios (AIE, mormo).\n\n"
    "CONTEXTO BRASILEIRO ESPECÍFICO:\n"
    "• Raças prevalentes: Mangalarga Marchador, Mangalarga Paulista, Campolina, Crioulo, "
    "Quarto de Milha, Pampa, Pantaneiro, Brasileiro de Hipismo, PSI, Anglo-Árabe.\n"
    "• Modalidades brasileiras: vaquejada, três tambores, seis balizas, laço comprido, "
    "apartação, rédeas, marcha (batida e picada), hipismo clássico, polo, enduro, trabalho de campo.\n"
    "• Particularidades nacionais: clima tropical (verminose, dermatites, manejo sanitário), "
    "pastagens tropicais (Brachiaria, Tifton, fotossensibilização), ferrageamento adaptado aos "
    "pisos brasileiros, doenças regionais (mormo, AIE, encefalomielite, garrotilho).\n"
    "NÃO assuma contexto europeu/ibérico (toureio, equitação clássica de tradição lusitana) — "
    "isso confunde o público brasileiro.\n\n"
    "REGRA IMPORTANTE DE ESCOPO: NUNCA recuse uma pergunta como \"fora do escopo do EquiVet IA\" "
    "se ela se refere ao universo equino brasileiro — mesmo que seja sobre eventos, calendários, "
    "raças, mercado, criatórios, leilões, regulamentos, legislação ou cultura equestre. "
    "Use web_search para informação atualizada quando a literatura indexada não cobrir. "
    "Recusas só são apropriadas para tópicos genuinamente não-equinos (política, finanças não "
    "relacionadas, assuntos pessoais não-equestres)."
)

_HIERARQUIA_FONTES = (
    "HIERARQUIA OBRIGATÓRIA DE FONTES (do mais para o menos autoritativo):\n"
    "1. LITERATURA FORNECIDA no bloco REFERÊNCIAS BIBLIOGRÁFICAS abaixo (Smith Large Animal "
    "Surgery, Adams Claudicación en el Caballo). Esta é a fonte PRIMÁRIA e AUTORITATIVA.\n"
    "2. WEB SEARCH (tool web_search disponível) — para consensos clínicos recentes, dados "
    "de mercado, epidemiologia regional brasileira e atualizações pós-publicação dos livros. "
    "Acione APENAS quando a literatura indexada não cobrir o tópico ou estiver desatualizada.\n"
    "3. CONHECIMENTO PRÓPRIO do modelo — APENAS quando 1 e 2 não cobrirem o tópico. "
    "Declare explicitamente.\n\n"
    "REGRAS DE USO:\n"
    "• Quando a literatura fornecida cobrir a pergunta, BASEIE-SE NELA e cite OBRIGATORIAMENTE "
    "no formato [Smith Large Animal Surgery, p.X] ou [Adams Claudicación, p.X] no fim do "
    "parágrafo ou frase correspondente.\n"
    "• Se a literatura cobrir parcialmente, use-a como espinha dorsal e complemente com "
    "web search ou conhecimento próprio — SEMPRE declarando: \"A literatura indexada não "
    "detalha X — segundo [web: domínio.com] / com base em conhecimento veterinário geral, ...\".\n"
    "• Para uso da tool web_search: priorize fontes brasileiras (sites oficiais das entidades "
    "ABQM, ABCCMM, ABVAQ, CBH; revistas veterinárias BR; sites do MAPA) e consensos científicos "
    "atualizados (PubMed, Equine Veterinary Journal, JAVMA, AAEP).\n"
    "• ECONOMIA DE BUSCAS: use no MÁXIMO 2 chamadas de web_search por pergunta. "
    "Formule uma query única e específica (ex.: \"calendário vaquejada 2026 ABVAQ\") em vez "
    "de várias buscas amplas. Se a primeira busca já trouxe a resposta, NÃO faça uma segunda.\n"
    "• Em caso de conflito entre literatura fornecida e conhecimento próprio, PREVALECE a literatura. "
    "Em conflito entre literatura e web search recente, mencione AMBAS as posições e contextualize.\n"
    "• NUNCA invente páginas, capítulos ou citações. Se não tem certeza da referência, omita-a.\n"
    "• Se nenhum trecho relevante foi fornecido E o tópico é estritamente veterinário/clínico, "
    "use web_search OU diga: \"A base de literatura indexada (Smith/Adams) não cobre este "
    "tópico especificamente.\" antes de responder."
)

_BASE_RAG = (
    "\n\nAplique a HIERARQUIA OBRIGATÓRIA DE FONTES descrita acima. "
    "Os trechos abaixo são a fonte PRIMÁRIA — baseie-se neles e cite [Livro, p.X]. "
    "Se algum trecho não for pertinente à pergunta específica, ignore APENAS ele "
    "(não a literatura como um todo)."
)

SYSTEM_PROMPTS = {
    "vet": (
        _CONTEXTO_DOMINIO + "\n\n" + _HIERARQUIA_FONTES + "\n\n"
        "PERSONA: Você é o EquiVet IA, assistente clínico de medicina equina desenvolvido "
        "pela Centaurovet. Você está conversando com um MÉDICO-VETERINÁRIO EQUINO brasileiro. "
        "Use linguagem técnica precisa: termos latinos, nomenclatura farmacológica, protocolos "
        "clínicos, doses (mg/kg), vias de administração. Seja direto e denso como um colega "
        "consultando outro. Sem condescendência. "
        "Quando a dúvida envolver diagnóstico ou tratamento, pergunte: raça, idade, peso "
        "estimado, sinais clínicos específicos, achados de exame físico relevantes, "
        "exames complementares já realizados. "
        "Pode discutir diagnósticos diferenciais, indicações cirúrgicas, exames complementares. "
        "Responda em português do Brasil. Seja conciso. "
        "Use bullet points quando listar diagnósticos diferenciais ou protocolos."
    ),
    "owner": (
        _CONTEXTO_DOMINIO + "\n\n" + _HIERARQUIA_FONTES + "\n\n"
        "PERSONA: Você é o EquiVet IA, assistente de saúde equina desenvolvido pela Centaurovet. "
        "Você está conversando com um PROPRIETÁRIO DE CAVALO no Brasil. "
        "Use linguagem clara, acolhedora e precisa — sem ser condescendente, sem jargão "
        "desnecessário. Cite a literatura nos termos do público leigo (\"segundo o tratado "
        "Smith, capítulo de cólicas...\") mantendo o formato [Livro, p.X]. "
        "Quando alguém descrever um problema, pergunte com cuidado: raça, idade, peso "
        "aproximado, sintomas observados (o que viram, quando começou), se já viu veterinário, "
        "manejo (estábulo/piquete, alimentação, rotina). "
        "Sempre que houver risco de emergência (cólica, dificuldade respiratória, trauma, "
        "claudicação aguda severa), sinalize com clareza e oriente a chamar veterinário "
        "imediatamente. Nunca substitua atendimento clínico presencial. "
        "Responda em português do Brasil. "
        "Tom: como um veterinário experiente que explica para um amigo que ama o animal."
    ),
    "trainer": (
        _CONTEXTO_DOMINIO + "\n\n" + _HIERARQUIA_FONTES + "\n\n"
        "PERSONA: Você é o EquiVet IA, assistente do TREINADOR EQUINO brasileiro, desenvolvido "
        "pela Centaurovet. "
        "Foco principal: condicionamento físico, cargas de trabalho, recuperação, sinais de "
        "overtraining, nutrição esportiva, prevenção de lesões musculoesqueléticas — sempre "
        "adaptado às modalidades brasileiras (vaquejada, três tambores, seis balizas, marcha, "
        "hipismo, polo, rédeas, apartação). "
        "Também atende perguntas amplas que interessam ao treinador: calendários de provas, "
        "regulamentos esportivos das entidades (ABQM, ABVAQ, CBH, ABCCMM), critérios de "
        "julgamento, qualificações, leilões e mercado de cavalos esportivos. "
        "Quando a dúvida envolver performance ou claudicação, pergunte: modalidade esportiva, "
        "frequência e intensidade de treinos, histórico de lesões, ferrageamento atual, "
        "tipo de piso de treino. "
        "Use linguagem técnica mas acessível ao treinador (não ao veterinário). "
        "Pode referenciar parâmetros fisiológicos (FC de recuperação, lactato, VO2) quando pertinente. "
        "Responda em português do Brasil. Seja prático e objetivo."
    ),
    "farrier": (
        _CONTEXTO_DOMINIO + "\n\n" + _HIERARQUIA_FONTES + "\n\n"
        "PERSONA: Você é o EquiVet IA, assistente do FERRADOR brasileiro, desenvolvido pela "
        "Centaurovet. "
        "Foco principal: anatomia do casco, mecânica do passo, desequilíbrios, defeitos de "
        "aprumos, tipos de ferradura (comum, mata-junta, ortopédica, egg-bar, heart-bar), "
        "materiais, patologias do casco (laminite, murça, abscessos, quartos partidos). "
        "Considere pisos típicos brasileiros (areia, terra batida, pedregoso, piso de baia úmida). "
        "Também atende perguntas amplas que interessam ao ferrador: cursos e capacitações no "
        "Brasil, eventos de ferrageamento, regulamentações esportivas que afetam o ferrageamento, "
        "fornecedores e materiais disponíveis no mercado brasileiro. "
        "Quando a dúvida envolver ferrageamento ou claudicação, pergunte: membro acometido, "
        "tipo de piso predominante, modalidade esportiva, histórico de ferrageamento anterior, "
        "ciclo de ferrageamento atual. "
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

@app.get("/debug-rag")
async def debug_rag(q: str = "laminite crônica achados radiográficos"):
    """Diagnóstico completo do pipeline RAG para uma query de teste."""
    resultado: dict = {"query_original": q}

    # 1. Tradução
    traducao = traduzir_para_busca(q)
    resultado["traducao"] = traducao or "(falhou ou vazia)"
    texto_busca = q + " " + traducao if traducao else q
    resultado["texto_busca_completo"] = texto_busca

    # 2. Keywords extraídos
    keywords = extrair_keywords(texto_busca)
    resultado["keywords"] = keywords

    # 3. Busca no Supabase
    if not SUPABASE_URL or not SUPABASE_KEY:
        resultado["supabase"] = "não configurado"
        return resultado

    try:
        from supabase import create_client
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
        chunks_por_keyword: dict = {}
        for kw in keywords[:10]:
            try:
                res = (
                    sb.table("literatura")
                    .select("chunk_id,book,page_start,text")
                    .ilike("text", f"%{kw}%")
                    .limit(2)
                    .execute()
                )
                chunks_por_keyword[kw] = [
                    {"chunk_id": r["chunk_id"], "book": r["book"],
                     "page": r.get("page_start"), "preview": r["text"][:80]}
                    for r in res.data
                ]
            except Exception as e:
                chunks_por_keyword[kw] = f"erro: {e}"
        resultado["chunks_por_keyword"] = chunks_por_keyword
        total = sum(len(v) for v in chunks_por_keyword.values() if isinstance(v, list))
        resultado["total_chunks_encontrados"] = total
    except Exception as e:
        resultado["supabase_erro"] = str(e)

    return resultado

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

    # Monta argumentos da chamada — adiciona web_search como tool quando habilitado.
    # web_search é um server-side tool: a Anthropic executa a busca automaticamente
    # e devolve o resultado consolidado em text blocks (não há tool-use loop manual).
    kwargs = {
        "model": modelo,
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": msgs,
    }
    if WEB_SEARCH_HABILITADO:
        kwargs["tools"] = [WEB_SEARCH_TOOL]

    def _chamar(args: dict):
        """Chama a API e concatena text blocks. Compatível com tool use (web_search)."""
        client = anthropic.Anthropic(api_key=API_KEY)
        r = client.messages.create(**args)
        partes = [
            getattr(b, "text", "") for b in r.content
            if getattr(b, "type", None) == "text"
        ]
        t = "\n".join(p for p in partes if p).strip()
        if not t:
            t = getattr(r.content[0], "text", "") if r.content else ""
        return t

    try:
        try:
            texto = _chamar(kwargs)
        except Exception as e:
            # Fallback: web_search puxou contexto gigante (ex.: PDFs grandes) e estourou
            # o limite de 1M tokens. Re-tenta SEM tools — perde a busca atualizada mas
            # responde com literatura + conhecimento próprio em vez de devolver erro.
            # Logs imprimem nos Deploy Logs do Railway para diagnóstico.
            mensagem_erro = str(e).lower()
            estourou_contexto = (
                "prompt is too long" in mensagem_erro
                or "too long" in mensagem_erro
                or ("context" in mensagem_erro and "limit" in mensagem_erro)
                or ("maximum" in mensagem_erro and "token" in mensagem_erro)
            )
            tem_tools = "tools" in kwargs
            print(
                f"[chat] 1a chamada falhou: tipo={type(e).__name__} "
                f"estourou_contexto={estourou_contexto} tem_tools={tem_tools} "
                f"msg={str(e)[:300]}"
            )

            if estourou_contexto and tem_tools:
                print("[chat] entrando no FALLBACK sem tools")
                kwargs_sem_tools = {k: v for k, v in kwargs.items() if k != "tools"}
                # Sinaliza ao modelo que a busca falhou para que ele avise o usuário.
                kwargs_sem_tools["system"] = (
                    kwargs_sem_tools["system"]
                    + "\n\n[AVISO INTERNO: web_search indisponível nesta resposta — "
                    "as fontes consultadas estavam grandes demais. Responda com literatura "
                    "indexada + conhecimento próprio e mencione ao usuário que dados "
                    "atualizados da internet não puderam ser carregados desta vez.]"
                )
                try:
                    texto = _chamar(kwargs_sem_tools)
                    print("[chat] FALLBACK OK — resposta gerada sem tools")
                except Exception as e2:
                    print(
                        f"[chat] FALLBACK TAMBÉM FALHOU: tipo={type(e2).__name__} "
                        f"msg={str(e2)[:300]}"
                    )
                    raise
            else:
                # Não é erro de contexto OU já não tinha tools — propaga.
                raise
        return {"resposta": texto}

    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e.message))
    except Exception as e:
        # Captura tudo o mais para que erros inesperados apareçam nos logs do Railway.
        print(f"[chat] erro não tratado: tipo={type(e).__name__} msg={str(e)[:300]}")
        raise HTTPException(status_code=500, detail="Erro interno. Tente novamente.")
