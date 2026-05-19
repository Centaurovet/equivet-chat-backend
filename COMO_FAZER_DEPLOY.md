# Deploy do Backend EquiVet IA Chat

## Opção 1 — Railway (recomendado, grátis para começar)

1. Acesse https://railway.app e crie uma conta gratuita
2. Clique em **New Project → Deploy from GitHub repo**
3. Conecte seu GitHub e faça upload desta pasta `chat_backend/`
   - Ou: crie um repositório no GitHub com apenas esta pasta
4. No painel do Railway, vá em **Variables** e adicione:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ALLOWED_ORIGINS=https://seusite.com.br
   ```
5. O Railway detecta o `Procfile` automaticamente e faz o deploy
6. Copie a URL gerada (ex: `https://equivet-chat.railway.app`)
7. Cole essa URL no `EquiVetChat.jsx` na linha:
   ```js
   const BACKEND_URL = "https://equivet-chat.railway.app";
   ```

---

## Opção 2 — Render (também grátis)

1. Acesse https://render.com e crie uma conta
2. Clique em **New → Web Service**
3. Conecte o repositório com a pasta `chat_backend/`
4. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Em **Environment Variables**, adicione `ANTHROPIC_API_KEY`
6. Copie a URL e atualize o `EquiVetChat.jsx`

---

## Testar localmente antes do deploy

```bash
cd chat_backend
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn main:app --reload
```

Acesse http://localhost:8000 — deve retornar `{"status": "EquiVet IA Chat API online"}`

---

## Checklist antes de abrir ao público

- [ ] ANTHROPIC_API_KEY configurada no servidor (nunca no frontend)
- [ ] ALLOWED_ORIGINS com o domínio real (não usar * em produção)
- [ ] URL do backend atualizada no EquiVetChat.jsx
- [ ] Teste com cada um dos 4 perfis
- [ ] Aviso legal visível na interface ("não substitui avaliação clínica")
