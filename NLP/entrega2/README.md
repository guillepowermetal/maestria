# Exploracion de HuggingChat con hugchat

Este mini proyecto prueba posibilidades conversacionales de HuggingChat usando
la libreria no oficial [`hugchat`](https://github.com/Soulter/hugging-chat-api).

## Hallazgos rapidos

- HuggingChat permite conversar con modelos alojados en Hugging Face desde una
  interfaz tipo chatbot.
- `hugchat` expone una API Python no oficial para chat basico, streaming,
  busqueda web, memoria de conversaciones, cambio de LLM y asistentes.
- La autenticacion se hace con una cuenta de Hugging Face y cookies locales.
- No conviene poner email ni password en el codigo. Usa variables de entorno.
- El repositorio advierte que no es producto oficial de Hugging Face y que no se
  deben hacer peticiones de alta frecuencia.

## Tecnicas de prompting para experimentar

1. Prompt base: pedir una respuesta directa y comparar su claridad.
2. Prompt con rol: asignar una perspectiva, por ejemplo docente, analista o
   revisor tecnico.
3. Prompt con formato: pedir tabla, JSON, lista corta o criterios de evaluacion.
4. Prompt con contexto: explicar audiencia, objetivo, restricciones y material.
5. Prompt iterativo: pedir una primera respuesta, luego refinar tono, longitud o
   profundidad.
6. Prompt socratico: pedir que el chatbot haga preguntas antes de responder.
7. Prompt con busqueda web: activar `web_search=True` para tareas que dependen
   de informacion reciente.

## Instalacion

```bash
uv venv
uv pip install -r requirements.txt
```

## Configuracion

Opcion A: usar un archivo JSON de cookies guardado. Esta suele ser la opcion
mas estable porque `hugchat` es una libreria no oficial y el login por
email/password puede romperse cuando Hugging Face cambia su flujo web.

```bash
HUGCHAT_COOKIE_FILE="./cookies/huggingface.json"
```

El JSON puede contener un objeto simple con cookies, por ejemplo:

```json
{
  "token": "...",
  "hf-chat": "..."
}
```

Opcion B: login con email/password desde `.env`.

```bash
HF_EMAIL="tu-email"
HF_PASSWORD="tu-password"
HUGCHAT_COOKIE_DIR="./cookies/"
```

Importante: no subas `cookies/` a ningun repositorio.

## Limitaciones y riesgos de autenticacion

Durante las pruebas, el login con email y password no fue confiable. En la
practica, `hugchat` funciono mejor usando cookies exportadas desde una sesion
activa de Hugging Face en el navegador. Esto pasa porque `hugchat` no es una
libreria oficial y depende de flujos web de Hugging Face que pueden cambiar,
incluir OAuth, redirecciones, verificaciones adicionales o pasos interactivos
que el script no puede resolver automaticamente.

Usar cookies de sesion permite ejecutar el chatbot, pero tiene varios cuidados:

- Las cookies actuan como una sesion iniciada. Si alguien obtiene ese archivo,
  podria intentar usar la cuenta mientras la sesion siga siendo valida.
- Las cookies pueden expirar o invalidarse al cerrar sesion, cambiar password,
  activar controles de seguridad o por cambios internos de Hugging Face.
- El proyecto deja de ser completamente reproducible para otra persona, porque
  cada usuario necesita exportar sus propias cookies desde su navegador.
- No deben subirse a GitHub, compartirse por correo ni incluirse en capturas o
  entregables. Por eso `cookies/` y `.env` estan en `.gitignore`.
- Si se sospecha que las cookies fueron expuestas, lo recomendable es cerrar
  sesion en Hugging Face, revocar sesiones activas si la plataforma lo permite
  y generar cookies nuevas.

Por estas razones, este proyecto usa cookies solo como mecanismo local de
experimentacion y no como una solucion de autenticacion apropiada para una app
publica o de produccion.

## Ejecutar pruebas

```bash
uv run hugchat_prompt_lab.py --mode baseline
uv run hugchat_prompt_lab.py --mode structured
uv run hugchat_prompt_lab.py --mode web_search
uv run hugchat_prompt_lab.py --mode all
```

Para cambiar de modelo, primero revisa los modelos disponibles desde una sesion
o CLI de `hugchat`, y luego pasa el indice:

```bash
uv run hugchat_prompt_lab.py --mode role --model-index 1
```

## Que observar

- Si la respuesta mejora al agregar rol y audiencia.
- Si el formato pedido se respeta.
- Si el chatbot pide aclaraciones cuando se le solicita un enfoque socratico.
- Si la busqueda web agrega fuentes utiles o ruido.
- Si cambiar de modelo altera estilo, precision o velocidad.

## Fuentes consultadas

- Repositorio `hugchat`: https://github.com/Soulter/hugging-chat-api
- PyPI `hugchat`: https://pypi.org/project/hugchat/
- Guia de prompting de Anthropic: https://docs.anthropic.com/en/docs/prompt-engineering
- Buenas practicas de prompting de OpenAI: https://help.openai.com/en/articles/10032626-prompt-engineering-best-practices-for-chatgpt
