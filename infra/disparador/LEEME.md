# Disparador de NOVOK (Cloudflare Worker)

Los horarios (`schedule`) de GitHub Actions no son fiables: en noviembre de 2026 la web pasó de 5 a 8 horas sin
actualizarse aunque el workflow está programado cada 30 minutos. Este Worker lanza los workflows `publicar.yml` y
`grabar_taf.yml` cada 30 minutos llamando a la API de GitHub. Cron Triggers de Cloudflare es gratis.

## Puesta en marcha (una vez)

1. En GitHub: Settings (de tu cuenta) → Developer settings → Personal access tokens → Fine-grained tokens →
   Generate new token.
   - Repository access: Only select repositories → `angelvf3/novok`.
   - Permissions → Repository permissions → **Actions: Read and write**. Nada más.
   - Caducidad: 1 año (apúntate cuándo renovarlo).
2. En PowerShell, desde esta carpeta:
   ```powershell
   cd infra\disparador
   npx wrangler deploy
   npx wrangler secret put GITHUB_TOKEN     # pega el token cuando lo pida
   ```
3. Comprueba en la pestaña Actions de GitHub que en la próxima media hora aparecen ejecuciones
   "workflow_dispatch" de "Publicar briefing" y "Grabar TAF de la red".

El token solo permite lanzar workflows de este repositorio. Se guarda cifrado en Cloudflare, nunca en el código.
Los `schedule` de los workflows se quedan como respaldo; si coinciden con el disparador, el `concurrency` de
`publicar.yml` cancela la ejecución repetida.
