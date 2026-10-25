# Intermediario de vuelos (Cloudflare Workers)

Hace que los vuelos en directo funcionen también en la web publicada (https://angelvf3.github.io/novok/).
Es gratis en el plan Free de Cloudflare (100.000 peticiones al día; NOVOK hace unas 6 por minuto y pestaña abierta).

## Desplegar (una vez, unos 10 minutos)

1. Crea una cuenta gratuita en https://dash.cloudflare.com/sign-up.
2. En PowerShell, desde esta carpeta (`infra\vuelos-worker`), con Node.js instalado:
   ```powershell
   npx wrangler@3 login
   npx wrangler@3 deploy
   ```
   Al terminar te da una dirección del tipo `https://novok-vuelos.TU-CUENTA.workers.dev`.
3. Comprueba que responde: abre `https://novok-vuelos.TU-CUENTA.workers.dev/api/vuelos/adsblol/v2/point/43.5/-6/100`.
4. Pon esa dirección en `web/vuelos.json`, en `"intermediario_publico"`, y súbelo a GitHub.

## Qué hace y qué no

- Solo reenvía a adsb.lol y adsb.fi, y solo las rutas de posiciones y rutas de vuelo: no es un proxy abierto.
- Solo responde a las webs de `ORIGENES` (en `wrangler.toml`).
- Guarda las posiciones 5 s: varias personas mirando la web a la vez hacen una sola petición al proveedor.
- Antes de dar servicio a un cliente, quita adsb.fi (solo uso no comercial) y usa un proveedor con licencia comercial.
