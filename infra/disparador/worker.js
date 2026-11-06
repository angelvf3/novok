// Disparador de NOVOK: Cloudflare Workers ejecuta esto cada 30 minutos (Cron Triggers, gratis) y pide a GitHub
// que lance los workflows. Hace falta porque los "schedule" de GitHub Actions se retrasan o se saltan horas enteras.
const WORKFLOWS = ["publicar.yml", "grabar_taf.yml"];

async function disparar(env) {
  const resultados = [];
  for (const wf of WORKFLOWS) {
    const r = await fetch(`https://api.github.com/repos/${env.REPO}/actions/workflows/${wf}/dispatches`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${env.GITHUB_TOKEN}`,
        Accept: "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "novok-disparador",
      },
      body: JSON.stringify({ ref: "main" }),
    });
    resultados.push(`${wf}: ${r.status}`);       // 204 = lanzado
  }
  return resultados;
}

export default {
  async scheduled(evento, env, ctx) {
    ctx.waitUntil(disparar(env).then(r => console.log(r.join(", "))));
  },
  // Abrir la URL del Worker no lanza nada (para que nadie pueda dispararlo desde fuera): solo dice que está vivo
  async fetch() {
    return new Response("Disparador de NOVOK activo: lanza los workflows cada 30 minutos.\n", { headers: { "content-type": "text/plain; charset=utf-8" } });
  },
};
