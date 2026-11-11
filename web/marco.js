/* NOVOK · marco común de las páginas (barra lateral, relojes, estado de los datos y "Mi aerolínea").
   Panel de red y aeropuerto ya pintan su propio marco; ahí solo se usa "Mi aerolínea" (body sin data-marco).
   Las páginas sencillas (desvíos, guía, contacto) llevan <body data-marco="completo"> y este fichero lo hace todo. */
(() => {
  const ZONA = "Europe/Madrid";
  const $ = id => document.getElementById(id);
  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const CAT = { VFR: "#2e8540", MVFR: "#0071bc", IFR: "#d83933", LIFR: "#b5179e" };
  const CLAVE = "novok-aerolinea";
  const leer = () => { try { return localStorage.getItem(CLAVE) || ""; } catch { return ""; } };
  const guardar = v => { try { localStorage.setItem(CLAVE, v); } catch { /* sin almacenamiento */ } };

  /* ---------- Mi aerolínea: se elige una vez y la usan todas las páginas ---------- */
  const oyentes = [];
  const marco = window.NovokMarco = {
    aerolinea: leer,
    nombres: {},
    alCambiar(fn) { oyentes.push(fn); },
    fijar(v, avisar = true) {                    // el panel de red lo llama cuando se cambia el filtro del mapa
      guardar(v);
      const sel = $("mi-aerolinea");
      if (sel && sel.value !== v) {
        if (v && ![...sel.options].some(o => o.value === v)) sel.add(new Option(marco.nombres[v] || v, v));
        sel.value = v;
      }
      if (avisar) oyentes.forEach(fn => fn(v));
    },
  };
  const sel = $("mi-aerolinea");
  if (sel) {
    fetch("aerolineas.json", { cache: "no-cache" }).then(r => r.ok ? r.json() : null).then(j => {
      marco.nombres = j?.aerolineas || {};
      const actual = leer();
      const codigos = Object.keys(marco.nombres).sort((x, y) => marco.nombres[x].localeCompare(marco.nombres[y], "es"));
      if (actual && !codigos.includes(actual)) codigos.unshift(actual);
      sel.innerHTML = `<option value="">Todas las aerolíneas</option>` +
        codigos.map(c => `<option value="${esc(c)}">${esc(marco.nombres[c] || c)}</option>`).join("");
      sel.value = actual;
    }).catch(() => {});
    sel.addEventListener("change", () => marco.fijar(sel.value));
  }

  if (document.body.dataset.marco !== "completo") return;

  /* ---------- relojes con segundos ---------- */
  const reloj = () => {
    const d = new Date(), hm = zona => d.toLocaleTimeString("es-ES", { timeZone: zona, hour: "2-digit", minute: "2-digit", second: "2-digit" });
    if ($("reloj-l")) $("reloj-l").textContent = hm(ZONA);
    if ($("reloj-u")) $("reloj-u").textContent = hm("UTC");
  };
  reloj(); setInterval(reloj, 1000);

  /* ---------- menú lateral en el móvil ---------- */
  const lateral = abrir => {
    document.body.classList.toggle("lat-abierto", abrir);
    $("velo").hidden = !abrir;
    $("abrir-lateral").setAttribute("aria-expanded", abrir);
    if (abrir) $("cerrar-lateral").focus();
  };
  $("abrir-lateral")?.addEventListener("click", () => lateral(true));
  $("cerrar-lateral")?.addEventListener("click", () => lateral(false));
  $("velo")?.addEventListener("click", () => lateral(false));
  $("lateral")?.addEventListener("click", e => { if (e.target.closest("a")) lateral(false); });
  document.addEventListener("keydown", e => { if (e.key === "Escape" && document.body.classList.contains("lat-abierto")) lateral(false); });

  /* ---------- aeropuertos de la red y estado de los datos ---------- */
  const fmt = iso => new Date(iso).toLocaleTimeString("es-ES", { timeZone: ZONA, hour: "2-digit", minute: "2-digit" });
  fetch("datos/red.json", { cache: "no-store" }).then(r => r.ok ? r.json() : null).then(red => {
    if (!red) return;
    const l = [...red.aeropuertos].sort((x, y) => (y.riesgo?.p ?? -1) - (x.riesgo?.p ?? -1) || x.icao.localeCompare(y.icao));
    $("lateral-red").innerHTML = l.map(a => `<li><a href="aeropuerto.html?a=${encodeURIComponent(a.icao)}">
      <i class="p-dot" style="background:${CAT[a.categoria] || "#a9aeb1"}"></i><b class="mono">${esc(a.icao)}</b><span>${esc(a.nombre)}</span>
      <em class="${a.riesgo?.nivel || ""}">${a.riesgo ? Math.round(a.riesgo.p * 100) + " %" : "—"}</em></a></li>`).join("");
    const min = Math.round((Date.now() - Date.parse(red.generado_utc)) / 60000), ec = $("estado-corto");
    ec.innerHTML = min > 75 ? `<span class="punto mal"></span>Datos con retraso · ${min < 120 ? min + " min" : Math.round(min / 60) + " h"}`
      : `<span class="punto"></span>En directo · ${fmt(red.generado_utc)}`;
    ec.classList.toggle("retraso", min > 75);
  }).catch(() => {});
})();
