export function openRefModLibrary(node, fetchEntries) {
    const previousFocus = document.activeElement;
    const dialog = document.createElement("dialog");
    dialog.className = "refmod-library";
    dialog.setAttribute("aria-label", "RefMod library");
    const style = document.createElement("style");
    style.textContent = `
      .refmod-library {box-sizing:border-box;width:min(1100px,96vw);height:90vh;max-height:90vh;padding:24px;border:1px solid #455266;border-radius:14px;background:#111820;color:#e8edf5;font:14px system-ui;}
      .refmod-library::backdrop {background:#000b}
      .refmod-library .layout {height:100%;display:flex;flex-direction:column;gap:16px;min-height:0}
      .refmod-library header {display:flex;gap:12px;align-items:center;justify-content:space-between}
      .refmod-library h2 {margin:0;font-size:23px}
      .refmod-library .controls {display:flex;gap:10px;flex-wrap:wrap}
      .refmod-library input,.refmod-library select,.refmod-library button {font:inherit;color:inherit;background:#222f3e;border:1px solid #455266;border-radius:7px;padding:9px 12px}
      .refmod-library input {flex:1;min-width:180px}
      .refmod-library button {cursor:pointer}
      .refmod-library button:focus-visible,.refmod-library input:focus-visible,.refmod-library select:focus-visible {outline:2px solid #ffb257;outline-offset:2px}
      .refmod-library .results {overflow:auto;min-height:0;flex:1;display:flex;flex-direction:column;gap:10px}
      .refmod-library article {padding:16px;border:1px solid #334254;border-radius:9px;background:#18222d;display:grid;grid-template-columns:1fr auto;gap:8px 18px}
      .refmod-library .description,.refmod-library .path {grid-column:1/-1;color:#afbed0;overflow-wrap:anywhere}
      .refmod-library .path {font-size:12px}
      .refmod-library .use {background:#ffb257;border-color:#ffb257;color:#171c22;font-weight:650}
      .refmod-library .status {color:#afbed0;min-height:20px}
      @media(max-width:600px){.refmod-library{padding:14px}.refmod-library article{grid-template-columns:1fr}.refmod-library header{flex-wrap:wrap}}
    `;
    const make = (tag, text, className) => {
        const el = document.createElement(tag);
        if (text !== undefined) el.textContent = text;
        if (className) el.className = className;
        return el;
    };
    const layout = make("div", undefined, "layout");
    const header = make("header");
    const close = make("button", "Close");
    header.append(make("h2", "RefMod library"), close);
    const controls = make("div", undefined, "controls");
    const search = make("input");
    search.placeholder = "Search name, folder or description";
    search.setAttribute("aria-label", "Search RefMods");
    const kind = make("select");
    kind.setAttribute("aria-label", "Reference type");
    for (const value of ["all", "image", "video", "audio", "character"]) {
        const option = make("option", value === "all" ? "All reference types" : value);
        option.value = value;
        kind.append(option);
    }
    const slot = make("select");
    slot.setAttribute("aria-label", "Loader slot");
    const widgets = (node.widgets || []).filter(w => /^mod_(?:[ab]_)?\d+$/.test(w.name));
    for (const widget of widgets) {
        const option = make("option", widget.name);
        option.value = widget.name;
        slot.append(option);
    }
    const refresh = make("button", "Refresh");
    controls.append(search, kind, slot, refresh);
    const status = make("div", "Loading library…", "status");
    status.setAttribute("role", "status");
    const results = make("div", undefined, "results");
    layout.append(header, controls, status, results);
    dialog.append(style, layout);
    document.body.append(dialog);
    let entries = [], disposed = false, controller;
    const cleanup = () => {
        if (disposed) return;
        disposed = true;
        controller?.abort();
        dialog.remove();
        previousFocus?.focus();
    };
    close.onclick = () => dialog.close();
    dialog.addEventListener("close", cleanup, {once:true});
    const render = () => {
        const query = search.value.trim().toLocaleLowerCase();
        const filtered = entries.filter(e => (kind.value === "all" || e.kind === kind.value)
            && [e.name,e.concept,e.description].join(" ").toLocaleLowerCase().includes(query));
        status.textContent = `${filtered.length} of ${entries.length} RefMods · choose a loader slot, then Use`;
        results.replaceChildren();
        for (const entry of filtered) {
            const row = make("article");
            const title = make("strong", entry.name);
            const use = make("button", "Use", "use");
            use.disabled = widgets.length === 0;
            use.onclick = () => {
                const widget = widgets.find(w => w.name === slot.value);
                const linked = node.inputs?.find(i => i.name === slot.value)?.link;
                if (linked != null) {
                    status.textContent = `${slot.value} is connected. Select an unconnected slot.`;
                    return;
                }
                if (!widget) return;
                widget.value = entry.name;
                widget.callback?.(entry.name);
                node.setDirtyCanvas?.(true, true);
                dialog.close();
            };
            const tokens = entry.tokens == null ? "tokens available after loading" : `${entry.tokens.toLocaleString()} tokens / copy`;
            row.append(title, use, make("div", `${entry.kind} · ${entry.concept} · ${tokens}`, "description"),
                make("div", entry.description || "No description saved", "description"),
                make("div", entry.path, "path"));
            results.append(row);
        }
        if (!filtered.length) results.append(make("p", "No matching RefMods. Try another search or Refresh after adding files."));
    };
    const load = async () => {
        controller?.abort();
        controller = new AbortController();
        const signal = controller.signal;
        refresh.disabled = true;
        try {
            const data = await fetchEntries(signal);
            if (disposed || signal.aborted) return;
            entries = data;
            render();
        } catch (error) {
            if (!disposed && !signal.aborted) status.textContent = `Could not load library: ${error.message}`;
        } finally {
            if (!disposed && !signal.aborted) refresh.disabled = false;
        }
    };
    search.oninput = render;
    kind.onchange = render;
    refresh.onclick = load;
    dialog.showModal();
    search.focus();
    load();
    return dialog;
}
