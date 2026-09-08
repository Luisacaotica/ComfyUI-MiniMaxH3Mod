import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { openRefModLibrary } from "./library_dialog.js";

export function updateRefModChoices(node, entries) {
    const names = ["(none)", ...new Set(entries.map(entry => entry.name))];
    for (const widget of node.widgets || []) {
        if (!/^mod_(?:[ab]_)?\d+$/.test(widget.name)) continue;
        // Preserve selections, including missing files, for an explicit load
        // error instead of silently switching the subject after a refresh.
        widget.options.values = names.includes(widget.value) || !widget.value
            ? [...names] : [...names, widget.value];
    }
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "MiniMaxH3.RefModLibrary",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!["MiniMaxH3RefModsLoader", "MiniMaxH3RefModsAxis"].includes(nodeData.name)) return;
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function(...args) {
            this._refmodRefresh?.abort();
            this._refmodDialog?.close();
            return removed?.apply(this, args);
        };
        const previous = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function(...args) {
            const result = previous?.apply(this, args);
            const refresh = this.addWidget("button", "Refresh RefMods", null, async () => {
                this._refmodRefresh?.abort();
                const controller = new AbortController();
                this._refmodRefresh = controller;
                refresh.name = "Refreshing RefMods…";
                this.setDirtyCanvas?.(true, true);
                try {
                    const response = await api.fetchApi("/refmods/library", {signal: controller.signal, cache: "no-store"});
                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                    const entries = await response.json();
                    if (controller.signal.aborted) return;
                    updateRefModChoices(this, entries);
                    refresh.name = "Refresh RefMods";
                } catch (error) {
                    if (controller.signal.aborted) return;
                    refresh.name = "Refresh failed — click to retry";
                    console.error("RefMod refresh failed:", error);
                } finally {
                    if (!controller.signal.aborted) {
                        this._refmodRefresh = null;
                        this.setDirtyCanvas?.(true, true);
                    }
                }
            }, {serialize:false});
            this.addWidget("button", "RefMod library", null, () => {
                this._refmodDialog?.close();
                this._refmodDialog = openRefModLibrary(this, async signal => {
                    const response = await api.fetchApi("/refmods/library", {signal});
                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                    const entries = await response.json();
                    if (!signal.aborted) updateRefModChoices(this, entries);
                    return entries;
                });
            }, {serialize:false});
            return result;
        };
    }
});
