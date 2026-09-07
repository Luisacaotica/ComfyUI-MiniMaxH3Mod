import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { openRefModLibrary } from "./library_dialog.js";

app.registerExtension({
    name: "MiniMaxH3.RefModLibrary",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!["MiniMaxH3RefModsLoader", "MiniMaxH3RefModsAxis"].includes(nodeData.name)) return;
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function(...args) {
            this._refmodDialog?.close();
            return removed?.apply(this, args);
        };
        const previous = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function(...args) {
            const result = previous?.apply(this, args);
            this.addWidget("button", "RefMod library", null, () => {
                this._refmodDialog?.close();
                this._refmodDialog = openRefModLibrary(this, async signal => {
                    const response = await api.fetchApi("/refmods/library", {signal});
                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                    return response.json();
                });
            }, {serialize:false});
            return result;
        };
    }
});
