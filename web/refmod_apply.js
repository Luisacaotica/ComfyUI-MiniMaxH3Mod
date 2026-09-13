import { app } from "../../scripts/app.js";

export function installRefModApply(nodeType) {
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function(...args) {
        const result = created?.apply(this, args);
        // MatchType installs its callback on the instance during construction.
        const connections = this.onConnectionsChange;
        this.onConnectionsChange = function(...args) {
            const index = this.inputs?.findIndex(input => input.name === "conditioning") ?? -1;
            if (index >= 0) {
                const input = this.inputs[index];
                try {
                    // Frontend 1.39.2 proxies MatchType slots, but NodeSlot.node
                    // reads a #private field. That throws during canvas drawing.
                    void input.node;
                } catch (error) {
                    if (!(error instanceof TypeError)) throw error;
                    this.inputs[index] = new input.constructor({...input}, this);
                    if (args[4] === input) args[4] = this.inputs[index];
                }
            }
            return connections?.apply(this, args);
        };
        return result;
    };
}

app.registerExtension({
    name: "MiniMaxH3.ApplyCanvasCompatibility",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "MiniMaxH3RefModApply") installRefModApply(nodeType);
    },
});
