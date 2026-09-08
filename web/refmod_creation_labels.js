import { app } from "../../scripts/app.js";

const modes = { training: "Compressed Reference", encode: "Full Reference" };

function updateLabels(node) {
    const mode = node.widgets?.find(widget => widget.name === "mode");
    if (mode) {
        mode.options.values = Object.values(modes);
        mode.value = modes[mode.value] ?? mode.value;
    }
    const steps = node.widgets?.find(widget => widget.name === "identity");
    if (steps) steps.label = "Refinement Steps";
}

app.registerExtension({
    name: "MiniMaxH3.CreationLabels",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!["MiniMaxH3RefModExtract", "MiniMaxH3RefModMasterExtract"].includes(nodeData.name)) return;
        for (const hook of ["onNodeCreated", "onConfigure"]) {
            const previous = nodeType.prototype[hook];
            nodeType.prototype[hook] = function(...args) {
                const result = previous?.apply(this, args);
                updateLabels(this);
                return result;
            };
        }
    }
});
