import { app } from "../../scripts/app.js";

const SLOT = /^(?:mod_(?:[ab]_)?|strength_|copies_|value_|components_|visual_strength_|audio_strength_)(\d+)$/;
const NONE = new Set(["", "None", "(none)"]);

export function installRefModSlots(nodeType) {
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function(...args) {
        const result = created?.apply(this, args);
        setupSlots(this);
        return result;
    };
    // LiteGraph stores widget values by position. Keep the original schema
    // order for saving/loading even though the canvas groups them by slot.
    for (const method of ["serialize", "configure"]) {
        const original = nodeType.prototype[method];
        nodeType.prototype[method] = function(...args) {
            const state = this._refmodSlots;
            if (!state) return original?.apply(this, args);
            const displayOrder = this.widgets;
            this.widgets = [...state.originals, ...displayOrder.filter(w => !state.originals.includes(w))];
            try {
                return original?.apply(this, args);
            } finally {
                this.widgets = displayOrder;
                if (method === "configure") state.sync();
            }
        };
    }
    const connections = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function(...args) {
        const result = connections?.apply(this, args);
        this._refmodSlots?.sync();
        return result;
    };
}

function setupSlots(node) {
    if (node._refmodSlots) return;
    const originals = [...(node.widgets || [])];
    const groups = new Map();
    for (const widget of originals) {
        const match = widget.name.match(SLOT);
        if (!match) continue;
        const slot = Number(match[1]);
        if (!groups.has(slot)) groups.set(slot, []);
        groups.get(slot).push(widget);
    }
    const slots = [...groups.keys()].sort((a, b) => a - b);
    if (!slots.length) return;
    const defaults = new Map(originals.map(w => [w, w.value]));
    const hidden = new Map();
    const removers = new Map();
    node.properties ??= {};
    node.properties.refmod_visible_slots ??= [slots[0]];
    const linked = slot => node.inputs?.some(input => Number(input.name.match(SLOT)?.[1]) === slot && input.link != null);
    const used = slot => groups.get(slot).some(w => w.name.startsWith("mod_")
        ? w.value != null && !NONE.has(String(w.value).trim())
        : w.value !== defaults.get(w));

    function show(widget, visible) {
        if (!visible && !hidden.has(widget)) {
            hidden.set(widget, {type: widget.type, computeSize: widget.computeSize,
                draw: widget.draw, mouse: widget.mouse, hidden: widget.hidden});
            widget.type = "converted-widget";
            widget.computeSize = () => [0, -4];
            widget.draw = () => {};
            widget.mouse = () => false;
            widget.hidden = true;
        } else if (visible && hidden.has(widget)) {
            const saved = hidden.get(widget);
            // Existing converted widgets keep their own socket-only layout.
            if (widget.type === "converted-widget") widget.type = saved.type;
            widget.computeSize = saved.computeSize;
            widget.draw = saved.draw;
            widget.mouse = saved.mouse;
            widget.hidden = saved.hidden;
            hidden.delete(widget);
        }
    }

    function sync() {
        const requested = node.properties.refmod_visible_slots;
        const visible = new Set((Array.isArray(requested) ? requested : [slots[0]])
            .filter(slot => groups.has(slot)));
        for (const slot of slots) if (used(slot) || linked(slot)) visible.add(slot);
        node.properties.refmod_visible_slots = [...visible].sort((a, b) => a - b);
        const grouped = [...groups.values()].flat();
        const extras = node.widgets.filter(w => !originals.includes(w) && ![...removers.values(), add].includes(w));
        node.widgets = [];
        for (const slot of slots) {
            for (const widget of groups.get(slot)) {
                show(widget, visible.has(slot));
                node.widgets.push(widget);
            }
            const remove = removers.get(slot);
            remove.disabled = !!linked(slot);
            show(remove, visible.has(slot));
            node.widgets.push(remove);
        }
        show(add, visible.size < slots.length);
        node.widgets.push(add, ...originals.filter(w => !grouped.includes(w)), ...extras);
        node.setSize?.([node.size[0], node.computeSize()[1]]);
        node.setDirtyCanvas?.(true, true);
    }

    for (const slot of slots) {
        removers.set(slot, node.addWidget("button", `Remove RefMod ${slot}`, null, () => {
            if (linked(slot)) return;
            for (const widget of groups.get(slot)) widget.value = defaults.get(widget);
            node.properties.refmod_visible_slots = node.properties.refmod_visible_slots.filter(i => i !== slot);
            sync();
        }, {serialize: false}));
        for (const widget of groups.get(slot)) {
            const callback = widget.callback;
            widget.callback = function(...args) {
                const result = callback?.apply(this, args);
                sync();
                return result;
            };
        }
    }
    const add = node.addWidget("button", "+ Add RefMod", null, () => {
        const next = slots.find(slot => !node.properties.refmod_visible_slots.includes(slot));
        if (next != null) node.properties.refmod_visible_slots.push(next);
        sync();
    }, {serialize: false});
    node._refmodSlots = {originals, sync};
    sync();
}

app.registerExtension({
    name: "MiniMaxH3.AdditiveSlots",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (["MiniMaxH3RefModsLoader", "MiniMaxH3RefModsAxis"].includes(nodeData.name)) {
            installRefModSlots(nodeType);
        }
    },
});
