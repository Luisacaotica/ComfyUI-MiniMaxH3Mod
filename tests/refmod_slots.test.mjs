import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const source = readFileSync(new URL('../web/refmod_slots.js', import.meta.url), 'utf8')
    .replace('import { app } from "../../scripts/app.js";', 'const app = {registerExtension() {}};');
const {installRefModSlots} = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));

function fixture(axis = false, stableView = false) {
    class Node {
        constructor() {
            if (stableView) {
                // Frontend v1.53.6 widgetsView.ts mutates a stable array on assignment.
                const view = [];
                Object.defineProperty(this, 'widgets', {
                    get: () => view,
                    set: widgets => view.splice(0, view.length, ...widgets),
                });
            }
            this.widgets = []; this.properties = {}; this.inputs = []; this.size = [360, 1400];
            this.addWidget('toggle', 'show_info', false);
            for (let i=1;i<=8;i++) {
                if (axis) {
                    this.addWidget('combo',`mod_a_${i}`,'(none)');
                    this.addWidget('combo',`mod_b_${i}`,'(none)');
                    this.addWidget('number',`value_${i}`,0);
                } else {
                    this.addWidget('combo',`mod_${i}`,'(none)');
                    this.addWidget('number',`strength_${i}`,1);
                    this.addWidget('number',`copies_${i}`,1);
                }
            }
            this.addWidget('number','max_total_tokens',0);
            for (let i=1;i<=8;i++) {
                this.addWidget('combo',`components_${i}`,'All');
                this.addWidget('number',`visual_strength_${i}`,1);
                this.addWidget('number',`audio_strength_${i}`,1);
            }
            this.onNodeCreated();
        }
        addWidget(type,name,value,callback,options={}) {
            const widget={type,name,value,callback,options}; this.widgets.push(widget); return widget;
        }
        computeSize() { return [360, 60 + this.widgets.reduce((h,w)=>h+(w.computeSize?.()[1] ?? 20)+4,0)]; }
        setSize(size) { this.size=size; }
        setDirtyCanvas() {}
        serialize() { return {properties:structuredClone(this.properties),widgets_values:this.widgets.map(w=>w.options.serialize===false?null:w.value)}; }
        // PR #20: modern frontend saves directly, without calling serialize().
        serializeFromStoreState() {
            return {properties:structuredClone(this.properties),widgets_values:this.widgets.map(w=>w.options.serialize===false?null:w.value)};
        }
        configure(info) {
            this.properties = structuredClone(info.properties ?? {});
            // LiteGraph invokes these even for unconnected input sockets,
            // before restoring widgets_values (frontend 1.39.2 and 1.52.7).
            this.inputs=info.inputs ?? [{name:'show_info',link:null}];
            for (const [i,input] of this.inputs.entries()) {
                this.onConnectionsChange?.(1,i,true,null,input);
            }
            info.widgets_values.forEach((value,i)=>{if(this.widgets[i])this.widgets[i].value=value;});
        }
    }
    installRefModSlots(Node);
    return Node;
}
const get=(node,name)=>node.widgets.find(w=>w.name===name);
const click=(node,name)=>get(node,name).callback();
for (const stableView of [false,true]) for (const axis of [false,true]) {
    const Node=fixture(axis, stableView), node=new Node();
    const mod=axis?'mod_a_':'mod_';
    const initial=node.size[1];
    assert.deepEqual(node.properties.refmod_visible_slots,[1]);
    assert.equal(get(node,mod+'8').type,'combo');
    assert.equal(get(node,mod+'8').options.hidden,true);
    // Nodes 2.0 renders converted-widget socket rows even when hidden. Real
    // widget types plus options.hidden eliminate those blank rows entirely.
    const rendered = () => node.widgets.filter(w => w.type === 'converted-widget' || !w.options.hidden).map(w => w.name);
    assert.ok(!rendered().includes(mod+'8'));
    assert.deepEqual(rendered().slice(0,6), axis
        ? ['mod_a_1','mod_b_1','value_1','components_1','visual_strength_1','audio_strength_1']
        : ['mod_1','strength_1','copies_1','components_1','visual_strength_1','audio_strength_1']);
    click(node,'+ Add RefMod');
    assert.deepEqual(node.properties.refmod_visible_slots,[1,2]);
    assert.ok(node.size[1]>initial);
    get(node,mod+'2').value='folder/hero';
    get(node,'audio_strength_2').value=.25;
    const saved=node.serialize();
    assert.equal(saved.widgets_values[4],'folder/hero');
    const displayOrder = [...node.widgets];
    node.serialize = () => { throw new Error('store serializer must be independent'); };
    assert.deepEqual(node.serializeFromStoreState(), saved);
    assert.deepEqual(node.widgets, displayOrder);
    assert.equal(node._refmodSlots.schemaOrderDepth, 0);
    // v0.2.7 wrote both snapshots from the slot-grouped display order.
    const oldSave = {
        properties:structuredClone(node.properties),
        widgets_values:node.widgets.filter(w=>w.serialize!==false).map(w=>w.options.serialize===false?null:w.value),
        widgets_values_named:Object.fromEntries(node.widgets.filter(w=>w.serialize!==false)
            .map(w=>[w.name,w.options.serialize===false?null:w.value])),
    };
    const oldSnapshot = structuredClone(oldSave);
    Object.freeze(oldSave.widgets_values);
    Object.freeze(oldSave.widgets_values_named);
    Object.freeze(oldSave);
    const repaired = new Node(); repaired.configure(oldSave);
    assert.equal(get(repaired,mod+'2').value,'folder/hero');
    assert.equal(get(repaired,'audio_strength_2').value,.25);
    assert.deepEqual(repaired.serializeFromStoreState().widgets_values,saved.widgets_values);
    assert.deepEqual(repaired.properties.refmod_visible_slots,[1,2]);
    assert.deepEqual(oldSave,oldSnapshot);
    // Current saves and partial named maps retain schema-position fallbacks.
    const namedSave = {...saved,widgets_values_named:oldSave.widgets_values_named};
    const namedRestored = new Node(); namedRestored.configure(namedSave);
    assert.deepEqual(namedRestored.serializeFromStoreState().widgets_values,saved.widgets_values);
    const partial = {...oldSave,widgets_values:[...saved.widgets_values],
        widgets_values_named:{...oldSave.widgets_values_named}};
    delete partial.widgets_values_named.audio_strength_2;
    const audioIndex = node._refmodSlots.originals.findIndex(w=>w.name==='audio_strength_2');
    partial.widgets_values[audioIndex]=.625;
    const partialSnapshot = structuredClone(partial);
    const partialRestored = new Node(); partialRestored.configure(partial);
    assert.equal(get(partialRestored,mod+'2').value,'folder/hero');
    assert.equal(get(partialRestored,'audio_strength_2').value,.625);
    assert.deepEqual(partial,partialSnapshot);
    for (const named of [null, false, 'invalid', []]) {
        const positional = new Node(); positional.configure({...saved,widgets_values_named:named});
        assert.deepEqual(positional.serializeFromStoreState().widgets_values,saved.widgets_values);
    }
    // Vue/store snapshots may restore visible rows as an indexed object.
    const fromStore = new Node();
    fromStore.configure({...saved, properties:{refmod_visible_slots:{0:1,1:2,2:4}}});
    assert.deepEqual(fromStore.properties.refmod_visible_slots,[1,2,4]);
    assert.equal(get(fromStore,mod+'2').value,'folder/hero');
    assert.equal(get(fromStore,'audio_strength_2').value,.25);
    assert.deepEqual(fromStore.serializeFromStoreState().widgets_values,saved.widgets_values);
    const restored=new Node(); restored.configure(saved);
    assert.equal(get(restored,mod+'2').value,'folder/hero');
    assert.equal(get(restored,'audio_strength_2').value,.25);
    assert.deepEqual(restored.serialize().widgets_values,saved.widgets_values);
    assert.equal(restored.widgets.indexOf(get(restored,'audio_strength_1')),5);
    // Repeated tab changes must keep all schema values and visible rows stable.
    let current = restored;
    for (let i=0;i<5;i++) {
        const snapshot = current.serialize();
        current = new Node();
        current.configure(snapshot);
        assert.deepEqual(current.serialize(), snapshot);
    }
    // A saved connected input also fires callbacks before value restoration.
    const connected = new Node();
    connected.configure({...saved,inputs:[{name:mod+'8',link:99}]});
    assert.equal(get(connected,mod+'2').value,'folder/hero');
    assert.equal(get(connected,'audio_strength_2').value,.25);
    assert.ok(connected.properties.refmod_visible_slots.includes(8));
    // Connected rows cannot be removed, even when their widget has no filename.
    restored.inputs=[{name:mod+'8',link:99}]; restored.onConnectionsChange();
    click(restored,'Remove RefMod 8');
    assert.ok(restored.properties.refmod_visible_slots.includes(8));
    assert.equal(restored.inputs[0].link,99);
    click(restored,'Remove RefMod 2');
    assert.equal(get(restored,mod+'2').value,'(none)');
    assert.equal(get(restored,'audio_strength_2').value,1);
    assert.ok(!restored.properties.refmod_visible_slots.includes(2));
    // Legacy positional workflow, including a high selected slot and no UI properties.
    const legacy=new Node().serialize(); delete legacy.properties;
    legacy.widgets_values=legacy.widgets_values.slice(0,26);
    legacy.widgets_values[22]='legacy/eighth';
    const migrated=new Node(); migrated.configure(legacy);
    assert.equal(get(migrated,mod+'8').value,'legacy/eighth');
    // Widgets absent from both maps keep their defaults instead of undefined.
    const sparse=new Node(); sparse.configure({...legacy,widgets_values_named:{[mod+'8']:'legacy/eighth'}});
    assert.equal(get(sparse,mod+'8').value,'legacy/eighth');
    assert.equal(get(sparse,'audio_strength_8').value,1);
    assert.ok(migrated.properties.refmod_visible_slots.includes(8));
    assert.equal(get(migrated,'audio_strength_8').value,1);
    // Library selection in a hidden slot reveals it via its existing callback.
    get(migrated,mod+'5').value='library/pick'; get(migrated,mod+'5').callback();
    assert.ok(migrated.properties.refmod_visible_slots.includes(5));
    while(migrated.properties.refmod_visible_slots.length<8) click(migrated,'+ Add RefMod');
    assert.equal(get(migrated,'+ Add RefMod').options.hidden,true);
}
// Frozen pre-library schema values: restoration must not depend on a snapshot
// produced by the current implementation under test.
const legacyFixture = JSON.parse(readFileSync(new URL('./fixtures/loader_widgets.json',import.meta.url),'utf8'));
const FixtureNode = fixture();
const fixtureNode = new FixtureNode();
fixtureNode.configure(legacyFixture);
assert.equal(get(fixtureNode,'mod_1').value,'characters/alice');
assert.equal(get(fixtureNode,'strength_1').value,.73);
assert.equal(get(fixtureNode,'copies_1').value,2);
assert.equal(get(fixtureNode,'mod_8').value,'motion/walk');
assert.equal(get(fixtureNode,'components_1').value,'Visual');
assert.equal(get(fixtureNode,'visual_strength_1').value,.85);
assert.equal(get(fixtureNode,'audio_strength_8').value,.4);
assert.equal(get(fixtureNode,'max_total_tokens').value,8192);
assert.deepEqual(fixtureNode.properties.refmod_visible_slots,[1,8]);
assert.deepEqual(fixtureNode.serializeFromStoreState().widgets_values.slice(0,50),legacyFixture.widgets_values);
console.log('PASS: Loader + Axis, legacy/store serialization, indexed visible slots, frozen fixture, linked slots and library selection');
