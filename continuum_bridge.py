"""Model-scoped RefMod injection before ComfyUI prepares each sampling run.

Works with Continuum and standard H3 samplers through the public model patcher
OUTER_SAMPLE hook. No process-global bundle or third-party function patch.
"""

from comfy.patcher_extension import WrappersMP

BRIDGE_KEY = "minimax_h3_refmod_bridge"


def attach(model, blocks, enabled=True):
    model = model.clone()
    model.remove_wrappers_with_key(WrappersMP.OUTER_SAMPLE, BRIDGE_KEY)
    if not enabled or not blocks:
        return model
    refs = tuple(dict(b) for b in blocks)

    def inject(executor, *args, **kwargs):
        guider = executor.class_obj
        original = guider.conds
        guider.conds = {
            key: [dict(entry, minimax_refs=list(entry.get("minimax_refs") or [])
                       + [dict(ref) for ref in refs]) for entry in entries]
            for key, entries in original.items()
        }
        try:
            return executor(*args, **kwargs)
        finally:
            guider.conds = original

    model.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, BRIDGE_KEY, inject)
    return model


def disarm():
    # Preserve the old output-node ID in saved workflows. There is no longer
    # global state to disarm; enable=False on the model line removes the hook.
    return "Bridge is model-scoped. Bypass it or set enable=False on the model line."
