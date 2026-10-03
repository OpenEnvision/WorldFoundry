"""Reversible model-level guards for fixed derived inference weights."""


def guard_fixed_inference(model, name, *, forbid_serialization=False):
    attributes = ["_apply", "train", "load_state_dict"]
    if forbid_serialization:
        attributes.append("state_dict")
    originals = {key: model.__dict__[key] for key in attributes if key in model.__dict__}
    original_train = model.train

    def reject(*args, **kwargs):
        raise RuntimeError(f"uninstall {name} before changing placement or loading/serializing checkpoints")

    def train(mode=True):
        if mode is True:
            raise RuntimeError(f"uninstall {name} before training")
        return original_train(mode)

    def undo():
        for key in attributes:
            if key in originals:
                setattr(model, key, originals[key])
            else:
                model.__dict__.pop(key, None)

    try:
        for key in attributes:
            setattr(model, key, train if key == "train" else reject)
    except BaseException:
        undo()
        raise
    return undo
