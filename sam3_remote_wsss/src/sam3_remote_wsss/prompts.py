from __future__ import annotations

from .config import ClassSpec, PromptingConfig


DEFAULT_TEMPLATES = (
    "{name}",
    "a satellite image of {name}",
    "an aerial view of {name}",
    "overhead view of {name}",
    "{name} in remote sensing imagery",
)

REMOTECLIP_B2C_TEMPLATES = (
    "{name}",
    "there is {article} {name} in the center of the remote sensing image",
    "there is {article} {name} away from the center of the aerial image",
    "there are several {plural} in the remote sensing image",
    "there are many {plural} in the aerial image",
    "a lot of {plural} can be seen in the satellite image",
    "a remote sensing image of {name}",
    "an aerial image containing {name}",
    "overhead view of {plural}",
)


NAME_OVERRIDES = {
    "impervious_surface": ("impervious surface", "impervious surfaces", "an"),
    "building": ("building", "buildings", "a"),
    "low_vegetation": ("low vegetation area", "low vegetation areas", "a"),
    "tree": ("tree", "trees", "a"),
    "car": ("car", "cars", "a"),
}


def prompts_for_class(spec: ClassSpec, prompting: PromptingConfig | None = None) -> list[str]:
    style = prompting.style if prompting is not None else "manual"
    include_manual = prompting.include_manual_prompts if prompting is not None else True

    prompts: list[str] = []
    if include_manual and spec.prompts:
        prompts = list(spec.prompts)
    elif include_manual:
        prompts = [spec.name.replace("_", " ")]

    if style == "default_templates":
        canonical = spec.name.replace("_", " ")
        prompts.extend(template.format(name=canonical) for template in DEFAULT_TEMPLATES)
    elif style == "remoteclip_b2c":
        name, plural, article = _remoteclip_names(spec)
        prompts.extend(
            template.format(name=name, plural=plural, article=article)
            for template in REMOTECLIP_B2C_TEMPLATES
        )
    elif style != "manual":
        raise ValueError(f"Unknown prompting style: {style}")

    prompts = _dedupe(prompts)
    if prompting is not None and prompting.max_prompts_per_class is not None:
        prompts = prompts[: prompting.max_prompts_per_class]
    return prompts


def _remoteclip_names(spec: ClassSpec) -> tuple[str, str, str]:
    if spec.name in NAME_OVERRIDES:
        return NAME_OVERRIDES[spec.name]
    name = spec.name.replace("_", " ")
    plural = name if name.endswith("s") else f"{name}s"
    article = "an" if name[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    return name, plural, article


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out
