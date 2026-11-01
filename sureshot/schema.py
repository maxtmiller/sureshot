"""Questions from a JSON schema, and the router's answers back as an object matching it, so sureshot can answer the
structured-output requests of other APIs (see formats.py)."""
import re

# The Anthropic SDK moves constraints its API doesn't take into the description, e.g. "Priority\n\n{maximum: 5, minimum: 1}".
_MOVED = re.compile(r"\{([^{}]*)\}\s*$")


def _resolve(node, root):
    """node with a local $ref (#/$defs/... or #/definitions/...) or a one-item allOf merged in, or None if it can't be."""
    while isinstance(node, dict) and ("$ref" in node or "allOf" in node):
        if "allOf" in node:
            if not (isinstance(node["allOf"], list) and len(node["allOf"]) == 1):
                return None
            target, rest = node["allOf"][0], {k: v for k, v in node.items() if k != "allOf"}
        else:
            if not node["$ref"].startswith("#/"):
                return None
            target, rest = root, {k: v for k, v in node.items() if k != "$ref"}
            for part in node["$ref"][2:].split("/"):
                target = target.get(part) if isinstance(target, dict) else None
        if not isinstance(target, dict):
            return None
        node = target | rest
    return node if isinstance(node, dict) else None


def _integers(values):
    return all(isinstance(v, int) and not isinstance(v, bool) for v in values)


def _integer(value):
    """value as an int when it is a whole number (Gemini's SDK sends bounds as 5.0), else None."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value if _integers([value]) else None


def _bounds(prop):
    low, high = _integer(prop.get("minimum")), _integer(prop.get("maximum"))
    moved = _MOVED.search(prop.get("description") or "")
    if (low is None or high is None) and moved:
        found = dict(part.split(":", 1) for part in moved.group(1).split(",") if ":" in part)
        found = {k.strip(): v.strip() for k, v in found.items()}
        if re.fullmatch(r"-?\d+", found.get("minimum", "")) and re.fullmatch(r"-?\d+", found.get("maximum", "")):
            low, high = int(found["minimum"]), int(found["maximum"])
    return low, high


def _question(prop, root):
    """(question, values) for one property, values being the JSON value of each score level, or None if it doesn't map."""
    prop = _resolve(prop, root)
    if prop is None:
        return None
    instructions = prop.get("description") or prop.get("title")
    base = {"instructions": instructions} if instructions else {}
    kind, enum = prop.get("type"), prop.get("enum")
    if isinstance(kind, str):
        kind = kind.lower()  # Gemini's responseSchema is OpenAPI, with types such as "STRING"
    if enum is not None:
        if not (isinstance(enum, list) and enum and len(set(map(repr, enum))) == len(enum)):
            return None
        if kind in (None, "string") and all(isinstance(v, str) for v in enum) and len(enum) <= 255:
            return {"type": "choice", **base, "criteria": {v: "" for v in enum}}, None
        if kind in (None, "integer") and _integers(enum) and 2 <= len(enum) <= 10:
            return {"type": "score", **base, "criteria": [str(v) for v in enum]}, list(enum)
        return None
    if kind == "boolean":
        return {"type": "noul", **base}, None
    low, high = _bounds(prop)
    if kind == "integer" and low is not None and high is not None and 2 <= high - low + 1 <= 10:
        values = list(range(low, high + 1))
        return {"type": "score", **base, "criteria": [str(v) for v in values]}, values
    return None


def from_schema(schema, prefix=""):
    """(questions, encode) when every property of an object schema maps to a question, else None: a string enum is a
    choice, a boolean a noul, and an integer enum or minimum-to-maximum range of 2 to 10 values a score. A property's
    description (or title) is its instructions. Keys are prefix + the property name, so apps that share a router and a
    property name (say "category") are calibrated apart. encode turns the router's answers into an object matching the
    schema."""
    root, schema = schema, _resolve(schema, schema)
    if schema is None or str(schema.get("type", "object")).lower() != "object" or not isinstance(schema.get("properties"), dict) or not schema["properties"]:
        return None
    questions, values = {}, {}
    for name, prop in schema["properties"].items():
        mapped = _question(prop, root)
        if mapped is None:
            return None
        questions[prefix + name], levels = mapped
        if levels:
            values[name] = levels

    def encode(answers):
        return {name: values[name][answers[prefix + name]["answer"]] if name in values else answers[prefix + name]["answer"] for name in schema["properties"]}

    return questions, encode
