"""The question format: an object of named questions, each a choice, score or noul (yes/no)."""
import json
from pathlib import Path


def validate(questions):
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty object of named questions")
    problems = []
    for key, q in questions.items():
        if not isinstance(q, dict) or q.get("type") not in ("choice", "score", "noul"):
            problems.append(f"{key}: type must be choice, score or noul")
            continue
        if extra := set(q) - {"type", "instructions", "criteria"}:
            problems.append(f"{key}: unsupported fields {sorted(extra)}")
        criteria = q.get("criteria")
        if q["type"] == "choice" and not (isinstance(criteria, dict) and 1 <= len(criteria) <= 255):
            problems.append(f"{key}: choice needs a criteria object with 1 to 255 options")
        if q["type"] == "score" and not (isinstance(criteria, list) and 2 <= len(criteria) <= 10):
            problems.append(f"{key}: score needs a criteria list with 2 to 10 levels")
        if q["type"] == "noul" and criteria is not None and not (isinstance(criteria, dict) and set(criteria) <= {"true", "false"}):
            problems.append(f"{key}: noul criteria may only have 'true' and 'false'")
    if problems:
        raise ValueError("; ".join(problems))
    return questions


def question_sets(data, source="questions"):
    """{name: questions} from either one object of named questions (named "default") or an object of named sets of
    them, with every set validated."""
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{source} must be a non-empty object of questions, or of named sets of them")
    sets = {"default": data} if any(isinstance(q, dict) and isinstance(q.get("type"), str) for q in data.values()) else data
    for name, questions in sets.items():
        try:
            validate(questions)
        except ValueError as e:
            raise ValueError(f"{source}, set {name!r}: {e}") from e
    return sets


def load(path):
    """question_sets from a JSON file, or a YAML one (.yaml or .yml, needing PyYAML)."""
    path = Path(path)
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError('YAML questions need PyYAML: pip install "sureshot[yaml]"') from e
        data = yaml.safe_load(path.read_text())
    else:
        data = json.loads(path.read_text())
    return question_sets(data, str(path))


def check_answer(question, answer):
    """Raises ValueError unless answer is a valid answer to the question."""
    if question["type"] == "noul":
        ok = isinstance(answer, bool)
    elif question["type"] == "score":
        ok = isinstance(answer, int) and not isinstance(answer, bool) and 0 <= answer < len(question["criteria"])
    else:
        ok = isinstance(answer, str) and answer in question["criteria"]
    if not ok:
        raise ValueError(f"{answer!r} is not a valid answer to this {question['type']} question")


def answer_schema(questions):
    """JSON Schema for one answer and a one-sentence reason per question, used to constrain System 2."""
    props = {}
    for key, q in questions.items():
        if q["type"] == "choice":
            answer = {"type": "string", "enum": list(q["criteria"])}
        elif q["type"] == "score":
            answer = {"type": "integer", "enum": list(range(len(q["criteria"])))}
        else:
            answer = {"type": "boolean"}
        props[key] = {"type": "object", "properties": {"answer": answer, "reason": {"type": "string"}},
                      "required": ["answer", "reason"], "additionalProperties": False}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def describe(questions):
    """The questions as text for a language model, with how each type is answered."""
    how = {"choice": "one of the option keys", "score": "the 0-based index of the level", "noul": "true or false"}
    return "\n\n".join(f"{key} ({q['type']}: answer with {how[q['type']]})\n{json.dumps({k: v for k, v in q.items() if k != 'type'}, indent=2)}"
                       for key, q in questions.items())
