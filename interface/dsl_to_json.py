def dsl_to_json(dsl_text):
    actors = []
    use_cases = []
    relationships = []

    lines = dsl_text.strip().split("\n")
    for line in lines:
        if line.startswith("Actor:"):
            actor = line.split(":", 1)[1].strip()
            if actor not in actors:
                actors.append(actor)
        elif line.startswith("UseCase:"):
            uc = line.split(":", 1)[1].strip()
            if uc not in use_cases:
                use_cases.append(uc)
        elif line.startswith("Rel:"):
            parts = line.split("->")
            if len(parts) == 2:
                actor = parts[0].split(":", 1)[1].strip()
                uc = parts[1].strip()
                relationships.append({"actor": actor, "use_case": uc})

    return {
        "actors": actors,
        "use_cases": use_cases,
        "relationships": relationships
    }
