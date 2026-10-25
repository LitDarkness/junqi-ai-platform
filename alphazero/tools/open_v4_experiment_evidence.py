"""Select teacher comparisons without mixing them with network duels."""
def teacher_deltas(arena, before, after):
    if arena.get("role") != "dev":
        raise ValueError("experiment selection requires a development arena")
    result = []
    for row in arena["paired_deltas"]:
        first = arena["summaries"][row["from_matchup"]]
        second = arena["summaries"][row["to_matchup"]]
        if (first["candidate"] == before and second["candidate"] == after
                and first["opponent"].startswith("teacher") and second["opponent"] == first["opponent"]):
            result.append(row["delta"])
    return result
