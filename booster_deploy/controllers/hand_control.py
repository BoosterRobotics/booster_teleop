"""State gating for real-robot hand control."""


def hand_control_active_for_state(
    control_state: str,
    output_mode: str,
) -> bool:
    """Enable live hand input only while holding or actively tracking in Custom."""
    state = getattr(control_state, "value", control_state)
    if state == "walking":
        return False
    if state == "pvt_hold":
        return output_mode == "hold"
    return state == "tracking" and output_mode == "policy"
