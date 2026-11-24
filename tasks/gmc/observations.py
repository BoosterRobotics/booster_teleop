"""Pack GMC input histories in the layout expected by the ONNX model."""
import torch


def state_term_widths(cfg, num_joints: int) -> tuple[int, ...]:
    joints = cfg.num_joints or num_joints
    actions = cfg.num_actions or num_joints
    return tuple(width for enabled, width in (
        (cfg.include_projected_gravity_in_state, cfg.projected_gravity_dim),
        (cfg.include_root_ang_vel_in_state, cfg.root_ang_vel_dim),
        (cfg.include_joint_pos_in_state, joints),
        (cfg.include_joint_vel_in_state, joints),
        (cfg.include_last_action_in_state, actions),
    ) if enabled)


def flatten_history(
    history: torch.Tensor,
    command_width: int,
    command_length: int,
    state_length: int,
    state_widths: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    commands = history[-command_length:, :command_width].reshape(1, -1)
    states = history[-state_length:]
    terms = []
    cursor = command_width
    for width in state_widths:
        terms.append(states[:, cursor : cursor + width].reshape(-1))
        cursor += width
    return commands, torch.cat(terms, dim=-1).reshape(1, -1)


def state_row(
    cfg,
    projected_gravity: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    last_action: torch.Tensor,
) -> torch.Tensor:
    return torch.cat([value for enabled, value in (
        (cfg.include_projected_gravity_in_state, projected_gravity),
        (cfg.include_root_ang_vel_in_state, root_ang_vel),
        (cfg.include_joint_pos_in_state, joint_pos),
        (cfg.include_joint_vel_in_state, joint_vel),
        (cfg.include_last_action_in_state, last_action),
    ) if enabled], dim=-1)
