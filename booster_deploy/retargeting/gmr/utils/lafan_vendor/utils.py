
import numpy as np


def length(x, axis=-1, keepdims=True):
    lgth = np.sqrt(np.sum(x * x, axis=axis, keepdims=keepdims))
    return lgth


def normalize(x, axis=-1, eps=1e-8):
    res = x / (length(x, axis=axis) + eps)
    return res


def quat_normalize(x, eps=1e-8):
    res = normalize(x, eps=eps)
    return res


def angle_axis_to_quat(angle, axis):
    c = np.cos(angle / 2.0)[..., np.newaxis]
    s = np.sin(angle / 2.0)[..., np.newaxis]
    q = np.concatenate([c, s * axis], axis=-1)
    return q


def euler_to_quat(e, order="zyx"):
    axis = {"x": np.asarray([1, 0, 0], dtype=np.float32), "y": np.asarray([0, 1, 0], dtype=np.float32), "z": np.asarray([0, 0, 1], dtype=np.float32)}

    q0 = angle_axis_to_quat(e[..., 0], axis[order[0]])
    q1 = angle_axis_to_quat(e[..., 1], axis[order[1]])
    q2 = angle_axis_to_quat(e[..., 2], axis[order[2]])

    return quat_mul(q0, quat_mul(q1, q2))


def quat_inv(q):
    res = np.asarray([1, -1, -1, -1], dtype=np.float32) * q
    return res


def quat_fk(lrot, lpos, parents):
    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(quat_mul_vec(gr[parents[i]], lpos[..., i : i + 1, :]) + gp[parents[i]])
        gr.append(quat_mul(gr[parents[i]], lrot[..., i : i + 1, :]))

    res = np.concatenate(gr, axis=-2), np.concatenate(gp, axis=-2)
    return res


def quat_ik(grot, gpos, parents):
    res = [
        np.concatenate(
            [
                grot[..., :1, :],
                quat_mul(quat_inv(grot[..., parents[1:], :]), grot[..., 1:, :]),
            ],
            axis=-2,
        ),
        np.concatenate(
            [
                gpos[..., :1, :],
                quat_mul_vec(quat_inv(grot[..., parents[1:], :]), gpos[..., 1:, :] - gpos[..., parents[1:], :]),
            ],
            axis=-2,
        ),
    ]

    return res


def quat_mul(x, y):
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    res = np.concatenate(
        [
            y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
            y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
            y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
            y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0,
        ],
        axis=-1,
    )

    return res


def quat_mul_vec(q, x):
    t = 2.0 * np.cross(q[..., 1:], x)
    res = x + q[..., 0][..., np.newaxis] * t + np.cross(q[..., 1:], t)

    return res


def quat_slerp(x, y, a):
    len = np.sum(x * y, axis=-1)

    neg = len < 0.0
    len[neg] = -len[neg]
    y[neg] = -y[neg]

    a = np.zeros_like(x[..., 0]) + a
    amount0 = np.zeros(a.shape)
    amount1 = np.zeros(a.shape)

    linear = (1.0 - len) < 0.01
    omegas = np.arccos(len[~linear])
    sinoms = np.sin(omegas)

    amount0[linear] = 1.0 - a[linear]
    amount0[~linear] = np.sin((1.0 - a[~linear]) * omegas) / sinoms

    amount1[linear] = a[linear]
    amount1[~linear] = np.sin(a[~linear] * omegas) / sinoms
    res = amount0[..., np.newaxis] * x + amount1[..., np.newaxis] * y

    return res


def quat_between(x, y):
    res = np.concatenate(
        [np.sqrt(np.sum(x * x, axis=-1) * np.sum(y * y, axis=-1))[..., np.newaxis] + np.sum(x * y, axis=-1)[..., np.newaxis], np.cross(x, y)], axis=-1
    )
    return res


def interpolate_local(lcl_r_mb, lcl_q_mb, n_past, n_future):

    start_lcl_r_mb = lcl_r_mb[:, n_past - 1, :, :][:, None, :, :]
    end_lcl_r_mb = lcl_r_mb[:, -n_future, :, :][:, None, :, :]

    start_lcl_q_mb = lcl_q_mb[:, n_past - 1, :, :]
    end_lcl_q_mb = lcl_q_mb[:, -n_future, :, :]


    n_trans = lcl_r_mb.shape[1] - (n_past + n_future)
    interp_ws = np.linspace(0.0, 1.0, num=n_trans + 2, dtype=np.float32)
    offset = end_lcl_r_mb - start_lcl_r_mb

    const_trans = np.tile(start_lcl_r_mb, [1, n_trans + 2, 1, 1])
    inter_lcl_r_mb = const_trans + (interp_ws)[None, :, None, None] * offset


    interp_ws = np.linspace(0.0, 1.0, num=n_trans + 2, dtype=np.float32)
    inter_lcl_q_mb = np.stack([(quat_normalize(quat_slerp(quat_normalize(start_lcl_q_mb), quat_normalize(end_lcl_q_mb), w))) for w in interp_ws], axis=1)

    return inter_lcl_r_mb, inter_lcl_q_mb


def remove_quat_discontinuities(rotations):
    rots_inv = -rotations

    for i in range(1, rotations.shape[0]):

        replace_mask = np.sum(rotations[i - 1 : i] * rotations[i : i + 1], axis=-1) < np.sum(rotations[i - 1 : i] * rots_inv[i : i + 1], axis=-1)
        replace_mask = replace_mask[..., np.newaxis]
        rotations[i] = replace_mask * rots_inv[i] + (1.0 - replace_mask) * rotations[i]

    return rotations



def rotate_at_frame(X, Q, parents, n_past=10):

    global_q, global_x = quat_fk(Q, X, parents)

    key_glob_Q = global_q[:, n_past - 1 : n_past, 0:1, :]
    forward = np.array([1, 0, 1])[np.newaxis, np.newaxis, np.newaxis, :] * quat_mul_vec(key_glob_Q, np.array([0, 1, 0])[np.newaxis, np.newaxis, np.newaxis, :])
    forward = normalize(forward)
    yrot = quat_normalize(quat_between(np.array([1, 0, 0]), forward))
    new_glob_Q = quat_mul(quat_inv(yrot), global_q)
    new_glob_X = quat_mul_vec(quat_inv(yrot), global_x)


    Q, X = quat_ik(new_glob_Q, new_glob_X, parents)

    return X, Q


def extract_feet_contacts(pos, lfoot_idx, rfoot_idx, velfactor=0.02):
    lfoot_xyz = (pos[1:, lfoot_idx, :] - pos[:-1, lfoot_idx, :]) ** 2
    contacts_l = np.sum(lfoot_xyz, axis=-1) < velfactor

    rfoot_xyz = (pos[1:, rfoot_idx, :] - pos[:-1, rfoot_idx, :]) ** 2
    contacts_r = np.sum(rfoot_xyz, axis=-1) < velfactor


    contacts_l = np.concatenate([contacts_l, contacts_l[-1:]], axis=0)
    contacts_r = np.concatenate([contacts_r, contacts_r[-1:]], axis=0)

    return contacts_l, contacts_r
