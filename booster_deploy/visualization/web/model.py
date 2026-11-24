"""MuJoCo model manifest for the browser renderer."""
from __future__ import annotations

from typing import Any
import struct

import mujoco
import numpy as np


def _name(model: mujoco.MjModel, obj: mujoco.mjtObj, index: int, fallback: str) -> str:
    return mujoco.mj_id2name(model, obj, int(index)) or fallback


def _list(values) -> list[float]:
    return np.asarray(values, dtype=np.float32).reshape(-1).tolist()


def build_model_manifest(
    model: mujoco.MjModel,
) -> dict[str, Any]:
    """Return stable body/visual metadata; dynamic poses travel separately."""
    bodies = [
        {
            "id": body_id,
            "name": _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id, f"body_{body_id}"),
            "parent": int(model.body_parentid[body_id]),
        }
        for body_id in range(int(model.nbody))
    ]
    visuals: list[dict[str, Any]] = []
    mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    for geom_id in range(int(model.ngeom)):
        if int(model.geom_type[geom_id]) != mesh_type or int(model.geom_group[geom_id]) != 1:
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            continue
        mesh_name = _name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id, f"mesh_{mesh_id}")
        visuals.append({
            "body": int(model.geom_bodyid[geom_id]),
            "mesh_id": mesh_id,
            "mesh": mesh_name,
            "url": f"/api/visualization/mesh/{mesh_id}",
            "pos": _list(model.geom_pos[geom_id]),
            "quat": _list(model.geom_quat[geom_id]),
            "rgba": _list(model.geom_rgba[geom_id]),
        })
    return {
        "version": 1,
        "model": _name(model, mujoco.mjtObj.mjOBJ_MODEL, 0, "robot"),
        "coordinate": "z_up_m_wxyz",
        "bodies": bodies,
        "visuals": visuals,
    }


def compiled_mesh_payload(model: mujoco.MjModel, mesh_id: int) -> dict[str, Any]:
    """Return MuJoCo-compiled vertices, in the exact geom-local coordinate frame."""
    mesh_id = int(mesh_id)
    if not 0 <= mesh_id < int(model.nmesh):
        raise IndexError(f"mesh id out of range: {mesh_id}")
    vert_adr = int(model.mesh_vertadr[mesh_id])
    vert_num = int(model.mesh_vertnum[mesh_id])
    face_adr = int(model.mesh_faceadr[mesh_id])
    face_num = int(model.mesh_facenum[mesh_id])
    vertices = np.asarray(model.mesh_vert[vert_adr:vert_adr + vert_num], dtype=np.float32)
    faces = np.asarray(model.mesh_face[face_adr:face_adr + face_num], dtype=np.int32)
    return {
        "version": 1,
        "mesh_id": mesh_id,
        "vertices": vertices.reshape(-1).tolist(),
        "faces": faces.reshape(-1).tolist(),
    }


def compiled_mesh_bytes(model: mujoco.MjModel, mesh_id: int) -> bytes:
    """Compact wire form: uint32 counts, float32 XYZ, uint32 triangle indices."""
    mesh_id = int(mesh_id)
    if not 0 <= mesh_id < int(model.nmesh):
        raise IndexError(f"mesh id out of range: {mesh_id}")
    vert_adr, vert_num = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    face_adr, face_num = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
    vertices = np.asarray(
        model.mesh_vert[vert_adr:vert_adr + vert_num], dtype="<f4"
    ).reshape(-1)
    faces = np.asarray(
        model.mesh_face[face_adr:face_adr + face_num], dtype="<u4"
    ).reshape(-1)
    return struct.pack("<II", vert_num, face_num) + vertices.tobytes() + faces.tobytes()


GEOMETRY_MAGIC = b"BVZ2"
GEOMETRY_VERSION = 2

FLAG_U16_INDICES = 1
FLAG_NORMALS = 2

#: header magic(4) + version(u16) + count(u16) + count * 7 * u32
GEOMETRY_HEADER = "<4sHH"
GEOMETRY_ENTRY_FIELDS = 7  # mesh_id, vert_num, face_num, flags, vert_off, norm_off, face_off
GEOMETRY_ENTRY = "<IIIIIII"


def _pad4(size: int) -> int:
    return (4 - size % 4) % 4


def geometry_bundle(model: mujoco.MjModel) -> bytes:
    """Every compiled mesh in one self-describing blob.

    A dashboard is often opened over a slow tunnel; 32 separate requests make a
    partial load (a robot rendered as "trunk only") both likely and invisible.
    One atomic request means the renderer either gets the whole robot or a clear
    error, never a silently incomplete one.

    Layout (little endian, every section offset recorded in the header):
        magic "BVZ2" | u16 version | u16 mesh_count
        mesh_count * (mesh_id, vertex_count, face_count, flags,
                      vertex_off, normal_off, face_off)   # u32 each
        float32 vertices, float32 normals, uint16/uint32 indices

    Indices are u16 whenever the mesh has fewer than 65536 vertices (halves the
    payload) and normals come from MuJoCo so the browser never has to run
    ``computeVertexNormals`` over a million faces at load time.
    """
    count = int(model.nmesh)
    header_size = struct.calcsize(GEOMETRY_HEADER) + count * struct.calcsize(GEOMETRY_ENTRY)
    entries: list[tuple[int, int, int, int, int, int, int]] = []
    chunks: list[bytes] = []
    offset = header_size
    for mesh_id in range(count):
        vert_adr, vert_num = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        face_adr, face_num = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
        normal_adr, normal_num = int(model.mesh_normaladr[mesh_id]), int(model.mesh_normalnum[mesh_id])

        vertices = np.asarray(
            model.mesh_vert[vert_adr:vert_adr + vert_num], dtype="<f4").reshape(-1)
        normals = np.asarray(
            model.mesh_normal[normal_adr:normal_adr + normal_num], dtype="<f4").reshape(-1)
        faces = np.asarray(
            model.mesh_face[face_adr:face_adr + face_num], dtype="<i4").reshape(-1)

        flags = 0
        if vert_num < 65536:
            flags |= FLAG_U16_INDICES
            face_bytes = faces.astype("<u2").tobytes()
        else:
            face_bytes = faces.astype("<u4").tobytes()
        if normal_num == vert_num and vert_num:
            # int8 normals (decoded as a normalised byte attribute in the browser)
            # cost 3 B/vertex instead of 12 B and keep shading smooth.
            flags |= FLAG_NORMALS
            normal_bytes = np.clip(np.rint(normals * 127.0), -127, 127).astype("i1").tobytes()
        else:
            normal_bytes = b""

        vertex_bytes = vertices.tobytes()
        vert_off = offset
        norm_off = vert_off + len(vertex_bytes)
        face_off = norm_off + len(normal_bytes)
        # int8 normals are 3 B/vertex, so the face block needs explicit padding to
        # stay 4-aligned: typed-array views on the browser side refuse odd offsets.
        normal_pad = b"\0" * _pad4(face_off)
        face_off += len(normal_pad)
        block = vertex_bytes + normal_bytes + normal_pad + face_bytes
        pad = _pad4(len(block))
        chunks.append(block + b"\0" * pad)
        entries.append((mesh_id, vert_num, face_num, flags, vert_off, norm_off, face_off))
        offset += len(block) + pad

    header = [struct.pack(GEOMETRY_HEADER, GEOMETRY_MAGIC, GEOMETRY_VERSION, count)]
    for entry in entries:
        header.append(struct.pack(GEOMETRY_ENTRY, *entry))
    return b"".join(header) + b"".join(chunks)
