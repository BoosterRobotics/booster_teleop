# NOITOM 动捕服穿戴与校准

简体中文 | [English](../EN/noitom-suit-calibration.md) | [README](../../README.zh-CN.md)

参考视频链接：

- [动捕服穿戴](https://www.bilibili.com/video/BV1pzzVBrEbQ/?spm_id_from=333.1387.homepage.video_card.click&vd_source=9e067b0e4d002a0c1995efce935abef7)
- [动捕服校准](https://www.bilibili.com/video/BV1wrzVBgEnn/?spm_id_from=333.1387.homepage.video_card.click&vd_source=9e067b0e4d002a0c1995efce935abef7)

其中，本项目使用 SDK 方式，并无视频中的动画提示。当执行程序 `python scripts/calibrate.py --viser` 后，终端会提示当前阶段的校准动作，按提示操作即可。

## 终端输出与操作说明

程序启动后，终端首先显示如下菜单：

```text
PN-Link opened | local=XXX server=XXX
==================================================
PN-Link MoCap Calibration
==================================================
N - Start Capture
C - Calibrate Motion
F/S - Stop Capture
Q - Exit
==================================================
[Stage 1] Press N to start capture. After motion data is received, press C to calibrate.
```

对应操作如下：

- `N`：开始采集（Start Capture）
- `C`：校准动作（Calibrate Motion）
- `F/S`：停止采集（Stop Capture）
- `Q`：退出（Exit）

**Stage 1**：按 `N` 开始采集，等待接收动捕数据后按 `C` 进入校准。

```text
> Key: Start Capture
Pending command Start Capture is running.
Start Capture done.
[Stage 2] Capture started. Waiting for motion data...
[Stage 2] Motion data received from device; capture is already running. Press C to calibrate.
```

**Stage 2**：采集已启动，等待设备上报动捕数据；收到数据后，按 `C` 进入校准流程。

```text
> Key: Calibrate Motion
Pending command Calibrate Motion is running.
Support poses: VB-Pose, P-Pose, T-Pose, A-Pose, F-Pose : Calibration-(VB-Pose)-Countdown 3.
Support poses: VB-Pose, P-Pose, T-Pose, A-Pose, F-Pose : Calibration-(VB-Pose)-Countdown 2.
Support poses: VB-Pose, P-Pose, T-Pose, A-Pose, F-Pose : Calibration-(VB-Pose)-Countdown 1.
Support poses: VB-Pose, P-Pose, T-Pose, A-Pose, F-Pose : Calibration-(VB-Pose)-Progress 0.
```

**校准阶段**：终端会依次提示需要摆出的姿势。`Calibration-(X-Pose)` 中的 `X-Pose` 即为当前阶段的校准动作提示，例如 `VB-Pose`、`P-Pose`、`T-Pose`、`A-Pose`、`F-Pose`。按提示摆好对应姿势后保持不动，程序会进行 `Countdown`（倒计时）与 `Progress`（进度，0~100）。


```text
Support poses: VB-Pose, P-Pose, T-Pose, A-Pose, F-Pose : Calibration-(F-Pose)-Progress 99.
Support poses: VB-Pose, P-Pose, T-Pose, A-Pose, F-Pose : Calibration-(F-Pose)-Progress 99.
Support poses: VB-Pose, P-Pose, T-Pose, A-Pose, F-Pose : Calibration-(F-Pose)-Progress 100.
Waiting for the completion of the calibration of human posture command execution
Calibrate Motion done.
PN-Link calibration complete. last_command_result=(3, 0)
Mocap ground alignment enabled | target z=0.000000
Waiting for first PN-Link HumanFrame before opening viser viewer...
Received first frame | seq=0 ts=14768.148983 joints=64
```

**校准完成**：当进度达到 `100` 并显示 `Calibrate Motion done.`、`PN-Link calibration complete.` 后，表示动捕服校准完成。随后系统会进行地面校准（`Mocap ground alignment enabled`），并等待接收第一帧人体数据（`Waiting for first PN-Link HumanFrame...`）。收到首帧后（`Received first frame ... joints=64`），会启动可视化查看器：

```text
──────────────── viser (listening *:8080) ────────────────
 HTTP      |  http://localhost:8080
 Websocket |  ws://localhost:8080
──────────────────────────────────────────────────────────
```

此时可在浏览器中打开 `http://localhost:8080` 查看动捕数据可视化效果。
