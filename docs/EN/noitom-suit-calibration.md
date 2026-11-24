# Wearing and Calibrating the Noitom Motion Capture Suit

English | [简体中文](../ZH/NOITOM动捕服校准.md) | [README](../../README.md)

Reference videos:

- [Putting on the suit](https://www.bilibili.com/video/BV1pzzVBrEbQ/?spm_id_from=333.1387.homepage.video_card.click&vd_source=9e067b0e4d002a0c1995efce935abef7)
- [Calibrating the suit](https://www.bilibili.com/video/BV1wrzVBgEnn/?spm_id_from=333.1387.homepage.video_card.click&vd_source=9e067b0e4d002a0c1995efce935abef7)

This project uses the SDK and does not show the animated instructions seen in the videos. After you run `python scripts/calibrate.py --viser`, the terminal prompts you for the calibration pose required at each stage. Follow those prompts.

## Terminal Output and Controls

When the program starts, the terminal displays this menu:

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

The controls are:

- **N**: Start capture.
- **C**: Calibrate motion.
- **F/S**: Stop capture.
- **Q**: Exit.

**Stage 1**: Press **N** to start capture. Wait until motion data is received, then press **C** to begin calibration.

```text
> Key: Start Capture
Pending command Start Capture is running.
Start Capture done.
[Stage 2] Capture started. Waiting for motion data...
[Stage 2] Motion data received from device; capture is already running. Press C to calibrate.
```

**Stage 2**: Capture has started and the program is waiting for motion data from the device. Once data arrives, press **C** to enter the calibration process.

```text
> Key: Calibrate Motion
Pending command Calibrate Motion is running.
Support poses: VB-Pose, P-Pose, T-Pose, A-Pose, F-Pose : Calibration-(VB-Pose)-Countdown 3.
Support poses: VB-Pose, P-Pose, T-Pose, A-Pose, F-Pose : Calibration-(VB-Pose)-Countdown 2.
Support poses: VB-Pose, P-Pose, T-Pose, A-Pose, F-Pose : Calibration-(VB-Pose)-Countdown 1.
Support poses: VB-Pose, P-Pose, T-Pose, A-Pose, F-Pose : Calibration-(VB-Pose)-Progress 0.
```

**Calibration**: The terminal prompts you for each pose in sequence. The `X-Pose` in `Calibration-(X-Pose)` identifies the pose required at the current stage, such as `VB-Pose`, `P-Pose`, `T-Pose`, `A-Pose`, or `F-Pose`. Assume the requested pose and hold still while the program displays `Countdown` and `Progress` (0–100).

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

**Completion**: Calibration is complete when progress reaches `100` and the terminal displays `Calibrate Motion done.` and `PN-Link calibration complete.`. The system then enables ground alignment (`Mocap ground alignment enabled`) and waits for the first human frame (`Waiting for first PN-Link HumanFrame...`). Once that frame arrives (`Received first frame ... joints=64`), the visualization viewer starts:

```text
──────────────── viser (listening *:8080) ────────────────
 HTTP      |  http://localhost:8080
 Websocket |  ws://localhost:8080
──────────────────────────────────────────────────────────
```

Open `http://localhost:8080` in a browser to view the motion capture data.
