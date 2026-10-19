# OpenArm MuJoCo

## Start

Install Python 3.13 (`brew install python@3.13` on macOS), then run from the project folder:

```bash
./scripts/setup.sh   # first time only
./scripts/start.sh
```

For later launches, just run `./scripts/start.sh`.

## Use

1. Move the joint sliders to pose A, let the arm settle, then click **Save A position**. Repeat for B.
2. Set **Move duration**, then click **Run A -> B sweep**.
3. Click **Open last report** to inspect torque and tracking error. Staying below the red torque limit alone does not mean the move succeeded.

- **Go to A / B:** move to a saved pose.
- **Jog speed:** change slider movement speed. **Zero arm / Randomize joints:** set new targets.
- **Link lengths / weights:** edit lengths (mm), link/actuator masses, end-effector mass, and payload (kg). Uncheck **Auto mass** to enter a link's mass excluding motors.
- **Apply + save:** rebuild and save geometry/masses, then close the editor and press **Resume**. **Apply** makes temporary changes.
- **Advanced P / D:** adjust each joint's gains. **Save controls** saves gains, speed, and duration.
- **Pause / Resume / Stop:** control motion.
- **Reload YAML:** load external edits to `configs/arm.yaml`.

Charts: J1–J7, top to bottom. Yellow = nominal torque; red = maximum torque.

Camera: drag to rotate, right-drag to pan, scroll to zoom.
