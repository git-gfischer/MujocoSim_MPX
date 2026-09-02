import readline
import readchar
import time
import numpy as np
import glfw

# Config imports
import mpx.config.robot_config.config_aliengo as config_aliengo

from mpx.utils.quad_utils_balance.foot_reference import (
    swing_foot_anchor_from_target,
    foot_target_foot_local_to_world,
)

class Console():
    def __init__(self, controller_node):
        self.controller_node = controller_node

        # Walking and Stopping
        self.walking = False

        # Go Up and Go Down motion
        self.isDown = True
        self.height_delta = config_aliengo.robot_height

        # Pitch Up and Pitch Down
        self.pitch_delta = 0

        # Step Height holder to keep track of the step height
        self.step_height_holder = config_aliengo.step_height

        # Autocomplete setup
        self.commands = [
            "stw", "ooo", "setStepHeight",
           "goUp", "goDown", "help", "ictp","setupGaitTimer"
        ]
        readline.set_completer(self.complete)
        readline.parse_and_bind("tab: complete")


    def complete(self, text, state):
        options = [cmd for cmd in self.commands if cmd.startswith(text)]
        if state < len(options):
            print(options[state])
            return options[state]
        else:
            return None


    def interactive_command_line(self, ):
        self.print_all_commands()
        while True:
            input_string = input(">>> ")
            try:
                if(input_string == "stw"):
                    if(self.walking):
                        print("The robot is already walking")
                    print("Starting Walking")
                    self.walking = True
                    self.controller_node.mpc.walking = True
                    self.controller_node.mpc.duty_factor = config_aliengo.duty_factor
                elif(input_string == "ooo"):
                    print("Stopping Walking")
                    self.walking = False
                    while(np.sum(self.controller_node.mpc.contact) < 3):
                        time.sleep(0.02)
                    self.controller_node.mpc.duty_factor = 1.0
                    self.controller_node.mpc.contact_time = self.controller_node.mpc.config.timer_t
                    self.controller_node.input[:6] = np.zeros(6)
                    ##TO DO stop walking
                elif(input_string == "goUp"):
                    print("Going Up")
                    start_time = time.time()
                    time_motion = 5.
                    initial_height = self.controller_node.mpc.robot_height
                    delta_height = self.controller_node.mpc.config.robot_height - initial_height
                    while(time.time() - start_time < time_motion):
                        time_diff = time.time() - start_time
                        self.controller_node.mpc.robot_height = initial_height + ( delta_height * time_diff / time_motion)
                        time.sleep(0.01)
                    self.controller_node.isDown = False
                    print("Ready to walk")
                elif(input_string == "goDown"):
                    print("Going Up")
                    start_time = time.time()
                    time_motion = 5.
                    initial_height = self.controller_node.mpc.robot_height
                    delta_height = 0.05 - initial_height
                    print("Initial Height: ", initial_height)
                    while(time.time() - start_time < time_motion):
                        time_diff = time.time() - start_time
                        self.controller_node.mpc.robot_height = initial_height + ( delta_height * time_diff / time_motion)
                        time.sleep(0.01)
                    self.controller_node.isDown = False
                elif(input_string == "setStepHeight"):
                    temp = input("Step Height: >>> ")
                    if(temp != ""):
                        temp = max(0.02, min(float(temp), 0.5))
                        self.controller_node.mpc.step_height = temp
                        
                elif(input_string == "setGaitTimer"):
                    
                    print("Current Step Frequency: ", self.mpc.step_freq)
                    temp = input("Step Frequency: >>> ")
                    if(temp != ""):
                        temp = max(0.4, min(float(temp), 2.0))
                        self.controller_node.mpc.step_freq = temp
                    
                    print("Current Duty Factor: ", self.mpc.duty_factor)
                    temp = input("Duty Factor: >>> ")
                    if(temp != ""):
                        temp = max(0.4, min(float(temp), 0.9))
                        self.controller_node.mpc.duty_factor = temp  

                elif(input_string == "robot_height"):
                    temp = input("Robot Height: >>> ")
                    if(temp != ""):
                        temp = max(0.1, min(float(temp), 0.4))
                        self.controller_node.mpc.robot_height = temp
                
                elif(input_string == "help"):
                    self.print_all_commands()

                
                elif(input_string == "ictp"):
                    print("Interactive Keyboard Control")
                    print("w: Move Forward")
                    print("s: Move Backward")
                    print("a: Move Left")
                    print("d: Move Right")
                    print("q: Rotate Left")
                    print("e: Rotate Right")
                    print("0: Stop")
                    print("1: Pitch Up")
                    print("2: Reset Pitch")
                    print("3: Pitch Down")
                    print("Press any other key to exit")
                    while True:
                        command = readchar.readkey()
                        if(command == "w"):
                            self.controller_node.input[0] += 0.1
                            print("w")
                        elif(command == "s"):
                            self.controller_node.input[0] -= 0.1
                            print("s")
                        elif(command == "a"):
                            self.controller_node.input[1] += 0.05
                            print("a")
                        elif(command == "d"):
                            self.controller_node.input[1] -= 0.05
                            print("d")
                        elif(command == "q"):
                            self.controller_node.input[5] += 0.2
                            print("q")
                        elif(command == "e"):
                            self.controller_node.input[5] -= 0.2
                            print("e")
                        elif(command == "0"):
                            self.controller_node.input[0] = 0
                            self.controller_node.input[1] = 0
                            self.controller_node.input[5] = 0 
                            print("0")
                        # elif(command == "1"):
                        #     self.controller_node.pitch_delta -= 0.1
                        #     print("1")
                        # elif(command == "2"):
                        #     self.controller_node.pitch_delta = 0
                        #     print("2")
                        # elif(command == "3"):
                        #     self.controller_node.pitch_delta += 0.1
                        #     print("3")
                        else:
                            #to do maybe stop the robot
                            break
            except Exception as e:
                print("Error: ", e)
                print("Invalid Command")
                self.print_all_commands()


    def print_all_commands(self):
        print("\nAvailable Commands")
        print("help: Display all available messages")
        print("stw: Start Walking")
        print("ooo: Stop Walking")
        print("ictp: Interactive Keyboard Control")
        print("########################")
        print("narrowstance: Narrow Stance")
        print("widestance: Wide Stance")
        print("goUp: The robot goes up")
        print("goDown: The robot goes down")
        print("########################")
        print("setGaitTimer: Set the gait type")
        print("setupGaitTimer: Setup the gait timer")
        print("setupLegsGains: Setup the leg gains")
        print("setupGeneral: Setup general parameters\n")

#=============================================================================
class KeyboardVelocityCommand:
    """Arrow-key forward and yaw command for passive MuJoCo viewers.

    Usage:
    `command = KeyboardVelocityCommand(vx=0.3)`
    `viewer = mujoco.viewer.launch_passive(..., key_callback=command.key_callback)`
    `mpc_input = command.mpc_input(robot_height)`
    """

    def __init__(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        wz: float = 0.0,
        forward_step: float = 0.1,
        yaw_step: float = 0.2,
        forward_limits: tuple[float, float] = (-1.0, 1.0),
        yaw_limits: tuple[float, float] = (-1.5, 1.5),
    ):
        self.vx = float(vx)
        self.vy = float(vy)
        self.wz = float(wz)
        self.forward_step = float(forward_step)
        self.yaw_step = float(yaw_step)
        self.forward_limits = tuple(float(value) for value in forward_limits)
        self.yaw_limits = tuple(float(value) for value in yaw_limits)
        self._overlay_dirty = True

    def _clip(self):
        self.vx = float(np.clip(self.vx, *self.forward_limits))
        self.wz = float(np.clip(self.wz, *self.yaw_limits))

    def reset(self):
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self._overlay_dirty = True

    def key_callback(self, key: int):
        """Update the planar command from GLFW arrow keys."""

        if glfw is None:
            raise RuntimeError("glfw is required to use KeyboardVelocityCommand.")

        # Left/right arrows steer the commanded yaw rate around the vertical axis.
        if key == glfw.KEY_UP:      self.vx += self.forward_step
        elif key == glfw.KEY_DOWN:  self.vx -= self.forward_step
        elif key == glfw.KEY_LEFT:  self.wz += self.yaw_step
        elif key == glfw.KEY_RIGHT: self.wz -= self.yaw_step
        elif key in (glfw.KEY_SPACE, glfw.KEY_ENTER, glfw.KEY_BACKSPACE):
            self.reset()
        else:
            return

        self._clip()
        self._overlay_dirty = True

    def planar_command(self) -> np.ndarray:
        """Return the current planar command `[vx, vy]`."""

        return np.array([self.vx, self.vy], dtype=np.float64)

    def mpc_input(self, robot_height: float) -> np.ndarray:
        """Return the 7D locomotion command used by the MPC examples."""

        return np.array(
            [self.vx, self.vy, 0.0, 0.0, 0.0, self.wz, robot_height],
            dtype=np.float64,
        )

    def overlay_text(self) -> tuple[str, str]:
        """Return short viewer text showing controls and the current command."""

        return (
            "Up/Down: forward | Left/Right: yaw | Space: stop",
            f"vx {self.vx:+.2f}  wz {self.wz:+.2f}",
        )

    def consume_overlay_text(self) -> tuple[str, str] | None:
        """Return overlay text only when the command was updated."""

        if not self._overlay_dirty:
            return None
        self._overlay_dirty = False
        return self.overlay_text()

#=============================================================================
class SwingFootCommand:
    """Keyboard control for the swing foot target in tripod balance mode.

    Keys (GLFW):
      I / K  : forward / backward  (robot X axis)
      J / L  : left / right        (robot Y axis)
      U / O  : up / down           (world Z axis)

    Usage::

        swing_cmd = SwingFootCommand(swing_leg_idx=0, step=0.025)

        # In key_callback:
        foot_anchor = swing_cmd.key_callback(key, foot_anchor, data.qpos[3:7])

        # Check if a key was handled before passing to other handlers:
        if not swing_cmd.is_swing_key(key):
            command_handle.key_callback(key)
    """

    def __init__(self, swing_leg_idx: int = 0, step: float = 0.025):
        self.swing_leg_idx = int(swing_leg_idx)
        self.step = float(step)
        self._overlay_dirty = False

    def _deltas(self):
        if glfw is None:
            return {}
        s = self.step
        return {
            glfw.KEY_I: np.array([ s, 0.0, 0.0]),   # forward
            glfw.KEY_K: np.array([-s, 0.0, 0.0]),   # backward
            glfw.KEY_J: np.array([0.0,  s, 0.0]),   # left
            glfw.KEY_L: np.array([0.0, -s, 0.0]),   # right
            glfw.KEY_U: np.array([0.0, 0.0,  s]),   # up
            glfw.KEY_O: np.array([0.0, 0.0, -s]),   # down
        }

    def is_swing_key(self, key: int) -> bool:
        """True if ``key`` is handled by this handler."""
        return key in self._deltas()

    def key_callback(
        self,
        key: int,
        foot_anchor: np.ndarray,
        base_quat_wxyz: np.ndarray,
    ) -> np.ndarray:
        """Move the swing foot target if a swing key was pressed.

        Returns the (possibly updated) flat ``(3 * n_contact,)`` foot anchor.
        """
        deltas = self._deltas()
        if key not in deltas:
            return foot_anchor

        idx = self.swing_leg_idx
        current_world = np.asarray(foot_anchor[3*idx : 3*idx+3], dtype=np.float64)
        new_target = foot_target_foot_local_to_world(
            current_world, base_quat_wxyz, deltas[key]
        )
        updated = swing_foot_anchor_from_target(
            np.asarray(foot_anchor, dtype=np.float64), idx, new_target
        )
        self._overlay_dirty = True
        return updated

    def overlay_text(self) -> tuple[str, str]:
        return (
            "I/K: fwd/bwd  J/L: left/right  U/O: up/down",
            f"swing leg {self.swing_leg_idx}  step {self.step:.3f} m",
        )

    def consume_overlay_text(self) -> tuple[str, str] | None:
        if not self._overlay_dirty:
            return None
        self._overlay_dirty = False
        return self.overlay_text()

    def reset(self) -> None:
        """Reset overlay dirty flag (called on respawn)."""
        self._overlay_dirty = False