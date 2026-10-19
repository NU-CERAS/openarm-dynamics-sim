"""One native window: MuJoCo rendering/physics plus a small ImGui control panel."""
# Import the bundle first so Python GLFW and ImGui use ONE shared GLFW library.
# This avoids duplicate Cocoa classes and incompatible window pointers on macOS.
from imgui_bundle import imgui
import glfw
from OpenGL import GL
import ctypes
from pathlib import Path
import subprocess
import sys
import time
import webbrowser
import yaml
import mujoco
import numpy as np
from .experiment import Experiment
from .torque_charts import chart_width, draw_torque_charts

PANEL_WIDTH = 380


def open_editor(path):
    if sys.platform == 'darwin':
        subprocess.Popen(['open','-t',str(path)])
    elif sys.platform == 'win32':
        import os
        os.startfile(str(path))
    else:
        subprocess.Popen(['xdg-open',str(path)])


class Window:
    def __init__(self, config_path, out):
        self.experiment = Experiment(config_path,out)
        self.selected = 0
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.lookat[:] = [0,0,.45]
        self.camera.distance = 1.5
        self.camera.azimuth, self.camera.elevation = 135,-15
        self.option = mujoco.MjvOption()
        self.scene = self.context = None
        self.drag_button = None
        self.cursor = (0,0)
        self.reload_requested = False
        self.closed = False

    def render_resources(self):
        if self.context is not None:
            self.context.free()
        self.scene = None
        self.scene = mujoco.MjvScene(self.experiment.arm.model,maxgeom=2000)
        self.context = mujoco.MjrContext(self.experiment.arm.model,mujoco.mjtFontScale.mjFONTSCALE_150)

    def act(self, action, *args):
        try:
            action(*args)
        except (ValueError,KeyError,TypeError,RuntimeError,OSError,yaml.YAMLError) as exc:
            self.experiment.message = str(exc)

    def reload(self):
        self.experiment.idle_required()
        self.experiment.load()
        self.reload_requested = True
        self.experiment.message = 'Reloaded YAML. Initialized at saved pose A.'

    def report(self):
        if self.experiment.last_report:
            webbrowser.open(self.experiment.last_report.as_uri())

    def key(self, window, key, scancode, action, mods):
        imgui.backends.glfw_key_callback(self.address,key,scancode,action,mods)
        if action != glfw.PRESS or imgui.get_io().want_capture_keyboard:
            return
        e = self.experiment
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window,True)
        elif key == glfw.KEY_SPACE:
            e.paused = not e.paused
        elif key == glfw.KEY_X:
            self.act(e.stop)
        elif key == glfw.KEY_P:
            self.report()
        elif key == glfw.KEY_E:
            open_editor(e.path)
        elif key == glfw.KEY_F5:
            self.act(self.reload)
        elif key == glfw.KEY_F8:
            self.act(e.save_settings)
        elif not e.busy:
            if glfw.KEY_1 <= key <= glfw.KEY_7:
                self.selected = key-glfw.KEY_1
            elif key in (glfw.KEY_UP,glfw.KEY_DOWN):
                j = e.cfg['joints'][self.selected]
                degrees = np.rad2deg(e.target[self.selected]) + (2 if key==glfw.KEY_UP else -2)
                self.act(e.set_target,self.selected,float(np.clip(degrees,*j['limits_deg'])))
            elif key in (glfw.KEY_A,glfw.KEY_B):
                self.act(e.save_pose,chr(key))
            elif key == glfw.KEY_ENTER:
                self.act(e.start_sweep)
            elif key in (glfw.KEY_LEFT_BRACKET,glfw.KEY_RIGHT_BRACKET):
                self.act(e.set_duration,round(max(.1,min(120,e.cfg['sweep']['duration_s']+(-.5 if key==glfw.KEY_LEFT_BRACKET else .5))),3))

    def mouse_button(self, window, button, action, mods):
        imgui.backends.glfw_mouse_button_callback(self.address,button,action,mods)
        x,y = glfw.get_cursor_pos(window)
        if action == glfw.PRESS and PANEL_WIDTH <= x < glfw.get_window_size(window)[0]-chart_width(glfw.get_window_size(window)[0]) and not imgui.get_io().want_capture_mouse:
            self.drag_button = button
            self.cursor = (x,y)
        elif action == glfw.RELEASE:
            self.drag_button = None

    def mouse_move(self, window, x, y):
        imgui.backends.glfw_cursor_pos_callback(self.address,x,y)
        if self.drag_button is not None:
            dx,dy = x-self.cursor[0],y-self.cursor[1]
            height = max(1,glfw.get_window_size(window)[1])
            shift = glfw.get_key(window,glfw.KEY_LEFT_SHIFT)==glfw.PRESS or glfw.get_key(window,glfw.KEY_RIGHT_SHIFT)==glfw.PRESS
            if self.drag_button == glfw.MOUSE_BUTTON_LEFT:
                action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            elif self.drag_button == glfw.MOUSE_BUTTON_RIGHT:
                action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
            else:
                action = mujoco.mjtMouse.mjMOUSE_ZOOM
            mujoco.mjv_moveCamera(self.experiment.arm.model,action,dx/height,dy/height,self.camera)
        self.cursor=(x,y)

    def scroll(self, window, xoffset, yoffset):
        imgui.backends.glfw_scroll_callback(self.address,xoffset,yoffset)
        if PANEL_WIDTH <= glfw.get_cursor_pos(window)[0] < glfw.get_window_size(window)[0]-chart_width(glfw.get_window_size(window)[0]) and not imgui.get_io().want_capture_mouse:
            mujoco.mjv_moveCamera(self.experiment.arm.model,mujoco.mjtMouse.mjMOUSE_ZOOM,0,-.05*yoffset,self.camera)

    def panel(self, height):
        e = self.experiment
        imgui.set_next_window_pos((0,0))
        imgui.set_next_window_size((PANEL_WIDTH,height))
        flags = imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_saved_settings
        imgui.begin('Arm controls',flags=flags)
        imgui.text('OpenArm / MuJoCo')
        imgui.text_colored((.4,.8,1,1),e.state)
        imgui.begin_disabled(e.busy)
        if imgui.button('Save A position',(171,30)): self.act(e.save_pose,'A')
        imgui.same_line()
        if imgui.button('Save B position',(171,30)): self.act(e.save_pose,'B')
        if imgui.button('Go to A',(171,30)): self.act(e.go_to,'A')
        imgui.same_line()
        if imgui.button('Go to B',(171,30)): self.act(e.go_to,'B')
        # A Go-to above may have started this frame. Experiment methods also
        # enforce idle state, so queued clicks cannot overlap motions.
        if imgui.button('Run A -> B sweep',(350,32)): self.act(e.start_sweep)
        imgui.end_disabled()
        if imgui.button('Resume' if e.paused else 'Pause',(171,28)): e.paused=not e.paused
        imgui.same_line()
        if imgui.button('Stop / hold',(171,28)): self.act(e.stop)
        imgui.separator()
        imgui.begin_disabled(e.busy)
        imgui.set_next_item_width(170)
        changed,value = imgui.input_float('Move duration (s)',e.cfg['sweep']['duration_s'],.5,1,'%.2f')
        if changed: self.act(e.set_duration,value)
        imgui.set_next_item_width(170)
        changed,value = imgui.input_float('Jog speed (deg/s)',e.cfg['sweep']['jog_speed_deg_s'],5,30,'%.1f')
        if changed: self.act(e.set_jog_speed,value)
        imgui.text_disabled('Duration applies to Go-to and A -> B.')
        imgui.text_disabled('Save A/B writes actual angles to YAML.')
        imgui.separator()
        imgui.text('Joint targets (degrees)')
        if imgui.button('Zero arm',(171,28)): self.act(e.zero_arm)
        imgui.same_line()
        if imgui.button('Randomize joints',(171,28)): self.act(e.randomize_joints)
        for i,j in enumerate(e.cfg['joints']):
            imgui.push_id(i)
            imgui.set_next_item_width(290)
            changed,value = imgui.slider_float(j['name'],float(np.rad2deg(e.target[i])),*j['limits_deg'],format='%.1f deg')
            if changed: self.act(e.set_target,i,value)
            imgui.pop_id()
        imgui.text_disabled('Ctrl-click a slider to type an exact angle.')
        if imgui.collapsing_header('Advanced: per-joint P / D'):
            imgui.set_next_item_width(170)
            _,self.selected = imgui.combo('Motor',self.selected,[j['name']+' / RobStride '+j['motor'] for j in e.cfg['joints']])
            j=e.cfg['joints'][self.selected]
            imgui.set_next_item_width(170)
            changed,p = imgui.input_float('P (Nm/rad)',j['kp_nm_rad'],5,20,'%.2f')
            if changed: self.act(e.set_gains,self.selected,p,j['kd_nm_s_rad'])
            imgui.set_next_item_width(170)
            changed,d = imgui.input_float('D (Nm s/rad)',j['kd_nm_s_rad'],1,5,'%.2f')
            if changed: self.act(e.set_gains,self.selected,j['kp_nm_rad'],d)
            imgui.text_wrapped('Applied live. Simulation PD gains, not firmware PID settings. Keep gains fixed when comparing link lengths.')
        if imgui.button('Save controls',(171,28)): self.act(e.save_settings)
        imgui.same_line()
        if imgui.button('Reload YAML',(171,28)): self.act(self.reload)
        imgui.end_disabled()
        if imgui.button('Edit all parameters',(171,28)): open_editor(e.path)
        imgui.same_line()
        imgui.begin_disabled(e.last_report is None)
        if imgui.button('Open last report',(171,28)): self.report()
        imgui.end_disabled()
        imgui.separator()
        imgui.text_wrapped(e.message)
        if imgui.collapsing_header('Saved poses / actual angles'):
            for name in ('A','B'):
                imgui.text_wrapped(name+': '+', '.join(f'{x:.1f}' for x in e.cfg['sweep'][name.lower()+'_deg'])+' deg')
            imgui.text_wrapped('Actual: '+', '.join(f'{x:.1f}' for x in np.rad2deg(e.data.qpos))+' deg')
        imgui.text_wrapped('Camera: drag to rotate, right-drag to pan, wheel to zoom.')
        imgui.end()

    def render_arm(self, width, height, sx, sy):
        e = self.experiment
        left = int(PANEL_WIDTH*sx)
        right = int(chart_width(width/sx)*sx)
        viewport = mujoco.MjrRect(left,0,max(1,width-left-right),height)
        # Refresh render transforms without altering logged pre-step samples.
        mujoco.mj_forward(e.arm.model,e.data)
        mujoco.mjv_updateScene(e.arm.model,e.data,self.option,None,self.camera,mujoco.mjtCatBit.mjCAT_ALL,self.scene)
        mujoco.mjr_render(viewport,self.scene,self.context)
        mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_TOPLEFT,viewport,
                           f'{e.state} | Move {e.cfg["sweep"]["duration_s"]:g}s | Jog {e.cfg["sweep"]["jog_speed_deg_s"]:g} deg/s',
                           'Contacts '+('ON' if e.cfg['simulation']['contacts'] else 'OFF'),self.context)

    def run(self):
        if not glfw.init(): raise RuntimeError('Could not initialize GLFW.')
        # MuJoCo's renderer uses compatibility OpenGL; ImGui's OpenGL2 backend
        # shares that context instead of requiring an incompatible core profile.
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR,2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR,1)
        window=glfw.create_window(1280,900,'OpenArm - MuJoCo',None,None)
        if not window:
            glfw.terminate();raise RuntimeError('Could not create the MuJoCo window.')
        glfw.set_window_size_limits(window,1180,780,glfw.DONT_CARE,glfw.DONT_CARE)
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        imgui.create_context()
        imgui.get_io().set_ini_filename(None)
        imgui.get_io().config_flags |= imgui.ConfigFlags_.nav_enable_keyboard
        imgui.style_colors_dark()
        imgui.get_style().font_scale_main=1.15
        self.address=ctypes.cast(window,ctypes.c_void_p).value
        imgui.backends.glfw_init_for_opengl(self.address,False)
        imgui.backends.opengl2_init()
        glfw.set_key_callback(window,self.key)
        glfw.set_cursor_pos_callback(window,self.mouse_move)
        glfw.set_mouse_button_callback(window,self.mouse_button)
        glfw.set_scroll_callback(window,self.scroll)
        glfw.set_char_callback(window,lambda w,c:imgui.backends.glfw_char_callback(self.address,c))
        glfw.set_window_focus_callback(window,lambda w,f:imgui.backends.glfw_window_focus_callback(self.address,f))
        glfw.set_cursor_enter_callback(window,lambda w,e:imgui.backends.glfw_cursor_enter_callback(self.address,e))
        self.render_resources()
        previous=time.monotonic();credit=0.0
        try:
            while not glfw.window_should_close(window):
                now=time.monotonic();credit+=min(.1,now-previous);previous=now
                glfw.poll_events()
                imgui.backends.opengl2_new_frame();imgui.backends.glfw_new_frame();imgui.new_frame()
                logical_width,logical_height=glfw.get_window_size(window)
                width,height=glfw.get_framebuffer_size(window)
                self.panel(logical_height)
                if self.reload_requested:
                    self.render_resources();self.reload_requested=False;credit=0
                e=self.experiment
                if not e.paused:
                    nsteps=int(credit/e.arm.model.opt.timestep)
                    credit-=nsteps*e.arm.model.opt.timestep
                    for _ in range(nsteps):
                        try: e.tick()
                        except RuntimeError: break
                    e.sample_history()
                else: credit=0
                if width>0 and height>0 and logical_width>0 and logical_height>0:
                    draw_torque_charts(e,logical_width,logical_height)
                    self.render_arm(width,height,width/logical_width,height/logical_height)
                    # MuJoCo uses multiple fixed-function texture units; ImGui's
                    # OpenGL2 backend expects unit 0 and an identity texture matrix.
                    for unit in (GL.GL_TEXTURE1, GL.GL_TEXTURE0):
                        GL.glActiveTexture(unit)
                        GL.glDisable(GL.GL_TEXTURE_2D)
                        GL.glDisable(GL.GL_TEXTURE_GEN_S)
                        GL.glDisable(GL.GL_TEXTURE_GEN_T)
                        GL.glMatrixMode(GL.GL_TEXTURE)
                        GL.glLoadIdentity()
                    GL.glClientActiveTexture(GL.GL_TEXTURE0)
                    GL.glDisable(GL.GL_FOG)
                    GL.glDisable(GL.GL_ALPHA_TEST)
                    imgui.render();imgui.backends.opengl2_render_draw_data(imgui.get_draw_data())
                    glfw.swap_buffers(window)
                else:
                    imgui.end_frame();glfw.wait_events_timeout(.05)
        finally:
            if self.experiment.run:
                self.experiment.stop('Window closed during sweep.')
            imgui.backends.opengl2_shutdown();imgui.backends.glfw_shutdown()
            imgui.destroy_context()
            self.context.free();self.scene = None
            glfw.destroy_window(window);glfw.terminate()


def launch(config_path,out):
    Window(Path(config_path).resolve(),Path(out).resolve()).run()
