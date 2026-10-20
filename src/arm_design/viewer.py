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
from .model import structure
from .torque_charts import chart_width, draw_torque_charts
from .fea import METRICS

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
        self.geometry_draft = self.experiment.geometry_settings()
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.lookat[:] = [0,0,.45]
        self.camera.distance = 1.5
        self.camera.azimuth, self.camera.elevation = 135,-15
        if self.experiment.cfg.get('model_type') == 'urdf':
            self.camera.lookat[:] = self.experiment.arm.model.stat.center
            self.camera.distance = max(.4, self.experiment.arm.model.stat.extent*1.8)
        self.option = mujoco.MjvOption()
        self.scene = self.context = None
        self.drag_button = None
        self.cursor = (0,0)
        self.reload_requested = False
        self.closed = False
        self.fea_body = 0
        self.fea_interface = 0
        self.fea_preview = None
        self.render_model = None

    def render_resources(self, model=None):
        model = self.experiment.arm.model if model is None else model
        if self.context is not None:
            self.context.free()
        self.scene = None
        self.scene = mujoco.MjvScene(model,maxgeom=2000)
        self.context = mujoco.MjrContext(model,mujoco.mjtFontScale.mjFONTSCALE_150)
        self.render_model = model

    def act(self, action, *args):
        try:
            action(*args)
        except (ValueError,KeyError,TypeError,RuntimeError,OSError,yaml.YAMLError) as exc:
            self.experiment.message = str(exc)

    def reload(self):
        self.experiment.idle_required()
        self.experiment.load()
        self.selected = min(self.selected,len(self.experiment.cfg['joints'])-1)
        self.reload_requested = True
        self.fea_preview = None
        self.experiment.message = 'Reloaded YAML. Initialized at saved pose A.'

    def report(self):
        if self.experiment.last_report:
            webbrowser.open(self.experiment.last_report.as_uri())

    def apply_geometry(self, save):
        self.experiment.apply_geometry(self.geometry_draft,save=save)
        self.reload_requested = True
        self.fea_preview = None

    def preview_fea(self, index):
        recording = self.experiment.fea
        if recording is None or recording.active:
            return
        self.experiment.paused = True
        row = recording.rows[index]
        data = mujoco.MjData(recording.model)
        data.qpos[:] = row['q_rad']
        data.qvel[:] = row['qvel_rad_s']
        data.ctrl[:] = row['target_rad']
        mujoco.mj_forward(recording.model, data)
        self.fea_preview = (recording, index, data)

    def fea_panel(self):
        e = self.experiment
        if not imgui.collapsing_header('FEA data collection tools'):
            return
        recording = e.fea
        active = recording is not None and recording.active
        label = 'Stop recording' if active else ('New recording (replace)' if recording else 'Start recording')
        if imgui.button(label, (350, 30)):
            self.act(e.stop_fea if active else e.start_fea)
            if not active:
                self.fea_preview = None
                self.fea_body = self.fea_interface = 0
            recording = e.fea
            active = recording is not None and recording.active
        imgui.text_wrapped('Record while jogging, moving to a pose, or sweeping. Stopping recording does not stop motion.')
        if recording is None:
            return
        imgui.text_colored((1,.4,.3,1) if active else (.4,.8,1,1),
                           ('RECORDING' if active else 'STOPPED') +
                           f' | {recording.duration:.2f}s | {len(recording.rows)} samples')
        imgui.text_disabled(f'{1/recording.model.opt.timestep:g} Hz / capacity {recording.max_samples} samples')
        if active:
            if e.paused:
                imgui.text_wrapped('Simulation paused: no new samples until Resume.')
            imgui.text_wrapped('Stop recording to inspect peaks and export. Controller gains are locked while recording.')
            return
        imgui.text_wrapped(recording.reason)
        if not recording.rows:
            imgui.text_wrapped('No samples. Resume simulation during the next recording.')
            return
        imgui.text('Maximum observed loads')
        imgui.text_wrapped('Forces in N; moments in Nm. Each connection is shown separately in the receiving link frame.')
        # Put moving links first, retaining fixed motor/tool bodies for their
        # own structural analyses.
        primary = list(dict.fromkeys(int(i) for i in recording.model.jnt_bodyid))
        bodies = primary + [i for i in recording.body_ids if i not in primary]
        self.fea_body = min(self.fea_body, len(bodies)-1)
        imgui.set_next_item_width(245)
        changed, self.fea_body = imgui.combo('Link / body', self.fea_body,
                                           [recording.names[i] for i in bodies])
        if changed:
            self.fea_interface = 0
        body = bodies[self.fea_body]
        interfaces = [i for i, x in enumerate(recording.interfaces) if x['body_id'] == body]
        self.fea_interface = min(self.fea_interface, len(interfaces)-1)
        imgui.set_next_item_width(245)
        _, self.fea_interface = imgui.combo('Connection', self.fea_interface,
            [recording.names[recording.interfaces[i]['other_body_id']] for i in interfaces])
        i = interfaces[self.fea_interface]
        if imgui.begin_table('FEA extrema', 3, imgui.TableFlags_.borders | imgui.TableFlags_.row_bg):
            for name in ('Load', 'Minimum', 'Maximum'):
                imgui.table_setup_column(name)
            imgui.table_headers_row()
            for j, metric in enumerate(METRICS):
                imgui.table_next_row()
                imgui.table_set_column_index(0)
                imgui.text(metric)
                for column, values, indices in (
                    (1, recording.minimum, recording.min_index),
                    (2, recording.maximum, recording.max_index)):
                    imgui.table_set_column_index(column)
                    index = int(indices[i,j])
                    if imgui.small_button(f'{values[i,j]:.4g}##peak{j}_{column}'):
                        self.preview_fea(index)
                    if imgui.is_item_hovered():
                        imgui.set_tooltip(f"Inspect pose at {recording.rows[index]['time_s']:.4f}s")
            imgui.end_table()
        imgui.text_wrapped('Click a value to inspect its recorded pose. Different peaks are separate load cases.')
        if self.fea_preview is not None and self.fea_preview[0] is recording:
            _, index, _ = self.fea_preview
            row = recording.rows[index]
            imgui.text_wrapped(f"Inspecting sample {index} at {row['time_s']:.4f}s | {row['phase']}")
            selected = index in recording.selected_samples
            if imgui.button('Remove load case' if selected else 'Select load case'):
                if selected:
                    recording.selected_samples.remove(index)
                else:
                    recording.selected_samples.add(index)
            imgui.same_line()
            if imgui.button('Return to live pose'):
                self.fea_preview = None
            imgui.text_wrapped('Pose inspection pauses motion; Resume returns to live simulation.')
        imgui.text_wrapped(f'Max tracking error: {recording.max_tracking_error_deg:.2f} deg | '
                          f'Saturated samples: {recording.saturated_samples} | '
                          f'Contact samples: {recording.contact_samples}')
        imgui.text_wrapped('Loads only; stress, deflection and safety factors require FEA.')
        if imgui.button('Export peak summary', (350, 26)):
            self.act(e.export_fea, 'peaks')
        cases_label = (f'Export {len(recording.selected_samples)} selected load cases'
                       if recording.selected_samples else 'Export all peak load cases')
        if imgui.button(cases_label, (350, 26)):
            self.act(e.export_fea, 'cases')
        if imgui.button('Export full recording + load cases', (350, 26)):
            self.act(e.export_fea, 'all')
        if recording.last_export:
            imgui.text_wrapped('Saved to '+str(recording.last_export))

    def geometry_popup(self):
        imgui.set_next_window_size((540,0))
        opened,_ = imgui.begin_popup_modal('Link lengths / weights',flags=imgui.WindowFlags_.always_auto_resize)
        if not opened:
            return
        e=self.experiment
        imgui.text_wrapped('Link mass includes tube and bracket, excluding motors. End effector and carried payload are separate.')
        imgui.separator()
        imgui.text('Link');imgui.same_line(110);imgui.text('Length (mm)')
        imgui.same_line(245);imgui.text('Mass (kg)')
        imgui.begin_disabled(e.busy)
        for i,edit in enumerate(self.geometry_draft['links']):
            imgui.push_id(i)
            imgui.text(f'J{i+1} -> '+(f'J{i+2}' if i<6 else 'tool'))
            imgui.same_line(110);imgui.set_next_item_width(120)
            _,edit['length_mm']=imgui.input_float('##length',edit['length_mm'],5,25,'%.1f')
            auto=edit['mass_override_kg'] is None
            link=dict(e.cfg['links'][i],length_mm=edit['length_mm'],mass_override_kg=None)
            estimated=structure(link)[0] if np.isfinite(edit['length_mm']) and edit['length_mm']>0 else 0.0
            mass=estimated if auto else edit['mass_override_kg']
            imgui.same_line(245);imgui.set_next_item_width(120)
            imgui.begin_disabled(auto)
            changed,mass=imgui.input_float('##mass',mass,.01,.1,'%.3f')
            if changed: edit['mass_override_kg']=mass
            imgui.end_disabled()
            imgui.same_line(380)
            changed,auto=imgui.checkbox('Auto mass',auto)
            if changed: edit['mass_override_kg']=None if auto else estimated
            imgui.pop_id()
        imgui.text_wrapped('Auto assumes hollow tubes + brackets from YAML, not solid bars. For solid or machined links, uncheck Auto mass and enter a CAD or measured mass. Inertia still uses the estimated shape.')
        if any(l['com_override_m'] is not None or l['inertia_override_kg_m2'] is not None for l in e.cfg['links']):
            imgui.text_wrapped('CAD COM/inertia overrides are present and remain fixed. Update them in YAML when changing geometry.')
        imgui.separator()
        imgui.text('Actuator masses (kg) - per joint')
        overrides=self.geometry_draft['motor_mass_overrides_kg']
        for i,joint in enumerate(e.cfg['joints']):
            if i%2: imgui.same_line(275)
            imgui.push_id('motor_mass_'+joint['name'])
            mass=overrides[joint['name']]
            if mass is None: mass=e.cfg['motors'][joint['motor']]['mass_kg']
            imgui.set_next_item_width(125)
            changed,mass=imgui.input_float(joint['name']+' / '+joint['motor'],mass,.01,.1,'%.3f')
            if changed: overrides[joint['name']]=mass
            imgui.pop_id()
        if imgui.button('Use catalog motor masses'):
            self.geometry_draft['motor_mass_overrides_kg']={j['name']:None for j in e.cfg['joints']}
        imgui.text_wrapped('Changes housing mass and inertia; motor torque ratings and controller gains stay the same.')
        imgui.separator()
        imgui.set_next_item_width(150)
        _,self.geometry_draft['tool_mass_kg']=imgui.input_float('End effector (kg)',self.geometry_draft['tool_mass_kg'],.05,.1,'%.3f')
        imgui.set_next_item_width(150)
        _,self.geometry_draft['payload_mass_kg']=imgui.input_float('Payload (kg)',self.geometry_draft['payload_mass_kg'],.05,.1,'%.3f')
        imgui.text_wrapped('Apply rebuilds geometry, mass and inertia; clears the charts; and pauses at the current joint angles. Saved poses and gains are kept.')
        if imgui.button('Apply',(150,28)): self.act(self.apply_geometry,False)
        imgui.same_line()
        if imgui.button('Apply + save',(150,28)): self.act(self.apply_geometry,True)
        imgui.end_disabled()
        imgui.same_line()
        if imgui.button('Close',(150,28)): imgui.close_current_popup()
        imgui.text_wrapped(e.message)
        imgui.end_popup()

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
            if glfw.KEY_1 <= key <= min(glfw.KEY_9,glfw.KEY_1+len(e.cfg['joints'])-1):
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
        imported = e.cfg.get('model_type') == 'urdf'
        imgui.text_wrapped(e.cfg['name']+' / MuJoCo')
        if imported:
            imgui.text_wrapped('URDF: '+Path(e.cfg['urdf']['path']).name)
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
        if imgui.button('Run A -> B sweep',(171,32)): self.act(e.start_sweep, 'A -> B')
        imgui.same_line()
        if imgui.button('Run B -> A sweep',(171,32)): self.act(e.start_sweep, 'B -> A')
        imgui.end_disabled()
        if imgui.button('Resume' if e.paused else 'Pause',(171,28)): e.paused=not e.paused
        imgui.same_line()
        if imgui.button('Stop / hold',(171,28)): self.act(e.stop)
        imgui.separator()
        self.fea_panel()
        imgui.separator()
        imgui.begin_disabled(e.busy)
        imgui.set_next_item_width(170)
        changed,value = imgui.input_float('Move duration (s)',e.cfg['sweep']['duration_s'],.5,1,'%.2f')
        if changed: self.act(e.set_duration,value)
        imgui.set_next_item_width(170)
        changed,value = imgui.input_float('Jog speed (deg/s)',e.cfg['sweep']['jog_speed_deg_s'],5,30,'%.1f')
        if changed: self.act(e.set_jog_speed,value)
        imgui.text_disabled('Duration applies to Go-to and both sweeps.')
        imgui.text_disabled('Save A/B writes actual angles to YAML.')
        if imported:
            imgui.text_wrapped('Geometry and inertia come from the URDF. Edit it, then Reload YAML to reimport.')
            if imgui.button('Edit source URDF',(350,30)): open_editor(Path(e.cfg['urdf']['path']))
        else:
            if imgui.button('Link lengths / weights',(350,30)):
                self.geometry_draft=e.geometry_settings()
                imgui.open_popup('Link lengths / weights')
            self.geometry_popup()
        imgui.separator()
        imgui.text('Joint targets (degrees)')
        if imgui.button('Zero arm',(171,28)): self.act(e.zero_arm)
        imgui.same_line()
        if imgui.button('Randomize joints',(171,28)): self.act(e.randomize_joints)
        for i,j in enumerate(e.cfg['joints']):
            imgui.push_id(i)
            # Imported joint names can be much longer than J1/J2.
            label_width = imgui.calc_text_size(j['name']).x + imgui.get_style().item_inner_spacing.x
            imgui.set_next_item_width(max(120, min(290, imgui.get_content_region_avail().x-label_width)))
            changed,value = imgui.slider_float(j['name'],float(np.rad2deg(e.target[i])),*j['limits_deg'],format='%.1f deg')
            if changed: self.act(e.set_target,i,value)
            imgui.pop_id()
        imgui.text_disabled('Ctrl-click a slider to type an exact angle.')
        if imgui.collapsing_header('Advanced: per-joint P / D'):
            imgui.begin_disabled(e.fea is not None and e.fea.active)
            imgui.set_next_item_width(170)
            _,self.selected = imgui.combo('Joint' if imported else 'Motor',self.selected,
                                        [j['name'] if imported else j['name']+' / RobStride '+j['motor'] for j in e.cfg['joints']])
            j=e.cfg['joints'][self.selected]
            imgui.set_next_item_width(170)
            changed,p = imgui.input_float('P (Nm/rad)',j['kp_nm_rad'],5,20,'%.2f')
            if changed: self.act(e.set_gains,self.selected,p,j['kd_nm_s_rad'])
            imgui.set_next_item_width(170)
            changed,d = imgui.input_float('D (Nm s/rad)',j['kd_nm_s_rad'],1,5,'%.2f')
            if changed: self.act(e.set_gains,self.selected,j['kp_nm_rad'],d)
            imgui.text_wrapped('Applied live. Simulation PD gains, not firmware PID settings. Keep gains fixed when comparing models.')
            imgui.end_disabled()
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
        if not e.paused:
            self.fea_preview = None
        if self.fea_preview is not None:
            recording, index, data = self.fea_preview
            model = recording.model
            label = f"FEA recorded pose | t={recording.rows[index]['time_s']:.4f}s"
        else:
            model, data = e.arm.model, e.data
            label = f'{e.state} | Move {e.cfg["sweep"]["duration_s"]:g}s | Jog {e.cfg["sweep"]["jog_speed_deg_s"]:g} deg/s'
        if self.render_model is not model:
            self.render_resources(model)
        left = int(PANEL_WIDTH*sx)
        right = int(chart_width(width/sx)*sx)
        viewport = mujoco.MjrRect(left,0,max(1,width-left-right),height)
        # Refresh render transforms without altering logged pre-step samples.
        mujoco.mj_forward(model,data)
        mujoco.mjv_updateScene(model,data,self.option,None,self.camera,mujoco.mjtCatBit.mjCAT_ALL,self.scene)
        mujoco.mjr_render(viewport,self.scene,self.context)
        mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_TOPLEFT,viewport,
                           label, 'FEA pose inspection' if self.fea_preview else
                           'Contacts '+('ON' if e.cfg['simulation']['contacts'] else 'OFF'),self.context)

    def run(self):
        if not glfw.init(): raise RuntimeError('Could not initialize GLFW.')
        # MuJoCo's renderer uses compatibility OpenGL; ImGui's OpenGL2 backend
        # shares that context instead of requiring an incompatible core profile.
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR,2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR,1)
        window=glfw.create_window(1280,900,self.experiment.cfg['name']+' - MuJoCo',None,None)
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
            if self.experiment.fea and self.experiment.fea.rows:
                if self.experiment.fea.active:
                    self.experiment.fea.stop('Window closed during recording.')
                if self.experiment.fea.last_export is None:
                    try:
                        self.experiment.export_fea()
                    except OSError as exc:
                        print(f'Could not save FEA recording on exit: {exc}', file=sys.stderr)
            imgui.backends.opengl2_shutdown();imgui.backends.glfw_shutdown()
            imgui.destroy_context()
            self.context.free();self.scene = None
            glfw.destroy_window(window);glfw.terminate()


def launch(config_path,out):
    Window(Path(config_path).resolve(),Path(out).resolve()).run()
