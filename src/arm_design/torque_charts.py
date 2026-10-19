"""Seven persistent torque plots drawn with the existing native UI toolkit."""
from imgui_bundle import imgui
import numpy as np
from .simulation import rating


def chart_width(window_width):
    return max(360, min(500, window_width*.32))


def draw_torque_charts(experiment, width, height):
    panel_width = chart_width(width)
    origin = width-panel_width
    imgui.set_next_window_pos((origin,0))
    imgui.set_next_window_size((panel_width,height))
    flags = (imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_resize |
             imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_title_bar |
             imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.no_scrollbar |
             imgui.WindowFlags_.no_scroll_with_mouse)
    imgui.begin('Joint torque charts',flags=flags)
    draw = imgui.get_window_draw_list()
    color = lambda rgba: imgui.get_color_u32(imgui.ImVec4(*rgba))
    text = color((.88,.91,.96,1)); muted = color((.57,.64,.74,1))
    blue = color((.35,.75,1,1)); yellow = color((1,.78,.25,1)); red = color((1,.35,.4,1))
    grid = color((.24,.28,.35,1))
    pane = color((.09,.115,.15,1))
    yellow_band = color((.27,.22,.10,1)); red_band = color((.29,.105,.13,1))
    draw.add_text((origin+12,8),text,'Applied joint torque (Nm)')
    draw.add_text((origin+12,27),yellow,'Dotted: +/- nominal')
    draw.add_text((origin+190,27),red,'+/- maximum')
    draw.add_text((origin+12,46),muted,'Configured ratings | last 10 seconds')

    nominal = rating(experiment.cfg,'rated_torque_nm')
    maximum = rating(experiment.cfg,'peak_torque_nm')
    history = list(experiment.history)
    times = np.array([r[0] for r in history])
    torques = np.array([r[1] for r in history]) if history else np.empty((0,7))
    end = max(1,float(experiment.data.time))
    start = max(0,end-10)
    row_height = (height-112)/7
    x0,x1 = origin+48,width-12
    tick_x = np.linspace(x0,x1,6)

    for i,joint in enumerate(experiment.cfg['joints']):
        top = 70+i*row_height
        y0,y1 = top+23,top+row_height-6
        values = torques[:,i]
        current = float(values[-1]) if len(values) else float(experiment.data.qfrc_actuator[i])
        bound = max(maximum[i]*1.2,float(np.max(np.abs(values)))*1.1 if len(values) else 0)
        y = lambda value: y0+(bound-value)/(2*bound)*(y1-y0)
        status_color = red if abs(current)>maximum[i] else yellow if abs(current)>nominal[i] else blue
        draw.add_text((origin+12,top),text,joint['name'])
        draw.add_text((origin+48,top),status_color,f'{current:+.2f} Nm')
        draw.add_text((origin+175,top),yellow,f'Nom {nominal[i]:g}')
        draw.add_text((origin+270,top),red,f'Max {maximum[i]:g}')
        draw.add_rect_filled((x0,y0),(x1,y1),pane)
        # Magnitude bands apply equally to positive and negative output torque.
        for low,high,shade in ((maximum[i],bound,red_band),
                               (-bound,-maximum[i],red_band),
                               (nominal[i],maximum[i],yellow_band),
                               (-maximum[i],-nominal[i],yellow_band)):
            draw.add_rect_filled((x0,y(high)),(x1,y(low)),shade)
        for x in tick_x:
            draw.add_line((x,y0),(x,y1),grid)
        for value in (-bound,0,bound):
            yy=y(value)
            draw.add_line((x0,yy),(x1,yy),grid)
            label=f'{value:.1f}' if value else '0'
            draw.add_text((x0-imgui.calc_text_size(label).x-5,yy-7),muted,label)
        for limit,line_color in ((nominal[i],yellow),(maximum[i],red)):
            for sign in (-1,1):
                for x in np.arange(x0,x1,6):
                    draw.add_circle_filled((float(x),y(sign*limit)),.85,line_color)
        # Clip to this plot so a new sample/roundoff cannot draw into a neighbor.
        draw.push_clip_rect((x0,y0),(x1,y1),True)
        if len(values):
            points=[(x0+(float(t)-start)/(end-start)*(x1-x0),y(float(v)))
                    for t,v in zip(times,values)]
            for a,b in zip(points,points[1:]):
                draw.add_line(a,b,blue,1.7)
            draw.add_circle_filled(points[-1],2.2,status_color)
        draw.pop_clip_rect()

    for x,t in zip(tick_x,np.linspace(start,end,6)):
        label=f'{t:.1f}'
        draw.add_text((min(width-12-imgui.calc_text_size(label).x,x-imgui.calc_text_size(label).x/2),height-40),muted,label)
    draw.add_text((origin+12,height-20),muted,'Simulation time (s)')
    imgui.end()
