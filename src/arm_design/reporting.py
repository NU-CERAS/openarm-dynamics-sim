"""Small, self-contained sweep record: CSV, YAML, MJCF, PNG and HTML."""
from datetime import datetime, timezone
from pathlib import Path
import csv
import html
import json
import numpy as np
import yaml
from .simulation import rating


def summary(run):
    task = [r for r in run.rows if r['phase'] in ('sweep','hold')]
    c = run.cfg
    if not task:
        return dict(completed=False, tracking_ok=False, direction=run.direction, reason=run.reason or 'No sweep samples', joints=[])
    torque = np.array([r['torque'] for r in task])
    velocity = np.array([r['velocity'] for r in task])
    error = np.rad2deg(np.array([r['q']-r['target'] for r in task]))
    cap = np.array([r['cap'] for r in task])
    rated = np.array([r['rated_cap'] for r in task])
    dt = c['simulation']['step_s']
    bounds = np.deg2rad([j['limits_deg'] for j in c['joints']])
    q = np.array([r['q'] for r in task])
    limits_ok = bool(np.all(q >= bounds[:,0] - np.deg2rad(.5)) and np.all(q <= bounds[:,1] + np.deg2rad(.5)))
    speed_ok = bool(np.all(np.abs(velocity)*30/np.pi <= rating(c,'no_load_speed_rpm')))
    result = dict(completed=run.complete, reason=run.reason, tracking_ok=bool(run.complete and np.max(np.abs(error)) <= c['sweep']['tracking_tolerance_deg']), joint_limits_ok=limits_ok, below_no_load_speed=speed_ok, duration_s=c['sweep']['duration_s'], measured_window='sweep + final hold; approach and settling excluded', contacts_enabled=c['simulation']['contacts'], maximum_contacts=max(r['contacts'] for r in task), joints=[])
    result['direction'] = run.direction
    for i,j in enumerate(c['joints']):
        saturated = (cap[:,i] > 1e-8) & (np.abs(torque[:,i]) >= .99*cap[:,i])
        above = np.abs(torque[:,i]) > rated[:,i]
        longest = current = 0
        for x in above:
            current = current+1 if x else 0
            longest = max(current,longest)
        result['joints'].append(dict(joint=j['name'], peak_torque_nm=float(np.max(np.abs(torque[:,i]))), rms_torque_nm=float(np.sqrt(np.mean(torque[:,i]**2))), peak_speed_rpm=float(np.max(np.abs(velocity[:,i]))*30/np.pi), peak_mechanical_power_w=float(np.max(np.abs(torque[:,i]*velocity[:,i]))), max_tracking_error_deg=float(np.max(np.abs(error[:,i]))), final_error_deg=float(error[-1,i]), seconds_above_rated_proxy=float(np.sum(above)*dt), longest_above_rated_proxy_s=float(longest*dt), saturated_percent=float(np.mean(saturated)*100)))
    return result


def save_result(arm, run, out):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-%fZ')
    folder = Path(out).resolve() / stamp
    folder.mkdir(parents=True)
    (folder/'config.yaml').write_text(yaml.safe_dump(run.cfg,sort_keys=False))
    # Gains can be edited live without rebuilding the viewer. Export the model
    # corresponding to the run snapshot, rather than stale launch-time XML.
    from .model import build_arm
    (folder/'model.xml').write_text(build_arm(run.cfg).xml)
    result = summary(run)
    (folder/'summary.json').write_text(json.dumps(result,indent=2))
    fields = ['time_s','phase','contacts','tip_x_m','tip_y_m','tip_z_m']
    metrics = ['angle_deg','target_deg','speed_rad_s','torque_nm','peak_limit_nm','rated_proxy_nm','mechanical_power_w']
    fields += [f'{j["name"]}_{key}' for j in run.cfg['joints'] for key in metrics]
    with (folder/'loads.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(fields)
        for r in run.rows:
            row = [r['time_s'],r['phase'],r['contacts'],*r['tip']]
            for i in range(len(run.cfg['joints'])):
                row += [np.rad2deg(r['q'][i]),np.rad2deg(r['target'][i]),r['velocity'][i],r['torque'][i],r['cap'][i],r['rated_cap'][i],r['torque'][i]*r['velocity'][i]]
            writer.writerow(row)
    if run.rows:
        time = np.array([r['time_s'] for r in run.rows]) - run.sweep_start
        count = len(run.cfg['joints'])
        fig,axes=plt.subplots(count,2,figsize=(13,max(4,count*2.15)),sharex=True,layout='constrained',squeeze=False)
        for i,j in enumerate(run.cfg['joints']):
            ax = axes[i,0]
            ax.plot(time,[r['torque'][i] for r in run.rows],label='Applied torque')
            for sign in (-1,1):
                ax.plot(time,[sign*r['cap'][i] for r in run.rows],color='red',linestyle=':',label='Peak proxy' if sign==1 else None)
                ax.plot(time,[sign*r['rated_cap'][i] for r in run.rows],color='gray',linestyle='--',label='Rated proxy' if sign==1 else None)
            ax.set_ylabel(j['name']+' / Nm')
            axes[i,1].plot(time,[np.rad2deg(r['q'][i]) for r in run.rows],label='Actual')
            axes[i,1].plot(time,[np.rad2deg(r['target'][i]) for r in run.rows],linestyle='--',label='Requested')
            axes[i,1].set_ylabel('Degrees')
            for a in axes[i]:
                a.axvline(0,color='green',alpha=.5)
                a.axvline(run.cfg['sweep']['duration_s'],color='green',alpha=.5)
                a.grid(alpha=.2)
        axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=8)
        for ax in axes[-1]: ax.set_xlabel(f'Seconds relative to {run.direction} start (negative = approach/settle)')
        fig.suptitle(f'MuJoCo {run.direction}: applied joint torque and motion tracking')
        fig.savefig(folder/'loads.png',dpi=120);plt.close(fig)
    status = 'Tracking within tolerance' if result['tracking_ok'] else 'Incomplete or tracking outside tolerance'
    rows=''.join('<tr>'+''.join(f'<td>{html.escape(str(v)) if isinstance(v,str) else f"{v:.3f}"}</td>' for v in (j['joint'],j['peak_torque_nm'],j['rms_torque_nm'],j['max_tracking_error_deg'],j['seconds_above_rated_proxy'],j['saturated_percent']))+'</tr>' for j in result['joints'])
    (folder/'report.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>Arm sweep</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 16px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}img{{width:100%}}</style>
<h1>{status}</h1><p>{html.escape(run.direction)}. {html.escape(run.reason)} Duration: {run.cfg['sweep']['duration_s']:g} s. Summary covers sweep + final hold.</p>
<p>Applied motor torque, not unlimited ideal demand. Inspect tracking error and saturation together. Rated/peak envelopes are assumptions; RMS is not a thermal certification. Contacts: {'enabled (primitive approximations)' if run.cfg['simulation']['contacts'] else 'disabled'}.</p>
<table><tr><th>Joint</th><th>Peak Nm</th><th>RMS Nm</th><th>Max error °</th><th>Above rated s</th><th>Saturated %</th></tr>{rows}</table>
<p><a href="loads.csv">CSV</a> · <a href="config.yaml">Parameters</a> · <a href="model.xml">MuJoCo model</a> · <a href="summary.json">Full summary</a></p><img src="loads.png" alt="Joint torque and motion tracking">
''')
    return folder
