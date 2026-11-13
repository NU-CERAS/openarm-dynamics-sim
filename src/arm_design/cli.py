import argparse
from pathlib import Path
from .config import DEFAULT_CONFIG, ROOT, load_config, validate


def main():
    p=argparse.ArgumentParser(description='MuJoCo joint jogging and timed A/B sweeps')
    p.add_argument('command',nargs='?',default='view',choices=['view','sweep','export','campaign','bounds','fea-cases','watch','restrained','dashboard','replay'])
    source=p.add_mutually_exclusive_group()
    source.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    source.add_argument('--urdf',type=Path,help='Import a fixed-base rotary robot; creates <robot>.sim.yaml beside the URDF')
    p.add_argument('--package-root',type=Path,action='append',default=[],help='Folder containing ROS packages, or a package itself (repeatable; used with --urdf)')
    p.add_argument('--out',type=Path,default=ROOT/'results')
    p.add_argument('--duration',type=float,help='A->B seconds (headless sweep only; viewer uses [/] or YAML)')
    p.add_argument('--direction',choices=['a-to-b','b-to-a'],default='a-to-b',
                   help='Direction for headless sweeps (default: a-to-b)')
    p.add_argument('--samples',type=int,help='Campaign: random A/B sweeps (default 200). Bounds: random poses (default 2000)')
    p.add_argument('--iterations',type=int,help='Refinement steps per connection force and moment (campaign default 20 sweeps, bounds 300 poses)')
    p.add_argument('--min-duration',type=float,default=.1,help='Campaign: shortest A->B seconds (default 0.1)')
    p.add_argument('--max-duration',type=float,default=5.,help='Campaign: longest A->B seconds (default 5)')
    p.add_argument('--workers',type=int,help='Campaign: parallel processes (default: CPU count)')
    p.add_argument('--seed',type=int,default=0,help='Campaign: random seed')
    p.add_argument('--aggressive', action='store_true', help='Campaign: search progressively faster random A/B moves in both directions; samples counts all attempts')
    p.add_argument('--search-steps', type=int, default=8, help='Aggressive: duration attempts per direction (default 8)')
    p.add_argument('--skip-timestep-validation', action='store_true', help='Campaign: skip additional finer-timestep reruns after the trial budget')
    p.add_argument('--include-failed',action='store_true',help='Campaign: also count sweeps that exceed tracking tolerance (joint-stop impacts)')
    p.add_argument('--max-overlap-mm',type=float,default=0.,help='Campaign: largest convex-hull overlap allowed before a pose counts as self-colliding (default 0)')
    p.add_argument('--allow-self-collision',action='store_true',help='Campaign/bounds: skip the self-collision check')
    p.add_argument('--derated',action='store_true',help='Bounds: cap motors at the YAML torque derating instead of full peak torque')
    p.add_argument('--min-lever-m',type=float,default=.05,help='Bounds: nearest blocking point to a joint axis (default 0.05 m)')
    p.add_argument('--contacts',choices=['tip','all'],default='tip',help='Bounds: block only the tool tip (default) or also every moving body origin')
    p.add_argument('--max-contact-force-n',type=float,help='Bounds: design limit on the blocking force (default: motor-limited only)')
    p.add_argument('--max-contact-moment-nm',type=float,help='Bounds: adds the clamped tip case (any force and moment) with this moment cap; needs --max-contact-force-n')
    p.add_argument('--live',action='store_true',help='Campaign: open a live plot of peak loads and their Fx..Mz record states every 10 counted sweeps (same as arm-sim watch)')
    p.add_argument('--results',type=Path,help='fea-cases: a campaign fea-* folder or a bounds-* folder. watch: a progress/convergence CSV or a folder to search (default --out)')
    p.add_argument('--part',help='fea-cases: the structural part to set up for FEA, e.g. part_3 (a merged part goes by its root body)')
    p.add_argument('--fixed',help='fea-cases: connection to constrain (default: the proximal one, toward the base)')
    p.add_argument('--all-extremes',action='store_true',help='fea-cases: include every Fx..Mz extreme, not just peak force/moment magnitudes')
    p.add_argument('--full-history', action='store_true', help='Retain full traces and all trial extrema (large files); default keeps 120-frame accepted previews and global extrema')
    p.add_argument('--poses', type=int, default=3, help='Restrained: independent random clamp poses')
    p.add_argument('--poses-file', type=Path, help='Restrained: JSON list of joint-angle lists in degrees')
    p.add_argument('--trial-duration', type=float, default=.5, help='Restrained: seconds per torque combination')
    p.add_argument('--torque-ramp', type=float, default=.05, help='Restrained: smooth ramp from zero torque, seconds')
    p.add_argument('--clamp-time-constant', type=float, default=.005, help='Restrained: weld solver time constant, seconds')
    p.add_argument('--clamp-position-tolerance', type=float, default=.005, help='Restrained: maximum weld drift, metres')
    p.add_argument('--clamp-angle-tolerance', type=float, default=.05, help='Restrained: maximum weld drift, radians')
    p.add_argument('--max-trials', type=int, help='Restrained: stop after this many combinations; result is explicitly partial')
    p.add_argument('--port', type=int, default=8050, help='Dashboard: localhost port')
    p.add_argument('--trial', type=int, default=0, help='Replay: campaign trial index')
    args=p.parse_args()
    if args.command=='dashboard':
        from .dashboard import serve
        serve(args.results or args.out, args.port)
        return
    if args.command=='replay':
        if args.results is None:
            p.error('replay needs --results pointing to a campaign export.')
        from .viewer import Window
        from .campaign import load_results
        import yaml, tempfile
        result=load_results(args.results)
        if not 0 <= args.trial < len(result['sweeps']):
            p.error('Trial index outside campaign.')
        # The original config snapshot supplies exact gains and mass settings.
        with tempfile.TemporaryDirectory() as temp:
            config=Path(temp)/'replay.yaml'
            config.write_text(yaml.safe_dump(result['cfg']))
            window=Window(config,args.out)
            window.experiment.load_campaign(args.results)
            window.experiment.replay_sweep(args.trial)
            window.run()
        return
    if args.command=='fea-cases':
        if args.results is None or args.part is None:
            p.error('fea-cases needs --results and --part.')
        from .fea_cases import exports, part_cases, report, write_csv
        try:
            for folder in exports(args.results):
                cases=part_cases(folder,args.part,args.fixed,args.all_extremes)
                print(report(folder,args.part,cases)+'\n')
                write_csv(folder/f'fea_setup_{args.part}.csv',cases)
                print(f'Saved {folder/f"fea_setup_{args.part}.csv"}\n')
        except (ValueError,OSError,KeyError) as exc:
            p.error(str(exc))
        return
    if args.command=='watch':
        from .watch import find, watch
        try:
            path=find(args.results or args.out)
        except ValueError as exc:
            p.error(str(exc))
        print(f'Watching {path}',flush=True)
        watch(path)
        return
    if args.package_root and args.urdf is None:
        p.error('--package-root requires --urdf; for saved configs edit urdf.package_roots in YAML.')
    if args.urdf is not None:
        from .urdf import prepare_urdf_config
        try:
            args.config=prepare_urdf_config(args.urdf,args.package_root)
        except (ValueError,KeyError,TypeError,OSError) as exc:
            p.error(str(exc))
        print(f'Robot controls: {args.config}',flush=True)
    if args.command=='view':
        if args.duration is not None:
            p.error('For the viewer, edit sweep.duration_s or use [ / ].')
        from .viewer import launch
        launch(args.config,args.out)
        return
    c=load_config(args.config)
    if args.duration is not None:
        c['sweep']['duration_s']=args.duration
        validate(c)
    if args.command=='restrained':
        from .restrained import restrained
        import json
        try:
            folder=restrained(c, args.out, poses=args.poses, seed=args.seed, workers=args.workers or 1,
                duration=args.trial_duration, ramp=args.torque_ramp, time_constant=args.clamp_time_constant,
                position_tolerance=args.clamp_position_tolerance, angle_tolerance=args.clamp_angle_tolerance,
                poses_deg=json.loads(args.poses_file.read_text()) if args.poses_file else None,
                full_history=args.full_history, max_trials=args.max_trials)
        except (ValueError, RuntimeError, OSError) as exc:
            p.error(str(exc))
        print(folder)
        return
    if args.command=='campaign':
        if args.duration is not None:
            p.error('For campaigns, use --min-duration and --max-duration.')
        from .campaign import campaign, PROGRESS
        from .fea import METRICS
        if args.live:
            import subprocess, sys
            (args.out/PROGRESS).unlink(missing_ok=True)
            subprocess.Popen([sys.executable,'-m','arm_design.cli','watch','--results',str(args.out/PROGRESS)])
        try:
            folder,rec=campaign(c,args.out,(20000 if args.aggressive else 200) if args.samples is None else args.samples,(0 if args.aggressive else 20) if args.iterations is None else args.iterations,(args.min_duration,args.max_duration),args.workers,args.seed,args.include_failed,
                                None if args.allow_self_collision else args.max_overlap_mm, full_history=args.full_history, aggressive=args.aggressive, search_steps=args.search_steps,
                                timestep_validation=not args.skip_timestep_validation)
        except (ValueError,RuntimeError) as exc:
            p.error(str(exc))
        if rec is None:
            print(f'Cancelled: {folder}')
            return
        st=rec.settings
        policy = 'motor-load samples, including valid partial trials; tracking is diagnostic' if st.get('acceptance_policy') == 'motor_loads_v1' else f'tracking within {st["tracking_tolerance_deg"]:g} deg'
        if st.get('acceptance_policy') == 'motor_loads_v2':
            policy += '; final speed ≤20 deg/s; complete trials only'
        print(f'{st["counted"]} of {st["sweeps"]} sweeps counted ({policy})')
        if st['max_overlap_mm'] is not None:
            print(f'Self-collision: {st["colliding_paths_skipped"]} colliding random paths skipped, {st["collided"]} sweeps stopped mid-motion')
        print(f'{"interface":32} {"max force N":>12} {"max moment Nm":>14}  sweeps')
        f,mo=METRICS.index('force_N'),METRICS.index('moment_Nm')
        for i,interface in enumerate(rec.interfaces):
            sweeps=[rec.rows[rec.max_index[i,k]]['sweep_id'] for k in (f,mo)]
            print(f'{interface["label"]:32} {rec.maximum[i,f]:12.2f} {rec.maximum[i,mo]:14.2f}  {sweeps[0]}, {sweeps[1]}')
        print(folder)
        return
    if args.command=='bounds':
        if args.duration is not None:
            p.error('Bounds are static; --duration does not apply.')
        from .bounds import bounds, OBJECTIVES
        try:
            folder,recs=bounds(c,args.out,2000 if args.samples is None else args.samples,300 if args.iterations is None else args.iterations,
                               args.workers,args.seed,c['drive']['torque_derating'] if args.derated else 1.,
                               None if args.allow_self_collision else args.max_overlap_mm,args.min_lever_m,args.contacts,args.max_contact_force_n,
                               args.max_contact_moment_nm)
        except (ValueError,RuntimeError) as exc:
            p.error(str(exc))
        if args.max_contact_force_n is None:
            print('WARNING: blocked loads are motor-limited only. Aligned poses let small torques resist huge forces, so they grow with sampling; set --max-contact-force-n to a design contact force.')
        f,mo=OBJECTIVES
        print(f'{"interface":28}'+''.join(f' {k+" N":>10} {k+" Nm":>11}' for k in recs))
        for i,interface in enumerate(recs['gravity'].interfaces):
            print(f'{interface["label"]:28}'+''.join(f' {r.maximum[i,f]:10.1f} {r.maximum[i,mo]:11.2f}' for r in recs.values()))
        print(folder)
        return
    if args.command=='export':
        from .model import build_arm
        args.out.mkdir(parents=True,exist_ok=True)
        path=args.out/'arm.xml'
        path.write_text(build_arm(c).xml)
        print(path.resolve())
    else:
        from .simulation import run_headless
        from .reporting import save_result, summary
        arm,run=run_headless(c, 'B -> A' if args.direction == 'b-to-a' else 'A -> B')
        folder=save_result(arm,run,args.out)
        s=summary(run)
        print('Tracking within tolerance' if s['tracking_ok'] else s['reason'] or 'Tracking exceeded tolerance')
        print(folder/'report.html')
        if not s['tracking_ok']:
            raise SystemExit(2)


if __name__=='__main__':
    main()
