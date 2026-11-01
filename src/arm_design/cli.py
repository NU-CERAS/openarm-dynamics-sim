import argparse
from pathlib import Path
from .config import DEFAULT_CONFIG, ROOT, load_config, validate


def main():
    p=argparse.ArgumentParser(description='MuJoCo joint jogging and timed A/B sweeps')
    p.add_argument('command',nargs='?',default='view',choices=['view','sweep','export','campaign','bounds','fea-cases'])
    source=p.add_mutually_exclusive_group()
    source.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    source.add_argument('--urdf',type=Path,help='Import a fixed-base rotary robot; creates <robot>.sim.yaml beside the URDF')
    p.add_argument('--package-root',type=Path,action='append',default=[],help='Folder containing ROS packages, or a package itself (repeatable; used with --urdf)')
    p.add_argument('--out',type=Path,default=ROOT/'results')
    p.add_argument('--duration',type=float,help='A→B seconds (headless sweep only; viewer uses [/] or YAML)')
    p.add_argument('--direction',choices=['a-to-b','b-to-a'],default='a-to-b',
                   help='Direction for headless sweeps (default: a-to-b)')
    p.add_argument('--samples',type=int,help='Campaign: random A/B sweeps (default 200). Bounds: random poses (default 2000)')
    p.add_argument('--iterations',type=int,help='Refinement steps per link force/moment (campaign default 20 sweeps, bounds 300 poses)')
    p.add_argument('--min-duration',type=float,default=.1,help='Campaign: shortest A→B seconds (default 0.1)')
    p.add_argument('--max-duration',type=float,default=5.,help='Campaign: longest A→B seconds (default 5)')
    p.add_argument('--workers',type=int,help='Campaign: parallel processes (default: CPU count)')
    p.add_argument('--seed',type=int,default=0,help='Campaign: random seed')
    p.add_argument('--include-failed',action='store_true',help='Campaign: also count sweeps that exceed tracking tolerance (joint-stop impacts)')
    p.add_argument('--max-overlap-mm',type=float,default=0.,help='Campaign: largest convex-hull overlap allowed before a pose counts as self-colliding (default 0)')
    p.add_argument('--allow-self-collision',action='store_true',help='Campaign/bounds: skip the self-collision check')
    p.add_argument('--derated',action='store_true',help='Bounds: cap motors at the YAML torque derating instead of full peak torque')
    p.add_argument('--min-lever-m',type=float,default=.05,help='Bounds: nearest blocking point to a joint axis (default 0.05 m)')
    p.add_argument('--contacts',choices=['tip','all'],default='tip',help='Bounds: block only the tool tip (default) or also every moving body origin')
    p.add_argument('--max-contact-force-n',type=float,help='Bounds: design limit on the blocking force (default: motor-limited only)')
    p.add_argument('--results',type=Path,help='fea-cases: a campaign fea-* folder or a bounds-* folder')
    p.add_argument('--part',help='fea-cases: the body to set up for FEA, e.g. part_3')
    p.add_argument('--fixed',help='fea-cases: connection to constrain (default: the proximal one, toward the base)')
    p.add_argument('--all-extremes',action='store_true',help='fea-cases: include every Fx..Mz extreme, not just peak force/moment magnitudes')
    args=p.parse_args()
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
    if args.command=='campaign':
        if args.duration is not None:
            p.error('For campaigns, use --min-duration and --max-duration.')
        from .campaign import campaign
        from .fea import METRICS
        try:
            folder,rec=campaign(c,args.out,200 if args.samples is None else args.samples,20 if args.iterations is None else args.iterations,(args.min_duration,args.max_duration),args.workers,args.seed,args.include_failed,
                                None if args.allow_self_collision else args.max_overlap_mm)
        except (ValueError,RuntimeError) as exc:
            p.error(str(exc))
        st=rec.settings
        print(f'{st["counted"]} of {st["sweeps"]} sweeps counted (tracking within {st["tracking_tolerance_deg"]:g} deg)')
        if st['max_overlap_mm'] is not None:
            print(f'Self-collision: {st["colliding_paths_skipped"]} colliding random paths skipped, {st["collided"]} sweeps stopped mid-motion')
        print(f'{"interface":32} {"max force N":>12} {"max moment Nm":>14}  sweeps')
        f,mo=METRICS.index('force_N'),METRICS.index('moment_Nm')
        for i,interface in enumerate(rec.interfaces):
            if interface['sign']==1:
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
                               None if args.allow_self_collision else args.max_overlap_mm,args.min_lever_m,args.contacts,args.max_contact_force_n)
        except (ValueError,RuntimeError) as exc:
            p.error(str(exc))
        if args.max_contact_force_n is None:
            print('WARNING: blocked loads are motor-limited only. Aligned poses let small torques resist huge forces, so they grow with sampling; set --max-contact-force-n to a design contact force.')
        f,mo=OBJECTIVES
        print(f'{"interface":28} {"gravity N":>10} {"gravity Nm":>11} {"blocked N":>10} {"blocked Nm":>11}')
        for i,interface in enumerate(recs['gravity'].interfaces):
            if interface['sign']==1:
                print(f'{interface["label"]:28} {recs["gravity"].maximum[i,f]:10.1f} {recs["gravity"].maximum[i,mo]:11.2f} '
                      f'{recs["blocked"].maximum[i,f]:10.1f} {recs["blocked"].maximum[i,mo]:11.2f}')
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
