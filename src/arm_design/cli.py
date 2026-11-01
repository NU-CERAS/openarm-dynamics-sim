import argparse
from pathlib import Path
from .config import DEFAULT_CONFIG, ROOT, load_config, validate


def main():
    p=argparse.ArgumentParser(description='MuJoCo joint jogging and timed A/B sweeps')
    p.add_argument('command',nargs='?',default='view',choices=['view','sweep','export','campaign'])
    source=p.add_mutually_exclusive_group()
    source.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    source.add_argument('--urdf',type=Path,help='Import a fixed-base rotary robot; creates <robot>.sim.yaml beside the URDF')
    p.add_argument('--package-root',type=Path,action='append',default=[],help='Folder containing ROS packages, or a package itself (repeatable; used with --urdf)')
    p.add_argument('--out',type=Path,default=ROOT/'results')
    p.add_argument('--duration',type=float,help='A→B seconds (headless sweep only; viewer uses [/] or YAML)')
    p.add_argument('--direction',choices=['a-to-b','b-to-a'],default='a-to-b',
                   help='Direction for headless sweeps (default: a-to-b)')
    p.add_argument('--samples',type=int,default=200,help='Campaign: random A/B sweeps before refinement')
    p.add_argument('--iterations',type=int,default=20,help='Campaign: refinement sweeps per link force/moment')
    p.add_argument('--min-duration',type=float,default=.1,help='Campaign: shortest A→B seconds (default 0.1)')
    p.add_argument('--max-duration',type=float,default=5.,help='Campaign: longest A→B seconds (default 5)')
    p.add_argument('--workers',type=int,help='Campaign: parallel processes (default: CPU count)')
    p.add_argument('--seed',type=int,default=0,help='Campaign: random seed')
    p.add_argument('--include-failed',action='store_true',help='Campaign: also count sweeps that exceed tracking tolerance (joint-stop impacts)')
    args=p.parse_args()
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
            folder,rec=campaign(c,args.out,args.samples,args.iterations,(args.min_duration,args.max_duration),args.workers,args.seed,args.include_failed)
        except (ValueError,RuntimeError) as exc:
            p.error(str(exc))
        print(f'{rec.settings["counted"]} of {rec.settings["sweeps"]} sweeps counted (tracking within {rec.settings["tracking_tolerance_deg"]:g} deg)')
        print(f'{"interface":32} {"max force N":>12} {"max moment Nm":>14}  sweeps')
        f,mo=METRICS.index('force_N'),METRICS.index('moment_Nm')
        for i,interface in enumerate(rec.interfaces):
            if interface['sign']==1:
                sweeps=[rec.rows[rec.max_index[i,k]]['sweep_id'] for k in (f,mo)]
                print(f'{interface["label"]:32} {rec.maximum[i,f]:12.2f} {rec.maximum[i,mo]:14.2f}  {sweeps[0]}, {sweeps[1]}')
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
