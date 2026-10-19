import argparse
from pathlib import Path
from .config import DEFAULT_CONFIG, ROOT, load_config, validate


def main():
    p=argparse.ArgumentParser(description='MuJoCo joint jogging and timed A/B sweeps')
    p.add_argument('command',nargs='?',default='view',choices=['view','sweep','export'])
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('--out',type=Path,default=ROOT/'results')
    p.add_argument('--duration',type=float,help='A→B seconds (headless sweep only; viewer uses [/] or YAML)')
    args=p.parse_args()
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
    if args.command=='export':
        from .model import build_arm
        args.out.mkdir(parents=True,exist_ok=True)
        path=args.out/'arm.xml'
        path.write_text(build_arm(c).xml)
        print(path.resolve())
    else:
        from .simulation import run_headless
        from .reporting import save_result, summary
        arm,run=run_headless(c)
        folder=save_result(arm,run,args.out)
        s=summary(run)
        print('Tracking within tolerance' if s['tracking_ok'] else s['reason'] or 'Tracking exceeded tolerance')
        print(folder/'report.html')
        if not s['tracking_ok']:
            raise SystemExit(2)


if __name__=='__main__':
    main()
