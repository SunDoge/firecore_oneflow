import typed_args as ta
import firecore
from firecore.config.lazy import LazyConfig
from icecream import ic


@ta.argument_parser()
class Args(ta.TypedArgs):
    config_file: str = ta.add_argument(
        '-c', '--config-file',
    )


def main():
    args = Args.parse_args()
    ic(args)
    cfg = LazyConfig.load(args.config_file)
    ic(cfg)
    

if __name__ == '__main__':
    main()
