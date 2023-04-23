import typed_args as ta
import firecore
from firecore.config.lazy import LazyConfig
from firecore.config.instantiate import instantiate
from icecream import ic
from firecore_oneflow.module.builder import GraphBuilder


@ta.argument_parser()
class Args(ta.TypedArgs):
    config_file: str = ta.add_argument(
        "-c",
        "--config-file",
    )
    fast_dev: bool = ta.add_argument("--fast-dev", action="store_true")
    eval_only: bool = ta.add_argument("--eval-only", action="store_true")


def main():
    args = Args.parse_args()
    ic(args)
    cfg = LazyConfig.load(args.config_file)
    # ic(cfg)

    if args.fast_dev:
        pass

    graph_builder = GraphBuilder(
        model_cfg=cfg.model,
        criterion_cfg=cfg.criterion,
        optimizer_cfg=cfg.optimizer,
        lr_scheduler_cfg=cfg.lr_scheduler,
    )

    test_runner = instantiate(
        cfg.test_runner,
        model=graph_builder.model,
    )

    if args.eval_only:
        pass


if __name__ == "__main__":
    main()
