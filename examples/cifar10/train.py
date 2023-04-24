import typed_args as ta
import firecore
from firecore.config.lazy import LazyConfig
from firecore.config.instantiate import instantiate
from icecream import ic
from firecore_oneflow.model.builder import GraphBuilder
from firecore_oneflow.runners.base import BaseRunner
import logging

logger = logging.getLogger(__name__)


@ta.argument_parser()
class Args(ta.TypedArgs):
    config_file: str = ta.add_argument(
        "-c",
        "--config-file",
    )
    fast_dev: bool = ta.add_argument("--fast-dev", action="store_true")
    eval_only: bool = ta.add_argument("--eval-only", action="store_true")


def main():
    firecore.logging.init()
    args = Args.parse_args()
    ic(args)
    cfg = LazyConfig.load(args.config_file)
    # ic(cfg)
    logger.info(f"{cfg}")
    if args.fast_dev:
        pass

    graph_builder = GraphBuilder(
        model_cfg=cfg.model,
        criterion_cfg=cfg.criterion,
        optimizer_cfg=cfg.optimizer,
        lr_scheduler_cfg=cfg.lr_scheduler,
    )

    test_runner: BaseRunner = instantiate(
        cfg.test_runner,
        forward_fn=graph_builder.forward,
        model=graph_builder.model,
    )

    if args.eval_only:
        test_runner.step(0)
        pass


if __name__ == "__main__":
    main()
