import typed_args as ta


@ta.argument_parser()
class Args(ta.TypedArgs):
    config_file: str = ta.add_argument(
        '-c', '--config-file',
    )
