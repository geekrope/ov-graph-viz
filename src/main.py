import argparse
import time
import pandas as pd

from parser import OpenVINOParser
from transformation import *
from renderer import GraphvizRenderer
from pathlib import Path
from collections.abc import Callable
from dataclasses import dataclass

type PassFactory = Callable[[argparse.Namespace], TransformationPass]


@dataclass
class PassInfo:
    factory: PassFactory
    description: str
    options: str = ""


PASSES: dict[str, PassInfo] = {
    "hide-constants": PassInfo(
        lambda args: ConstantHidingPass(),
        "Removes Constant nodes and shows their shape/dtype inside the consumer "
        "node as [port]: dtype[shape]. Use to declutter weights and scalars."),
    "highlight": PassInfo(
        lambda args: HighlightingPass(set(args.highlight_names.split(","))),
        "Draws a red border around nodes with the given names (exact match, "
        "all duplicates). Use to find specific ops in a large graph.",
        "--highlight-names NAME[,NAME...]"),
    "truncate": PassInfo(
        lambda args: TruncationPass(args.truncate_size, args.truncate_offset),
        "Keeps N nodes starting from position OFFSET in as-late-as-possible "
        "topological order, i.e. a contiguous part of the network. Nodes at the "
        "top of the window lose their inputs. Retains the connection with nodes left out of frame. "
        "Use to render large models quickly. Use --truncate-size = -1 to keep everything from offset to the end",
        "--truncate-size N (default -1), --truncate-offset OFFSET (default 0)"),
    "proxy": PassInfo(
        lambda args: ProxyPass(args.max_consumers),
        "Draws a dashed copy of a node next to each consumer instead of long "
        "edges when it has more than K consumers. Use for shared inputs "
        "(attention mask, positions).",
        "--max-consumers K (default 3)"),
}


def list_passes():
    for name, info in PASSES.items():
        print(name)
        print(f"    {info.description}")
        print(f"    options: {info.options or 'none'}")


def timing_report(parsing_time: float, rendering_time: float,
                  transformation_time: dict[str, float]):
    df = pd.DataFrame([("parsing", parsing_time),
                       *transformation_time.items(),
                       ("rendering_time", rendering_time)],
                      columns=["action", "time, s"])
    return df


def visualize(parser: argparse.ArgumentParser, args: argparse.Namespace):
    if not args.model.exists():
        parser.error(f"the file {args.model} is not found")

    src = str(args.model)
    dst = str(args.output)

    pass_names = str(args.passes).replace(" ", "").split(",")
    for name in pass_names:
        if name and name not in PASSES:
            parser.error(f"unknown pass '{name}', available: {', '.join(PASSES)}")
    passes = [PASSES[name].factory(args) for name in pass_names if name != ""]
    pipeline = Pipeline([(f"{pass_names[i]}_{i}", transform) for i, transform in enumerate(passes)])

    parsing_time = time.time()
    graph = OpenVINOParser().parse(Path(src))
    parsing_time = time.time() - parsing_time

    graph = pipeline.apply(graph)

    rendering_time = time.time()
    GraphvizRenderer().render(graph, Path(dst))
    rendering_time = time.time() - rendering_time

    if args.timing:
        print(timing_report(parsing_time, rendering_time, pipeline.timings))

    return graph, pipeline


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Visualize an OpenVINO model graph")
    sub = parser.add_subparsers(dest="command", required=True)

    vis = sub.add_parser("visualize", help="render model graph to SVG")
    vis.add_argument("--model", type=Path, required=True)
    vis.add_argument("--output", type=Path, default=Path("simple_graph.svg"))
    vis.add_argument("--passes", default="")
    vis.add_argument("--highlight-names", default="")
    vis.add_argument("--truncate-size", type=int, default=-1)
    vis.add_argument("--truncate-offset", type=int, default=0)
    vis.add_argument("--max-consumers", type=int, default=3)
    vis.add_argument("--timing", action="store_true",
                     help="print stage timings")

    sub.add_parser("list-passes", help="show available passes")

    args = parser.parse_args(argv)

    if args.command == "list-passes":
        list_passes()
    elif args.command == "visualize":
        visualize(parser, args)


if __name__ == "__main__":
    main()
