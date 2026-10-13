import argparse

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
    "hide-constant": PassInfo(
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
        "top of the window lose their inputs. Use to render large models quickly. "
        "Use --truncate-size = -1 to keep everything from offset to the end",
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
    pipeline = Pipeline([(str(i), transform) for i, transform in enumerate(passes)])

    graph = OpenVINOParser().parse(Path(src))
    graph = pipeline.apply(graph)

    GraphvizRenderer().render(graph, Path(dst))

    return graph, pipeline


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

sub.add_parser("list-passes", help="show available passes")

args = parser.parse_args()

if args.command == "list-passes":
    list_passes()
elif args.command == "visualize":
    visualize(parser, args)
