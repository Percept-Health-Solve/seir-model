from dataclasses import dataclass, field, is_dataclass
from typing import Optional, List
import argparse

from seir.argparser import DataClassArgumentParser
from seir.cli import OdeCLI


@dataclass
class BasicArgs:
    test_int: int
    test_float: float
    test_str: str
    test_bool: bool

@dataclass
class BasicArgs2:
    test_args: int


@dataclass
class DefaultArgs:
    test_int: int = 42
    test_float: float = 1.1
    test_str: str = field(default='test', metadata={"help": "test str"})
    test_false: bool = False
    test_true: bool = True


@dataclass
class OptionalArgs:
    test_int: Optional[int] = None
    test_float: Optional[float] = None
    test_str: Optional[str] = None
    test_list: Optional[List[int]] = field(default_factory=lambda: [])


@dataclass
class ListArgs:
    test_int: List[int] = field(default_factory=lambda: [])
    test_float: List[float] = field(default_factory=lambda: [1.1, 1.2, 1.3])
    test_str: List[str] = field(default_factory=lambda: [])


def argparser_equal(parser: argparse.ArgumentParser, expected: argparse.ArgumentParser):
    """Pseudo checks if two argparsers are equal"""
    assert len(parser._actions) == len(expected._actions)
    for x, y in zip(parser._actions, expected._actions):
        xx = {k: v for k, v in vars(x).items() if k != "container"}
        yy = {k: v for k, v in vars(y).items() if k != "container"}
    for k, v in xx.items():
        assert k in yy
        assert v == yy[k]


def test_base_args():
    parser = DataClassArgumentParser(BasicArgs)

    expected = argparse.ArgumentParser()
    expected.add_argument('--test_int', type=int, required=True)
    expected.add_argument('--test_float', type=float, required=True)
    expected.add_argument('--test_str', type=str, required=True)
    expected.add_argument('--test_bool', action="store_true")

    argparser_equal(parser, expected)


def test_default_arg():
    parser = DataClassArgumentParser(DefaultArgs)

    expected = argparse.ArgumentParser()
    expected.add_argument('--test_int', type=int, default=42)
    expected.add_argument('--test_float', type=float, default=1.1)
    expected.add_argument('--test_str', type=str, default='test', help='test str')
    expected.add_argument('--test_false', action='store_true')
    expected.add_argument('--no-test_true', action='store_false', dest='test_true')

    argparser_equal(parser, expected)


def test_optional_args():
    parser = DataClassArgumentParser(OptionalArgs)

    expected = argparse.ArgumentParser()
    expected.add_argument('--test_int', default=None, type=int)
    expected.add_argument('--test_float', default=None, type=float)
    expected.add_argument('--test_str', default=None, type=str)
    expected.add_argument('--test_bool', action='store_true', help='test bool')
    expected.add_argument('--test_list', default=[], nargs='+', type=int)

    args = parser.parse_args([])
    assert args == argparse.Namespace(test_int=None, test_float=None, test_str=None, test_list=[])

    args = parser.parse_args("--test_int 1 --test_float 1.1 --test_str xyz --test_list 1 2 3".split())
    assert args == argparse.Namespace(test_int=1, test_float=1.1, test_str='xyz', test_list=[1, 2, 3])


def test_list_args():
    parser = DataClassArgumentParser(ListArgs)

    expected = argparse.ArgumentParser()
    expected.add_argument('--test_int', default=[], nargs='+', type=int)
    expected.add_argument('--test_float', default=[1.1, 1.2, 1.3], nargs='+', type=float)
    expected.add_argument('--test_str', default=[], nargs='+', type=str)

    argparser_equal(parser, expected)


def test_return_args_type():
    parser = DataClassArgumentParser(BasicArgs)
    args, = parser.parse_args_into_dataclasses("--test_int 1 --test_float 1.1 --test_str xyz".split())
    assert is_dataclass(args)


def test_accept_list():
    parser = DataClassArgumentParser([BasicArgs, BasicArgs2])
    args1, args2 = parser.parse_args_into_dataclasses("--test_int 1 --test_float 1.1 --test_str xyz --test_args 1".split())
    assert is_dataclass(args1)
    assert is_dataclass(args2)
    assert args1 == BasicArgs(test_int=1, test_float=1.1, test_str='xyz', test_bool=False)
    assert args2 == BasicArgs2(test_args=1)


def test_covid_model_cli_args():
    parser = DataClassArgumentParser(OdeCLI)
    assert parser is not None
