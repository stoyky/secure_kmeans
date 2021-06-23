# -*- coding: utf-8 -*-
import json
import statistics
import time
from collections import OrderedDict, namedtuple
from functools import wraps
from itertools import groupby
from operator import itemgetter, attrgetter

import pytest
from _pytest.mark import MarkerError, MarkInfo, MarkDecorator
from _pytest.python import Function


DEFAULT_MODE = 'safe'
DEFAULT_REPS = 3

try:
    # `perf_counter` has a better resolution but only exists in Python >= 3.3
    TIME_FUNC = time.perf_counter
except AttributeError:
    TIME_FUNC = time.clock


# def pytest_addoption(parser):
#     group = parser.getgroup('timeit')
#     group.addoption(
#         '--foo',
#         action='store',
#         dest='dest_foo',
#         default='2016',
#         help='Set the value for the fixture "bar".'
#     )
#
#     parser.addini('HELLO', 'Dummy pytest.ini setting')

TimeItResult = namedtuple('TimeItResult', ('id', 'min', 'max', 'mean', 'stdev'))


def _key_func(item):
    return (
        item[0].keywords['_timeit']['source'].nodeid,
        item[0].keywords['_timeit']['rep']
    )


class TimeIt(object):
    def __init__(self, config):
        self.config = config
        self._durations = OrderedDict()
        self._finalized = False
        self.report_items = []

    @property
    def durations(self):
        return self._durations

    def add_duration(self, item, duration):
        if self._finalized:
            raise RuntimeError("Can't add duration after call to `.finalize()`")
        self._durations[item] = duration

    def finalize(self):
        self._finalized = True
        samples = [
            (item, duration)
            for item, duration
            in self._durations.items()
            if '_timeit' in item.keywords
        ]
        if not samples:
            return
        # group by test
        grouped_items = groupby(
            sorted(
                samples,
                key=_key_func
            ),
            key=_key_func
        )
        # group by repetition
        grouped_items = groupby(
            (
                (nodeid.replace("::()::", "::"), sum(duration for _, duration in items))
                for (nodeid, rep), items in grouped_items
            ),
            key=itemgetter(0)
        )

        results = []
        for nodeid, items in grouped_items:
            durations = [i[1] for i in items]
            multiple = len(durations) > 1
            results.append(
                TimeItResult(
                    nodeid,
                    min(durations),
                    max(durations),
                    statistics.mean(durations) if multiple else durations[0],
                    statistics.stdev(durations) if multiple else 0
                )
            )

        self.report_items = sorted(results, key=attrgetter('mean'))


@pytest.hookimpl
def pytest_configure(config):
    config.timeit = TimeIt(config)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    timeit_ = pyfuncitem.keywords.get('_timeit')
    if not timeit_:
        yield
        return
    start = TIME_FUNC()
    yield
    stop = TIME_FUNC()
    duration = (stop - start) * 1000 * 1000
    if timeit_['mode'] == 'fast':
        duration = duration / timeit_['number']
    pyfuncitem.config.timeit.durations[pyfuncitem] = duration


@pytest.hookimpl()
def pytest_collection_modifyitems(session, config, items):
    final_items = []
    for item in items:
        assert isinstance(item, Function)
        marker = item.get_marker('timeit')
        if not marker:
            final_items.append(item)
            continue
        if marker.args:
            raise MarkerError(
                "'timeit' mark doesn't accept positional arguments (on '{}')".format(
                    item.nodeid))
        number = marker.kwargs.get('number', marker.kwargs.get('n'))
        if number is None:
            raise MarkerError(
                "'timeit' mark on '{}' needs 'n'/'number' kwarg".format(
                    item.nodeid))
        reps = marker.kwargs.get('repetitions', marker.kwargs.get('r', DEFAULT_REPS))
        mode = marker.kwargs.get('mode', DEFAULT_MODE)

        original_name = item.name
        if reps > 1:
            item.name = "{}[n={}, r={}]".format(item.name, number, reps)
        else:
            item.name = "{}[n={}]".format(item.name, number)

        if mode == 'safe':
            if reps > 1:
                template = "{name}[n={n}/{number}, r={r}/{reps}]"
            else:
                template = "{name}[n={n}/{number}]"
        elif mode == 'fast':
            if reps > 1:
                template = "{name}[n={number}, r={r}/{reps}]"
            else:
                template = "{name}[n={number}]"
        else:
            raise MarkerError(
                "'timeit' mark 'mode' may only be one of 'safe' or 'fast' (on '{}')".format(
                    item.nodeid
                )
            )

        for r in range(reps):
            if mode == 'safe':
                for n in range(number):
                    final_items.append(
                        _add_function(item, original_name, mode, n, number, r, reps, template)
                    )
            elif mode == 'fast':
                final_items.append(
                    _add_function(item, original_name, mode, 0, number, r, reps, template)
                )

    items[:] = final_items


def _runtest(number, obj):
    @wraps(obj)
    def run(*args, **kwargs):
        for _ in range(number):
            obj(*args, **kwargs)
    return run


def _add_function(item, original_name, mode, n, number, r, reps, template):
    return Function(
        template.format(
            name=original_name,
            n=n + 1,
            number=number,
            r=r + 1,
            reps=reps,
        ),
        item.parent,
        callobj=item.obj if mode == 'safe' else _runtest(number, item.obj),
        originalname=item.name,
        keywords={
            '_timeit': {
                'source': item, 'rep': r, 'mode': mode, 'number': number
            }
        }
    )


@pytest.hookimpl
def pytest_terminal_summary(terminalreporter, exitstatus):
    tr = terminalreporter
    tr.ensure_newline()
    tr.rewrite("Computing stats ...", black=True, bold=True)
    tr.config.timeit.finalize()
    tr.rewrite("")
    if not tr.config.timeit.report_items:
        return
    tr.write_sep('+', title="TimeIt results")
    tr.write_line(u"{:40s} {:>10} {:>10} {:>10} {:>6}".format(
        u"Test (times in µs)", "mean", "min", "max", "StDev"))
    for item in tr.config.timeit.report_items:
        if len(item.id) <= 40:
            id_ = "{:40}".format(item.id)
        else:
            id_ = "...{}".format(item.id[-37:])
        tr.write_line(
            "{id} {i.mean:10.3f} {i.min:10.3f} {i.max:10.3f} {i.stdev:6.2f}".format(
                id=id_,
                i=item
            )
        )
    with tr.config.rootdir.join(".timeit.json").open("w") as f:
        json.dump(
            OrderedDict(
                (item.id, dict(list(item._asdict().items()) + [('unit', 'us')]))
                for item in tr.config.timeit.report_items
            ),
            f
        )

