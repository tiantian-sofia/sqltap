# -*- encoding: utf8 -*-
"""Concurrency and memory-boundedness tests for SQLTapMiddleware.

These tests exercise the middleware the way gunicorn's gthread worker
does: one middleware instance shared by many threads. They run under
both nose and pytest.
"""
from __future__ import print_function

import threading

import sqltap
import sqltap.wsgi
from sqlalchemy import create_engine, text
from werkzeug.test import EnvironBuilder


class MockResults(object):
    def __init__(self, rowcount=1):
        self.rowcount = rowcount


def make_qstats(text_):
    # a minimal but renderable stack frame: (filename, lineno, func, line)
    stack = [('app.py', 1, 'view', 'run_query()')]
    return sqltap.QueryStats(text_, stack, 1.0, 2.0, None, {},
                             MockResults(1))


def dummy_app(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/plain')])
    return [b'ok']


def call_wsgi(app, method='GET', body=None):
    builder = EnvironBuilder(path=app.path, method=method, data=body)
    captured = {}

    def start_response(status, headers, exc_info=None):
        captured['status'] = status

    result = app(builder.get_environ(), start_response)
    captured['body'] = b''.join(result).decode('utf-8')
    return captured


class CheckedFakeProfiler(object):
    """Stand-in for sqltap.ProfilingSession with the same transition
    contract: start()/stop() raise AssertionError on invalid
    transitions. Used instead of the real profiler so the tests don't
    touch SQLAlchemy's global event registry.
    """

    def __init__(self, started=False):
        self.started = started

    def start(self):
        if self.started:
            raise AssertionError("Profiling session is already started!")
        self.started = True

    def stop(self):
        if not self.started:
            raise AssertionError("Profiling session is already stopped")
        self.started = False


class OnRaceGate(object):
    """Data descriptor temporarily installed as SQLTapMiddleware.on
    to deterministically reproduce the check-then-act race: the first
    thread to *read* the flag captures the current value and parks
    until a second thread also reads it; both then observe the same
    captured value, so on the unfixed middleware both proceed to call
    profiler.start()/stop() and the second one raises AssertionError
    (a 500 under gunicorn).

    With the fix, the whole check-and-act is serialized by self._lock,
    so the second reader can only arrive after the first thread
    finished its transition; the gate notices the race window has
    closed (first_done is set) and returns the real, updated value,
    making the second toggle a no-op.
    """

    def __init__(self):
        self.first_read = threading.Event()
        self.second_read = threading.Event()
        self.first_done = threading.Event()

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if not self.first_read.is_set():
            # first reader: capture the value and wait for a concurrent
            # reader to arrive inside the race window
            self._value = obj.__dict__['on']
            self.first_read.set()
            self.second_read.wait(1)
            self.first_done.set()
            return self._value
        if not self.first_done.is_set():
            # concurrent second reader inside the race window: observe
            # exactly what the first reader observed, then release it
            self.second_read.set()
            return self._value
        return obj.__dict__['on']

    def __set__(self, obj, value):
        obj.__dict__['on'] = value


def run_concurrently(fn, threads=2):
    barrier = threading.Barrier(threads)
    outcomes = []

    def worker():
        barrier.wait()
        try:
            outcomes.append(('ok', fn()))
        except Exception as exc:
            outcomes.append(('error', exc))

    workers = [threading.Thread(target=worker) for _ in range(threads)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(30)
    return outcomes


def test_concurrent_turn_on_does_not_500():
    """Two simultaneous POST turn=on requests must both succeed.

    With the old check-then-act toggle, both threads passed the
    if not self.on check and both called ProfilingSession.start(),
    the second of which raised AssertionError -> 500 under gunicorn.
    """
    app = sqltap.wsgi.SQLTapMiddleware(app=dummy_app)
    app.profiler = CheckedFakeProfiler()
    sqltap.wsgi.SQLTapMiddleware.on = OnRaceGate()
    try:
        outcomes = run_concurrently(
            lambda: call_wsgi(app, 'POST', 'turn=on')['status'])
    finally:
        del sqltap.wsgi.SQLTapMiddleware.on

    errors = [exc for kind, exc in outcomes if kind == 'error']
    assert not errors, "concurrent turn=on raised: %r" % (errors,)
    statuses = [status for kind, status in outcomes]
    assert statuses == ['200 OK', '200 OK'], statuses
    assert app.on is True
    assert app.profiler.started is True


def test_concurrent_turn_off_does_not_500():
    """Two simultaneous POST turn=off requests must both succeed."""
    app = sqltap.wsgi.SQLTapMiddleware(app=dummy_app)
    app.profiler = CheckedFakeProfiler(started=True)
    app.on = True
    sqltap.wsgi.SQLTapMiddleware.on = OnRaceGate()
    try:
        outcomes = run_concurrently(
            lambda: call_wsgi(app, 'POST', 'turn=off')['status'])
    finally:
        del sqltap.wsgi.SQLTapMiddleware.on

    errors = [exc for kind, exc in outcomes if kind == 'error']
    assert not errors, "concurrent turn=off raised: %r" % (errors,)
    statuses = [status for kind, status in outcomes]
    assert statuses == ['200 OK', '200 OK'], statuses
    assert app.on is False
    assert app.profiler.started is False


def test_sequential_toggles_are_idempotent():
    """Repeated turn=on / turn=off POSTs are harmless no-ops."""
    app = sqltap.wsgi.SQLTapMiddleware(app=dummy_app)
    try:
        for _ in range(3):
            assert call_wsgi(app, 'POST', 'turn=on')['status'] == '200 OK'
        assert app.on is True
        for _ in range(3):
            assert call_wsgi(app, 'POST', 'turn=off')['status'] == '200 OK'
        assert app.on is False
    finally:
        if app.on:
            app.stop()


def test_clear_also_drains_pending_queue():
    """After POST clear=1, a refresh must not resurrect cleared queries.

    Regression test: the old code only emptied self.stats, leaving
    already-collected queries in the collector queue; the next GET
    drained them right back onto the page.
    """
    engine = create_engine('sqlite:///:memory:')
    app = sqltap.wsgi.SQLTapMiddleware(app=dummy_app)
    probe = 'sqltap_clear_probe_7f3a'
    try:
        assert call_wsgi(app, 'POST', 'turn=on')['status'] == '200 OK'
        with engine.connect() as conn:
            for _ in range(5):
                conn.execute(text("SELECT 1 AS %s" % probe))
        # queries are buffered in the queue, not yet rendered
        assert app.collector.qsize() == 5

        assert call_wsgi(app, 'POST', 'clear=1')['status'] == '200 OK'
        assert app.collector.qsize() == 0
        assert len(app.stats) == 0

        refreshed = call_wsgi(app, 'GET')
        assert refreshed['status'] == '200 OK'
        assert probe not in refreshed['body']
        assert len(app.stats) == 0
    finally:
        if app.on:
            app.stop()
        engine.dispose()


def test_stats_are_bounded_keep_newest():
    """self.stats never exceeds max_stats; oldest entries are dropped."""
    app = sqltap.wsgi.SQLTapMiddleware(app=dummy_app, max_stats=10)
    for i in range(25):
        app.collector.put_nowait(make_qstats('SELECT %d' % i))

    response = call_wsgi(app, 'GET')
    assert response['status'] == '200 OK'
    assert len(app.stats) == 10
    kept = [str(q.text) for q in app.stats]
    assert kept == ['SELECT %d' % i for i in range(15, 25)], kept


def test_queue_is_bounded_and_never_blocks_producer():
    """A full collector queue drops the incoming item instead of
    blocking the application thread that produced it."""
    app = sqltap.wsgi.SQLTapMiddleware(app=dummy_app, max_queue=5)
    for i in range(10):
        app._collect(make_qstats('SELECT %d' % i))
    assert app.collector.qsize() == 5
    assert app.dropped == 5


def test_render_uses_a_snapshot():
    """The report must be rendered from a copy of self.stats, not from
    the live list that other threads may be mutating."""
    app = sqltap.wsgi.SQLTapMiddleware(app=dummy_app)
    app.stats.append(make_qstats('SELECT 1'))

    seen = []
    # sqltap.wsgi calls sqltap.sqltap.report (the submodule, not the
    # re-export on the package), so patch it there.
    real_report = sqltap.sqltap.report

    def spy(stats, **kwargs):
        seen.append(stats)
        return real_report(stats, **kwargs)

    sqltap.sqltap.report = spy
    try:
        response = call_wsgi(app, 'GET')
    finally:
        sqltap.sqltap.report = real_report

    assert response['status'] == '200 OK'
    assert len(seen) == 1
    assert seen[0] is not app.stats
    assert list(seen[0]) == list(app.stats)
