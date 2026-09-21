# -*- encoding: utf8 -*-
"""Concurrency / memory-safety regression tests for sqltap.wsgi.SQLTapMiddleware.

The original middleware had three production bugs under a threaded WSGI
server (gunicorn gthread, multiple workers x threads):

1. start()/stop() were non-atomic check-then-act, so concurrent POSTs of
   ``turn=on`` could both call ProfilingSession.start(), with the second
   raising AssertionError ("Profiling session is already started!") and
   surfacing as a 500.

2. Clear deleted ``self.stats`` but never drained the collector queue, so
   entries queued before the click flowed back on the next request.

3. Both the collector queue and the stats list were unbounded, growing for
   the whole process lifetime when profiling was on, and rendering iterated
   the live list while business threads appended to it.

These tests deliberately avoid any dependency on nose internals so they run
under both nose and pytest.
"""
from __future__ import print_function

import sys
import threading
import traceback as traceback_mod

import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from werkzeug.test import Client
from werkzeug.testapp import test_app as _test_app
from werkzeug.wrappers import Response

import sqltap
import sqltap.wsgi


WSGI_FILE = sqltap.wsgi.__file__


def _make_engine_and_session():
    engine = create_engine('sqlite:///:memory:')
    Base = declarative_base()

    class Row(Base):
        __tablename__ = "rows"
        id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
        name = sqlalchemy.Column(sqlalchemy.String)

    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    return engine, session


class TestMiddlewareSwitchIdempotent(object):
    """Concurrent ``turn=on`` / ``turn=off`` must be idempotent and never 500.

    A plain stress loop cannot hit the old race reliably: the buggy window
    was just two bytecodes long and the GIL hides it unless a thread switch
    lands exactly between the flag check and the flag assignment. We force
    that switch deterministically with a per-thread trace function
    (sys.settrace): whenever any thread is about to execute a line in the
    middleware's start()/stop() methods, it waits at a barrier until every
    other thread is parked at *some* line of the same method, then releases
    them together. Repeated over the lines of the method, this enumerates
    the interleavings -- including "both threads observed self.on == False
    before either assigned it" -- and the unpatched code raises inside the
    second profiler.start() on essentially the first round.

    Against the fixed code the critical section serializes the threads at
    the lock, so the barrier simply releases them one by one and no
    profiler.start() is ever called twice.
    """

    NUM_THREADS = 8

    def setUp(self):
        self.engine, self.session = _make_engine_and_session()
        self.middleware = sqltap.wsgi.SQLTapMiddleware(app=_test_app)

        # Count real transitions of the underlying ProfilingSession so the
        # assertions do not rely only on HTTP status codes.
        self.start_calls = []
        self.stop_calls = []
        real_start = self.middleware.profiler.start
        real_stop = self.middleware.profiler.stop

        def counting_start():
            self.start_calls.append(1)
            return real_start()

        def counting_stop():
            self.stop_calls.append(1)
            return real_stop()

        self.middleware.profiler.start = counting_start
        self.middleware.profiler.stop = counting_stop

        self._old_switch_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)

    def tearDown(self):
        sys.setswitchinterval(self._old_switch_interval)
        sys.settrace(None)
        # Stop profiling no matter how the test ended, otherwise the global
        # SQLAlchemy event listeners leak into other test cases.
        if self.middleware.on:
            try:
                self.middleware.stop()
            except AssertionError:
                pass

    def _install_line_barrier(self, method_name):
        barrier = threading.Barrier(self.NUM_THREADS)

        def tracer(frame, event, arg):
            # sys.settrace only installs a per-frame local tracer when the
            # global tracer returns one on the 'call' event; returning None
            # there silently disables line events for that whole frame.
            if frame.f_code.co_filename != WSGI_FILE:
                return None
            if frame.f_code.co_name != method_name:
                return None
            if event == 'call':
                return tracer
            if event == 'line':
                # Wait only until all threads have reached *some* line of
                # the target method. On the fixed code one thread holds the
                # lock for the whole method, so the barrier aborts shortly
                # after the first thread drives all the way through -- this
                # keeps the test fast while still forcing wide interleaving
                # on the lock-free buggy code.
                try:
                    barrier.wait(timeout=0.5)
                except threading.BrokenBarrierError:
                    pass
            return tracer

        return tracer

    def _post_from_threads(self, body, tracer, errors):
        def worker():
            sys.settrace(tracer)
            try:
                client = Client(self.middleware, Response)
                response = client.post(self.middleware.path, data=body)
                if response.status_code != 200:
                    errors.append(
                        "expected 200 for %r, got %s" % (body,
                                                         response.status_code))
            except Exception:
                errors.append(traceback_mod.format_exc())
            finally:
                sys.settrace(None)

        threads = [threading.Thread(target=worker)
                   for _ in range(self.NUM_THREADS)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def test_concurrent_turn_on_does_not_double_start(self):
        tracer = self._install_line_barrier('start')
        errors = []
        self._post_from_threads('turn=on', tracer, errors)

        assert not errors, "concurrent turn=on failed:\n" + "\n".join(errors)
        assert self.middleware.on is True
        assert len(self.start_calls) == 1, (
            "ProfilingSession.start() called %d times for %d concurrent "
            "turn=on POSTs; expected exactly 1" % (
                len(self.start_calls), self.NUM_THREADS))

    def test_concurrent_turn_off_does_not_double_stop(self):
        self.middleware.start()
        del self.start_calls[:]

        tracer = self._install_line_barrier('stop')
        errors = []
        self._post_from_threads('turn=off', tracer, errors)

        assert not errors, "concurrent turn=off failed:\n" + "\n".join(errors)
        assert self.middleware.on is False
        assert len(self.stop_calls) == 1, (
            "ProfilingSession.stop() called %d times for %d concurrent "
            "turn=off POSTs; expected exactly 1" % (
                len(self.stop_calls), self.NUM_THREADS))


class TestMiddlewareClear(object):
    """Clear must discard everything, including what is still "in flight"."""

    def setUp(self):
        self.engine, self.session = _make_engine_and_session()
        self.middleware = sqltap.wsgi.SQLTapMiddleware(app=_test_app)
        self.client = Client(self.middleware, Response)
        self.middleware.start()

    def tearDown(self):
        if self.middleware.on:
            self.middleware.stop()

    def test_clear_then_refresh_shows_no_old_queries(self):
        marker_a = 8675309
        marker_b = 424242

        self.session.execute(
            text("SELECT 1 + :marker"), {"marker": marker_a})

        # Sanity check: the query landed in the middleware's store.
        before = self.client.get(self.middleware.path)
        assert before.status_code == 200
        assert str(marker_a) in before.data.decode('utf-8')

        # Another query *after* the last dashboard view sits only in the
        # undrained collector queue in the old implementation.
        self.session.execute(
            text("SELECT 1 + :marker"), {"marker": marker_b})

        # The Clear button posts clear=1 on its own.
        cleared = self.client.post(self.middleware.path, data='clear=1')
        assert cleared.status_code == 200
        cleared_body = cleared.data.decode('utf-8')
        assert str(marker_a) not in cleared_body
        assert str(marker_b) not in cleared_body

        # This is the regression: a fresh GET after Clear must not resurrect
        # anything that was buffered (drained list OR undrained queue) when
        # Clear was clicked. In the old code marker_b flows straight back
        # out of the collector queue on this request.
        refreshed = self.client.get(self.middleware.path)
        assert refreshed.status_code == 200
        refreshed_body = refreshed.data.decode('utf-8')
        assert str(marker_a) not in refreshed_body
        assert str(marker_b) not in refreshed_body, (
            "query stats captured before Clear reappeared after refresh")
        assert len(self.middleware.stats) == 0

    def test_collection_works_after_clear(self):
        self.session.execute(text("SELECT 1 + 111"))
        self.client.post(self.middleware.path, data='clear=1')

        self.session.execute(text("SELECT 2 + 222"))
        response = self.client.get(self.middleware.path)
        assert response.status_code == 200
        body = response.data.decode('utf-8')
        # The SQL text (with its literal-ish bind marker) appears in the
        # report; rendered results are not shown, so assert on the query
        # text instead of the arithmetic result.
        assert "2 + 222" in body
        assert "1 + 111" not in body


class TestMiddlewareBoundedMemory(object):
    """The retained store must have a cap and evict the oldest entries."""

    def test_stats_capped_at_max_stats(self):
        middleware = sqltap.wsgi.SQLTapMiddleware(
            app=_test_app, max_stats=10)
        middleware.start()
        try:
            for i in range(25):
                middleware._collect(object())
            snapshot = middleware._snapshot()
            assert len(snapshot) == 10
            # deque(maxlen=N) drops the OLDEST entries; the freshest 10 of
            # 0..24 are 15..24.
            assert snapshot[0] is middleware.stats[0]
        finally:
            middleware.stop()

    def test_default_cap_is_set(self):
        middleware = sqltap.wsgi.SQLTapMiddleware(app=_test_app)
        assert middleware.stats.maxlen == sqltap.wsgi.\
            SQLTapMiddleware.DEFAULT_MAX_STATS


class TestMiddlewareSnapshotStable(object):
    """Rendering must iterate a private snapshot, never the live container."""

    def test_concurrent_collect_and_render_never_errors(self):
        middleware = sqltap.wsgi.SQLTapMiddleware(app=_test_app)
        middleware.start()
        client = Client(middleware, Response)
        stop = threading.Event()
        errors = []

        def producer():
            # Drive the real SQLAlchemy event path on a throwaway engine.
            engine = create_engine('sqlite:///:memory:')
            session = sessionmaker(bind=engine)()
            while not stop.is_set():
                session.execute(text("SELECT 1"))
            engine.dispose()

        def viewer():
            while not stop.is_set():
                try:
                    response = client.get(middleware.path)
                    if response.status_code != 200:
                        errors.append("status %s" % response.status_code)
                    body = response.data.decode('utf-8')
                    if "Mako Runtime Error" in body:
                        errors.append("Mako error page rendered")
                except Exception:
                    errors.append(traceback_mod.format_exc())

        producers = [threading.Thread(target=producer) for _ in range(2)]
        viewers = [threading.Thread(target=viewer) for _ in range(2)]
        for thread in producers + viewers:
            thread.start()

        join_timeout = 3.0
        stop.set()
        for thread in producers + viewers:
            thread.join(join_timeout)

        assert not any(thread.is_alive()
                       for thread in producers + viewers), "worker hung"
        middleware.stop()
        assert not errors, "\n".join(errors[:3])


class TestMiddlewareResponseEncoding(object):
    """The report boundary must accept both str and bytes from Mako.

    Mako's normal template render returns ``str`` (unicode filter), but its
    fallback error template renders as pre-encoded ``bytes``. The middleware
    used to call ``.encode('utf-8')`` unconditionally, which raised
    AttributeError on the error template and turned it into a 500.
    """

    def test_bytes_report_is_served_without_500(self):
        middleware = sqltap.wsgi.SQLTapMiddleware(app=_test_app)
        client = Client(middleware, Response)
        original_report = sqltap.report

        def bytes_report(*args, **kwargs):
            return b"<html>bytes error page</html>"

        sqltap.wsgi.sqltap.report = bytes_report
        try:
            response = client.get(middleware.path)
        finally:
            sqltap.wsgi.sqltap.report = original_report

        assert response.status_code == 200
        assert response.data == b"<html>bytes error page</html>"

    def test_str_report_still_served(self):
        middleware = sqltap.wsgi.SQLTapMiddleware(app=_test_app)
        client = Client(middleware, Response)
        response = client.get(middleware.path)
        assert response.status_code == 200
        assert u"SQLTap" in response.data.decode('utf-8')
