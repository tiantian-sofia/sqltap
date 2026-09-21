from __future__ import absolute_import

import collections
import threading

try:
    import urllib.parse as urlparse
except ImportError:
    import urlparse
from . import sqltap

from werkzeug.wrappers import Response


class SQLTapMiddleware(object):
    """ SQLTap dashboard middleware for WSGI applications.

    For example, if you are using Flask::

        app.wsgi_app = SQLTapMiddleware(app.wsgi_app)

    And then you can use SQLTap dashboard from ``/__sqltap__`` page (this
    path prefix can be set by ``path`` parameter).

    :param app: A WSGI application object to be wrap.
    :param path: A path prefix for access. Default is `'/__sqltap__'`
    :param max_stats: Maximum number of :class:`~sqltap.QueryStats` objects
        retained for the dashboard. ``None`` (or any non-positive value)
        removes the cap. Defaults to :data:`DEFAULT_MAX_STATS`.
    """

    # Memory budget for retained QueryStats.
    #
    # Every QueryStats carries a full traceback.extract_stack() frame list,
    # so an entry is much heavier than it looks. 10k entries is plenty for a
    # debugging dashboard (the report already aggregates duplicate queries)
    # while bounding worst-case memory regardless of how long profiling stays
    # switched on or how rarely anybody opens the page. The cap is a
    # constructor argument rather than a hard-coded constant because the
    # right number depends on query rate and dashboard usage.
    DEFAULT_MAX_STATS = 10000

    def __init__(self, app, path='/__sqltap__',
                 max_stats=DEFAULT_MAX_STATS):
        self.app = app
        self.path = path.rstrip('/')

        # One re-entrant lock guards all state touched concurrently by
        # business threads (via the SQLAlchemy after_execute event hook) and
        # dashboard request threads.
        #
        # Why the lock is needed: the old start()/stop() were plain
        # check-then-act sequences. Two simultaneous POSTs (double click,
        # browser replay, two people) could both observe self.on == False and
        # both call ProfilingSession.start(); the session's own state machine
        # answers the second call with AssertionError ("Profiling session is
        # already started!") and the user sees a 500. stop() had the mirror
        # race. Folding the check, the profiler call and the flag flip into
        # one critical section makes the switch idempotent -- a repeated
        # request becomes a no-op instead of an error.
        #
        # RLock rather than Lock is defense in depth: no critical section
        # currently nests another locked method, but a re-entrant lock makes
        # that future refactor fail-safe.
        self._lock = threading.RLock()

        self.on = False

        # A single bounded store replaces the old "unbounded hand-off Queue
        # drained, only on dashboard visits, into an unbounded list" design.
        # The old design had two failure modes:
        #
        #   1. With profiling on and nobody viewing the page, the Queue
        #      grew for the entire life of the process; after the first
        #      view, self.stats grew forever too.
        #   2. Clear did `del self.stats[:]` but left entries buffered in
        #      the Queue, so the next request drained them straight back
        #      ("cleared" queries reappeared on refresh).
        #
        # Passing the bounded store directly to ProfilingSession as the
        # collect_fn removes the intermediate Queue entirely, so there is
        # never stale data that can flow back after a Clear.
        #
        # Eviction policy: drop the OLDEST entries (deque maxlen). The
        # dashboard answers "what are the most recent queries doing";
        # silently discarding fresh queries under load would make the report
        # lie about current behavior. Bounded deque.append is O(1) and never
        # blocks, so the event hook on business threads can never stall a
        # real query, and memory is bounded even when the dashboard is never
        # opened (each append both enters and evicts under the cap).
        self.max_stats = max_stats if max_stats and max_stats > 0 else None
        self.stats = collections.deque(maxlen=self.max_stats)
        self.profiler = sqltap.ProfilingSession(collect_fn=self._collect)

    def _collect(self, query_stats):
        """ collect_fn handed to ProfilingSession.

        Invoked on business threads from the after_execute event hook, so it
        must be cheap and must never raise. Bounded ``deque.append`` under
        the lock gives both guarantees; when full, the oldest entry is
        evicted atomically.
        """
        with self._lock:
            self.stats.append(query_stats)

    def _snapshot(self):
        """ Return a stable, point-in-time copy of the retained stats.

        Business threads keep appending while the Mako report renders, and
        the report itself rewrites QueryStats.stack_text while iterating.
        Rendering against the live container could therefore raise
        ("deque mutated during iteration") or produce a half-updated page
        (which surfaced as a Mako error page). Copying under the lock
        decouples rendering from concurrent collection. The copy is shallow
        on purpose: QueryStats is effectively immutable apart from
        stack_text, which the reporter deterministically overwrites for
        every entry on every render.
        """
        with self._lock:
            return list(self.stats)

    def _clear(self):
        """ Atomically discard every retained QueryStats.

        With no intermediate Queue anymore, nothing buffered exists from
        which "cleared" entries could reappear on the next request.
        """
        with self._lock:
            self.stats.clear()

    def __call__(self, environ, start_response):
        path = environ.get('PATH_INFO', '')
        if path == self.path or path == self.path + '/':
            return self.render(environ, start_response)
        return self.app(environ, start_response)

    def start(self):
        with self._lock:
            if not self.on:
                self.profiler.start()
                # Flip the flag only once the profiler really started: if
                # start() ever raised, the state is unchanged and a later
                # retry must still be allowed.
                self.on = True

    def stop(self):
        with self._lock:
            if self.on:
                self.profiler.stop()
                self.on = False

    def render(self, environ, start_response):
        verb = environ.get('REQUEST_METHOD', 'GET').strip().upper()
        if verb not in ('GET', 'POST'):
            response = Response('405 Method Not Allowed', status=405,
                                mimetype='text/plain')
            response.headers['Allow'] = 'GET, POST'
            return response(environ, start_response)

        # handle on/off switch
        if verb == 'POST':
            try:
                clen = int(environ.get('CONTENT_LENGTH', '0'))
            except ValueError:
                clen = 0
            body = environ['wsgi.input'].read(clen).decode('utf-8')
            body = urlparse.parse_qs(body)
            if body.get('clear', None):
                # Clear takes precedence and requires no turn parameter:
                # the Clear button posts `clear=1` on its own, and the
                # original implementation returned before validating turn.
                # Unlike the old code we still fall through to the
                # snapshot/render below -- clearing first and then
                # reporting over the fresh (empty) snapshot is what stops
                # queued entries from reappearing on the next request.
                self._clear()
            else:
                turn = body.get('turn', ' ')[0].strip().lower()
                if turn not in ('on', 'off'):
                    response = Response('400 Bad Request: parameter '
                                        '"turn=(on|off)" required',
                                        status='400', mimetype='text/plain')
                    return response(environ, start_response)
                if turn == 'on':
                    self.start()
                else:
                    self.stop()

        # Snapshot after any state change so the response shows exactly the
        # stats visible at this instant; collection during rendering goes to
        # the live deque and shows up on the next view.
        stats = self._snapshot()
        return self.render_response(environ, start_response, stats)

    def render_response(self, environ, start_response, stats=None):
        if stats is None:
            stats = self._snapshot()
        html = sqltap.report(stats, middleware=self, report_format="wsgi")
        # sqltap.report normally returns ``str`` (Mako renders with the
        # 'unicode' filter), but when rendering raises, Mako substitutes its
        # own error template, whose default ``output_encoding`` makes
        # ``render()`` return pre-encoded ``bytes`` instead. An unconditional
        # ``html.encode('utf-8')`` then raised AttributeError and turned the
        # Mako error page itself into an opaque 500 -- the "dashboard
        # sometimes shows a Mako error" symptom seen in production. Accept
        # either type here so the error page is actually delivered (and a
        # stable snapshot already removes the mutation that caused it).
        if isinstance(html, bytes):
            html_bytes = html
        else:
            html_bytes = html.encode('utf-8')
        response = Response(html_bytes, mimetype="text/html")
        return response(environ, start_response)
