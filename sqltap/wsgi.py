from __future__ import absolute_import

import threading

try:
    import urllib.parse as urlparse
except ImportError:
    import urlparse
from . import sqltap
try:
    import queue
except ImportError:
    import Queue as queue

from werkzeug.wrappers import Response


class SQLTapMiddleware(object):
    """ SQLTap dashboard middleware for WSGI applications.

    For example, if you are using Flask::

        app.wsgi_app = SQLTapMiddleware(app.wsgi_app)

    And then you can use SQLTap dashboard from ``/__sqltap__`` page (this
    path prefix can be set by ``path`` parameter).

    :param app: A WSGI application object to be wrap.
    :param path: A path prefix for access. Default is `'/__sqltap__'`
    :param max_queue: Maximum number of :class:`QueryStats` buffered in the
        collector queue before new arrivals are dropped. Default is 1000.
        ``0`` means unbounded (not recommended, see notes below).
    :param max_stats: Maximum number of :class:`QueryStats` retained for
        rendering. When the cap is exceeded the *oldest* entries are
        discarded. Default is 10000. ``0`` means unbounded (not
        recommended, see notes below).

    Concurrency and memory notes
    ----------------------------

    This middleware is shared by every thread of the WSGI worker it is
    installed in (e.g. all gunicorn gthread workers' threads), so all
    mutable state is guarded by a single lock:

    * **On/off is idempotent and atomic.** ``start()``/``stop()`` check
      and flip the ``self.on`` flag while holding the lock, so two
      concurrent ``turn=on`` POSTs (double click, browser retry, two
      operators) can no longer both pass the check and both call
      ``ProfilingSession.start()``, which used to raise
      ``AssertionError`` and surface as a 500. Repeated toggles are now
      harmless no-ops.

    * **The collector queue is bounded and never blocks producers.**
      The ``collect_fn`` callback runs on *application* threads inside
      SQLAlchemy's ``after_execute`` hook, so it must never block or
      raise -- a full buffer must not slow down or break the application
      we are merely observing. When the queue is full the incoming
      (newest) item is dropped and counted in ``self.dropped``. A full
      queue means nobody is looking at the dashboard anyway.

    * **Retained stats are bounded, keep-newest.** ``self.stats`` is
      trimmed to ``max_stats`` after each drain, discarding the oldest
      entries. A live dashboard is almost always used to look at recent
      activity, and each :class:`QueryStats` carries a full traceback,
      so unbounded retention is a real memory leak on busy services.

    * **Draining happens only on dashboard access.** The queue is
      drained into ``self.stats`` when the dashboard page is rendered.
      This is deliberate: the dashboard is the sole consumer, and now
      that both structures are bounded, leaving profiling on with no
      viewers can no longer grow memory without limit, so no background
      thread is needed.

    * **Rendering uses a stable snapshot.** The report is rendered from
      a copy of ``self.stats`` taken under the lock, so a concurrent
      drain/append can no longer mutate the list mid-render (which used
      to occasionally produce a Mako error page).
    """

    def __init__(self, app, path='/__sqltap__', max_queue=1000,
                 max_stats=10000):
        self.app = app
        self.path = path.rstrip('/')
        self.on = False
        self.max_queue = max_queue
        self.max_stats = max_stats
        self.dropped = 0
        self.collector = queue.Queue(max_queue)
        self.stats = []
        # Guards self.on transitions and every mutation of self.stats.
        self._lock = threading.Lock()
        self.profiler = sqltap.ProfilingSession(collect_fn=self._collect)

    def _collect(self, qstats):
        """ ``collect_fn`` for the profiling session.

        Runs on application threads inside SQLAlchemy's ``after_execute``
        hook, so it must never block or raise. When the bounded queue is
        full, drop the incoming item and count it in ``self.dropped``.
        """
        try:
            self.collector.put_nowait(qstats)
        except queue.Full:
            self.dropped += 1

    def __call__(self, environ, start_response):
        path = environ.get('PATH_INFO', '')
        if path == self.path or path == self.path + '/':
            return self.render(environ, start_response)
        return self.app(environ, start_response)

    def start(self):
        with self._lock:
            if self.on:
                return
            # Flip the flag only after the transition succeeded, so a
            # failure cannot leave the middleware in a state that
            # disagrees with the profiler.
            self.profiler.start()
            self.on = True

    def stop(self):
        with self._lock:
            if not self.on:
                return
            self.profiler.stop()
            self.on = False

    def _drain(self):
        """ Pop everything currently in the collector queue.

        Returns the items as a list. Called with ``self._lock`` held so
        that "clear stats + discard queue" is atomic with respect to
        "drain queue + append to stats".
        """
        items = []
        try:
            while True:
                items.append(self.collector.get(block=False))
        except queue.Empty:
            pass
        return items

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
            clear = body.get('clear', None)
            if clear:
                with self._lock:
                    del self.stats[:]
                    # Also discard everything still pending in the
                    # collector queue, otherwise the next render would
                    # drain it straight back into self.stats and the
                    # cleared queries would reappear on refresh.
                    self._drain()
                return self.render_response(environ, start_response)

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

        with self._lock:
            self.stats.extend(self._drain())
            # Keep-newest trim: a dashboard is used to inspect recent
            # activity, so when the cap is exceeded the oldest entries
            # are the least valuable ones.
            if self.max_stats > 0:
                overflow = len(self.stats) - self.max_stats
                if overflow > 0:
                    del self.stats[:overflow]

        return self.render_response(environ, start_response)

    def render_response(self, environ, start_response):
        # Render from a stable snapshot taken under the lock; the report
        # iterates the stats and must not see concurrent mutations.
        with self._lock:
            stats = list(self.stats)
        html = sqltap.report(stats, middleware=self, report_format="wsgi")
        response = Response(html.encode('utf-8'), mimetype="text/html")
        return response(environ, start_response)
