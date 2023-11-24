import time


class TimeContext:
    def __init__(self, timer, name, verbose):
        self.timer = timer
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, ex_type, value, traceback):
        self.end = time.time()
        dt = self.end - self.start
        self.timer._context_exit(self.name, dt, self.verbose)


class Timer:
    """
    A timer for measuring elapsed time statistics. The intended usage involves a with statement, like so:
        timer = Timer()
        with timer.time('iteration'):
            ... # do some stuff
        timer.averages['iteration']
    """
    def __init__(self, log, verbose=True):
        self.log = log
        self.verbose = verbose
        self._averages = {}
        self._counts = {}

    def __getitem__(self, item):
        return self._averages[item]

    def _context_exit(self, name, dt, verbose):
        if name in self._counts:
            old_n = self._counts[name]
            new_n = old_n + 1
            self._averages[name] = (old_n * self._averages[name] + dt) / new_n
            self._counts[name] = new_n
        else:
            self._averages[name] = dt
            self._counts[name] = 1

        verbose = self.verbose if verbose is None else verbose
        assert type(verbose) is bool
        if verbose:
            self.log(f'Completed {name} in {dt} seconds. Average = {self[name]}')

    def time(self, name, verbose=None):
        return TimeContext(self, name, verbose)
