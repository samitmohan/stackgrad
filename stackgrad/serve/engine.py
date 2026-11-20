import itertools
import queue
import threading

from .scheduler import Request


class Engine:
    """A background thread steps the scheduler under `lock`; each request's tokens stream back through a queue."""
    def __init__(self, scheduler, lock):
        self.scheduler, self.lock = scheduler, lock
        self.inbox, self.outputs, self.ids = queue.Queue(), {}, itertools.count()
        threading.Thread(target=self._loop, daemon=True, name="engine").start()

    def submit(self, prompt, max_tokens, temperature=0.7, top_p=0.95, eos_id=None):
        """Queue a request. Returns a queue that receives each new token, then None (or an exception if a step failed)."""
        req = Request(next(self.ids), list(prompt), max_tokens, temperature, top_p, eos_id)
        if not self.scheduler.fits(req):
            raise ValueError(f"{len(prompt)} prompt tokens plus {max_tokens} new ones do not fit")
        out = self.outputs[req.id] = queue.Queue()
        self.inbox.put(req)
        return out

    def _loop(self):
        s = self.scheduler
        while True:
            s.add(self.inbox.get())
            while s.has_work():
                while not self.inbox.empty():
                    s.add(self.inbox.get_nowait())
                try:
                    with self.lock:
                        emitted = s.step()
                except Exception as e:  # noqa: BLE001 - every caller must hear about it, or it waits forever
                    # self.outputs, not s.running: it also holds a request popped from waiting whose prefill raised
                    for rid in list(self.outputs):  # snapshot: submit() may add one from another thread
                        s.blocks.release(rid)
                        s.runner.release(rid)
                        self.outputs.pop(rid).put(e)
                    s.waiting.clear()
                    s.running.clear()
                    while not self.inbox.empty():  # submitted during the failed step: tell any caller still listening
                        out = self.outputs.pop(self.inbox.get_nowait().id, None)
                        if out is not None:
                            out.put(e)
                    continue
                for req, token in emitted:
                    out = self.outputs.get(req.id)
                    if out is None:  # submitted while a step failed: its caller already has the error
                        continue
                    out.put(token)
                    if req.done:
                        self.outputs.pop(req.id).put(None)
