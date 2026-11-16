#!/usr/bin/env python3
"""Small SQLite FIFO that spreads one extraction stage's segments over several GPU workers.

Used by scripts/run_split_extraction.sh. Each row is `pending`, `running` (claimed by a worker) or `done`. A worker
marks a row done only after it has finished the segment, so a worker that crashes leaves its row `running`, and
`requeue` returns such rows to `pending` for a fresh process.

Usage:
    python -m gtf.extraction.action100m.stage_queue init --db Q.sqlite --stage sam3 --segment-file SEG_LIST.txt
    python -m gtf.extraction.action100m.stage_queue status --db Q.sqlite --stage sam3
    python -m gtf.extraction.action100m.stage_queue requeue --db Q.sqlite --stage sam3   # prints the number of rows requeued
"""

import argparse
import os
import socket
import sqlite3
import time
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    stage TEXT NOT NULL,
    seq INTEGER NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    worker TEXT,
    claimed_at REAL,
    PRIMARY KEY (stage, name)
);
CREATE INDEX IF NOT EXISTS tasks_stage_status ON tasks(stage, status, seq);
"""


class StageQueue:
    def __init__(self, path, stage, worker=None):
        self.path = str(path)
        self.stage = stage
        self.worker = worker or f"{socket.gethostname()}:{os.getpid()}"
        self.connection = sqlite3.connect(self.path, timeout=120, isolation_level=None)
        self.connection.execute("PRAGMA busy_timeout=120000")

    def claim(self):
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute(
                "SELECT name FROM tasks WHERE stage=? AND status='pending' "
                "ORDER BY seq LIMIT 1",
                (self.stage,),
            ).fetchone()
            if row is None:
                self.connection.execute("COMMIT")
                return None
            name = row[0]
            self.connection.execute(
                "UPDATE tasks SET status='running', worker=?, claimed_at=? "
                "WHERE stage=? AND name=? AND status='pending'",
                (self.worker, time.time(), self.stage, name),
            )
            self.connection.execute("COMMIT")
            return name
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise

    def complete(self, name):
        self.connection.execute(
            "UPDATE tasks SET status='done' WHERE stage=? AND name=? AND worker=?",
            (self.stage, name, self.worker),
        )

    def close(self):
        self.connection.close()


def initialize(path, stage, names):
    """Replace the stage's rows with `names` (duplicates dropped, order kept), all pending."""
    path = Path(path)
    names = list(dict.fromkeys(names))
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=120)
    # The queue files live in the run's log directory, which can be on a network
    # filesystem. SQLite's WAL mode needs shared memory, which SQLite does not
    # support there (it can corrupt the file). A rollback journal is slower by
    # milliseconds and safe for a small queue with a few writers on one host.
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA busy_timeout=120000")
    connection.executescript(SCHEMA)
    connection.execute("DELETE FROM tasks WHERE stage=?", (stage,))
    connection.executemany(
        "INSERT INTO tasks(stage, seq, name, status) VALUES (?, ?, ?, 'pending')",
        [(stage, index, name) for index, name in enumerate(names)],
    )
    connection.commit()
    integrity = connection.execute("PRAGMA integrity_check").fetchone()
    if integrity != ("ok",):
        connection.close()
        raise sqlite3.DatabaseError(f"stage queue failed integrity check: {integrity}")
    connection.close()


def requeue_running(path, stage):
    """Return the stage's `running` rows (left by workers that failed) to `pending`; returns how many."""
    connection = sqlite3.connect(path, timeout=120)
    cursor = connection.execute(
        "UPDATE tasks SET status='pending', worker=NULL, claimed_at=NULL "
        "WHERE stage=? AND status='running'",
        (stage,),
    )
    connection.commit()
    count = cursor.rowcount
    connection.close()
    return count


def counts(path, stage):
    connection = sqlite3.connect(path, timeout=120)
    rows = connection.execute(
        "SELECT status, COUNT(*) FROM tasks WHERE stage=? GROUP BY status", (stage,)
    ).fetchall()
    connection.close()
    return dict(rows)


def iter_queued_stage_names(path, stage, worker, allowed_names):
    """Yield the names this worker claims until the queue is empty (claimed names outside allowed_names are marked
    done and skipped)."""
    queue = StageQueue(path, stage, worker)
    allowed = set(allowed_names)
    try:
        while True:
            name = queue.claim()
            if name is None:
                return
            if name not in allowed:
                queue.complete(name)
                continue
            # Completion comes after the yield on purpose: the row is acknowledged only when the loop body has
            # finished and asks for the next name. If the consumer raises or the worker process dies, the row stays
            # `running` and run_split_extraction.sh requeues it for a fresh process; marking it done in a finally
            # block would lose a segment that left the CUDA context unusable.
            yield name
            queue.complete(name)
    finally:
        queue.close()


def read_names(path):
    return [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def main():
    parser = argparse.ArgumentParser(prog="python -m gtf.extraction.action100m.stage_queue")
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("init")
    init.add_argument("--db", required=True)
    init.add_argument("--stage", required=True)
    init.add_argument("--segment-file", required=True)
    status = subparsers.add_parser("status")
    status.add_argument("--db", required=True)
    status.add_argument("--stage", required=True)
    requeue = subparsers.add_parser("requeue")
    requeue.add_argument("--db", required=True)
    requeue.add_argument("--stage", required=True)
    args = parser.parse_args()

    if args.command == "init":
        initialize(args.db, args.stage, read_names(args.segment_file))
    elif args.command == "requeue":
        print(requeue_running(args.db, args.stage))
    else:
        values = counts(args.db, args.stage)
        print(" ".join(f"{key}={values[key]}" for key in sorted(values)))


if __name__ == "__main__":
    main()
