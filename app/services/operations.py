from threading import Event


class OperationCancelled(Exception):
    """Raised when a client asks the API to stop a long-running operation."""


def raise_if_cancelled(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise OperationCancelled
