"""The web hub: a local dashboard for the live feed and the decision log.

Imported lazily by the app - it pulls in ``http.server`` and, on the encode
path, OpenCV. Nothing here can act on the user's machine; the hub is a
consumer of pipeline state exactly like the OpenCV overlay.
"""

from .server import DEFAULT_PORT, HOST, HubServer, port_is_free
from .state import HubState, utc_now

__all__ = ["DEFAULT_PORT", "HOST", "HubServer", "HubState", "port_is_free", "utc_now"]
