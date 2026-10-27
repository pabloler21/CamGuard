import numpy as np

from camguard.events import Event, build_events
from camguard.zone import feet_in_zone

zone = np.zeros((10, 10), bool)
zone[5:, 5:] = True
feet = np.array([[7, 7], [2, 2], [9.9, 9.9], [50, 50]])  # adentro, afuera, borde, fuera del cuadro
assert feet_in_zone(feet, zone).tolist() == [True, False, True, True]

# fps=5: #1 entra en 0 s y sale en 1 s; #2 entra, sale 2 s y vuelve (dos eventos); un hueco de 1 cuadro no corta
events = build_events({1: [0, 1, 2, 3, 4], 2: [10, 12, 30]}, fps=5)
assert events == [Event(1, 0.0, 1.0), Event(2, 2.0, 2.6), Event(2, 6.0, 6.2)], events
assert build_events({}, fps=5) == []
print("ok")
