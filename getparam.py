from dataclasses import dataclass
import numpy as np

# Turn this into a function that get_parameters(ver) -> dataclass TODO
@dataclass
class P:
    """

    """
    name:  str
    value:  float

    def vector(self, tenors) -> np.matrix:
        return np.repeat(self.value, tenors)


ver = '6.4'
with open("Parameters.txt") as f:
    for line in f:
        (key, val) = line.split(':')
        if eval(key)[0] == ver:
            P(eval(key)[2], val)


