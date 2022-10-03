class LCG:
    a = 3
    b = 0
    m = 31
    def __init__(self, seed: int) -> None:
        self.state = seed % self.m
    
    def __call__(self) -> float:
        self.state = ((self.a * self.state + self.b) % self.m)
        return self.state / self.m