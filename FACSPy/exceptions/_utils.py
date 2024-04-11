
class GateNotProvidedError(Exception):

    def __init__(self,
                 gate):
        self.message = (
            f"No gate was provided to the function. The input was {gate}"
        )
        super().__init__(self.message)


class ExhaustedHierarchyError(Exception):

    def __init__(self,
                 gate):
        self.message = (
            "You have reached the maximum of the gating hierarchy. "
            f"No Parent could be defined for {gate}"
        )
        super().__init__(self.message)


