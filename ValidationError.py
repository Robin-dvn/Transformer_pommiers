class ValidationError(Exception):
    """Exception levée lors d'une erreur de validation."""
    pass

class GPUOutOfMemoryError(Exception):
    """Exception levée lors d'une erreur de mémoire GPU."""
    pass
