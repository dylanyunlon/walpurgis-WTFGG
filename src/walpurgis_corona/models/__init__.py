# Corona models — lazy imports to avoid circular dependency
def _get_d2stgnn():
    from .model import D2STGNN
    return D2STGNN

def _get_trainer():
    from .trainer import trainer
    return trainer
