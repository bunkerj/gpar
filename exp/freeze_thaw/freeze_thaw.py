from exp.freeze_thaw.model_aggregator import ModelAggregator


class FreezeThaw:
    def __init__(self, param_ranges):
        self.param_ranges = param_ranges
        self.models = []

    def run(self):
        pass
        """
        1) Define a range of suitable hyperparameters
        2) Use EI to compute a basket of model configurations
        3) Use ES to compute which model to train within the basket
        4) Get new observation by training the model
        5) Repeat
        """

    def plot(self):
        pass
