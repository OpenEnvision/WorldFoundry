class Metric:
    """Generic Metric class"""

    def __init__(self):
        # To be implemented by each metric
        self.name = None

    def compute_metric(
        self,
        train_feat,
        test_feat,
        gen_feat,
    ):
        """Computes the metric value for the given sets of features (TO BE IMPLEMENTED BY EACH METRIC)
        - train_feat: Features from set of samples used to train generative model
        - test_feat: Features from test samples
        - gen_feat: Features from generated samples

        returns: Metric value
        """
