from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..MoG import MoG, preprocess_feat
    from ..utils import shuffle
    from .Metric import Metric
else:
    from fld.metrics.Metric import Metric
    from fld.MoG import MoG, preprocess_feat
    from fld.utils import shuffle

GEN_SIZE = 10000
TEST_SIZE = 10000


class FLD_P(Metric):
    """Precision equivalent of FLD (i.e. model the density of the test and get likelihood of gen)"""

    def __init__(self, baseline_nll=None, gen_size=GEN_SIZE, test_size=TEST_SIZE):
        super().__init__()

        self.name = "FLD-P"

        # Corresponds to the likelihood of the test set under a MoG centered at half of the train set fit to the other half of the train set
        self.baseline_nll = baseline_nll

        self.gen_size = gen_size
        self.test_size = test_size

    def get_nll_diff(self, nll, baseline_nll):
        return (nll - baseline_nll) * 100

    def compute_metric(self, train_feat, test_feat, gen_feat):
        if len(gen_feat) > self.gen_size:
            gen_feat = shuffle(gen_feat, min(len(gen_feat), self.gen_size))

        if len(test_feat) > self.test_size:
            test_feat = shuffle(test_feat, min(len(test_feat), self.test_size))

        """Preprocess"""
        train_feat, test_feat, gen_feat = preprocess_feat(train_feat, test_feat, gen_feat)
        train_feat = shuffle(train_feat)

        # Fit MoG at test feat to train feat
        mog = MoG(test_feat)
        mog.fit(train_feat)

        # Get nll of gen_feat
        nlls = mog.get_dim_adjusted_nlls(gen_feat)
        nll = nlls.mean().item()

        baseline_nll = self.get_baseline_nll(train_feat, test_feat) if not self.baseline_nll else self.baseline_nll

        return self.get_nll_diff(nll, baseline_nll)

    def get_baseline_nll(self, train_feat, test_feat):
        """Preprocess"""
        train_feat, test_feat, _ = preprocess_feat(train_feat, test_feat, train_feat)
        train_feat = shuffle(train_feat)
        if len(test_feat) > self.test_size:
            test_feat = shuffle(test_feat, min(len(test_feat), self.test_size))

        # Fit MoGs
        split_size = min(self.gen_size, len(train_feat) // 2)
        mog = MoG(test_feat)
        mog.fit(train_feat[split_size:])

        baseline_nlls = mog.get_dim_adjusted_nlls(train_feat[:split_size])
        return baseline_nlls.mean().item()
