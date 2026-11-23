import warnings
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


class FLD(Metric):
    def __init__(self, eval_feat="test", baseline_nll=None, gen_size=GEN_SIZE):
        super().__init__()

        # One of ("train", "test", "gap")
        # Corresponds to the set whose likelihood is evaluated with the MoG
        self.eval_feat = eval_feat
        self.name = f"FLD {eval_feat.title()}"

        # Corresponds to the likelihood of the test set under a MoG centered at half of the train set fit to the other half of the train set
        self.baseline_nll = baseline_nll

        self.gen_size = gen_size

    def get_nll_diff(self, nll, baseline_nll):
        return (nll - baseline_nll) * 100

    def compute_metric(self, train_feat, test_feat, gen_feat):
        if len(gen_feat) > self.gen_size:
            gen_feat = shuffle(gen_feat, min(len(gen_feat), self.gen_size))

        """Preprocess"""
        train_feat, test_feat, gen_feat = preprocess_feat(train_feat, test_feat, gen_feat)

        # Fit MoGs
        mog_gen = MoG(gen_feat)
        mog_gen.fit(train_feat)

        # Default eval_feat, fits a MoG centered at generated samples to train samples, then gets LL of test samples
        if self.eval_feat == "test":
            nlls = mog_gen.get_dim_adjusted_nlls(test_feat)
        # As above but evalutes the LL of train samples
        elif self.eval_feat == "train":
            nlls = mog_gen.get_dim_adjusted_nlls(train_feat)
        elif self.eval_feat == "gap":
            train_nll = nlls = mog_gen.get_dim_adjusted_nlls(train_feat).mean().item()
            test_nll = mog_gen.get_dim_adjusted_nlls(test_feat).mean().item()
            metric_val = self.get_nll_diff(train_nll, test_nll)
            if metric_val < -1_000:
                warnings.warn(
                    "Very high FLD gen gap value: your generated data is likely completely memorized.", stacklevel=1
                )
            return metric_val
        else:
            raise Exception(f"Invalid mode for FLD metric: {self.eval_feat}")

        nll = nlls.mean().item()

        if not self.baseline_nll:
            baseline_nll = self.get_baseline_nll(train_feat, test_feat, size=len(gen_feat))
        else:
            baseline_nll = self.baseline_nll

        metric_val = self.get_nll_diff(nll, baseline_nll)

        if metric_val > 1_000:
            warnings.warn("Very high FLD value, your generated data is likely completely memorized.", stacklevel=1)

        return metric_val

    def get_baseline_nll(self, train_feat, test_feat, size=GEN_SIZE):
        """Preprocess"""
        train_feat, test_feat, _ = preprocess_feat(train_feat, test_feat, train_feat)

        train_feat = shuffle(train_feat)

        # Fit MoGs
        split_size = min(size, len(train_feat) // 2)
        mog = MoG(train_feat[:split_size])
        mog.fit(train_feat[split_size:])

        baseline_nlls = mog.get_dim_adjusted_nlls(test_feat)
        return baseline_nlls.mean().item()
