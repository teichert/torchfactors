import torch  # noqa
from numpy.lib.function_base import average  # noqa
from pytest import approx  # noqa
from torchmetrics.functional import f1, precision_recall  # noqa


def test_multi_label_f1():
    gold = [0, 1, 1, 2, 0, 0, 3, 3, 3, 4, 4, 0, 4, 4, 4, 4]
    pred = [1, 1, 0, 2, 2, 0, 0, 3, 0, 4, 4, 4, 0, 0, 0, 0]
    tps = [1, 1, 1, 2]
    gps = [2, 1, 3, 6]
    pps = [2, 2, 1, 3]
    micro_prec = sum(tps) / sum(pps)
    micro_rec = sum(tps) / sum(gps)
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec)
    macro_precs = [tp / pp for tp, pp in zip(tps, pps)]
    macro_recs = [tp / gp for tp, gp in zip(tps, gps)]
    # note: these are not computed like this
    # macro_prec = sum(macro_precs) / len(macro_precs)
    # macro_rec = sum(macro_recs) / len(macro_recs)
    # macro_f1 = 2 * macro_prec * macro_rec / (macro_prec + macro_rec)
    # like this instead:
    macro_f1s = [2 * p * r / (p + r) for p, r in zip(macro_precs, macro_recs)]
    macro_f1 = sum(macro_f1s) / len(macro_f1s)
    out_micro_f1 = f1(torch.tensor(pred), torch.tensor(
        gold), average='micro', num_classes=5, ignore_index=0)
    out_macro_f1 = f1(torch.tensor(pred), torch.tensor(
        gold), average='macro', num_classes=5, ignore_index=0)

    assert out_micro_f1 == approx(micro_f1)
    assert out_macro_f1 == approx(macro_f1)
