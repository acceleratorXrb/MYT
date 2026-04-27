from types import SimpleNamespace

from ultralytics.engine.trainer import should_validate_epoch


def test_val_period_runs_every_n_epochs_and_final_epoch():
    args = SimpleNamespace(val=True, val_period=10)

    assert should_validate_epoch(args, epoch=0, final_epoch=False, possible_stop=False, stop=False)
    assert should_validate_epoch(args, epoch=9, final_epoch=False, possible_stop=False, stop=False)
    assert not should_validate_epoch(args, epoch=10, final_epoch=False, possible_stop=False, stop=False)
    assert should_validate_epoch(args, epoch=11, final_epoch=True, possible_stop=False, stop=False)


def test_val_false_disables_periodic_validation_but_keeps_stop_checks():
    args = SimpleNamespace(val=False, val_period=10)

    assert not should_validate_epoch(args, epoch=9, final_epoch=False, possible_stop=False, stop=False)
    assert should_validate_epoch(args, epoch=2, final_epoch=True, possible_stop=False, stop=False)
    assert should_validate_epoch(args, epoch=2, final_epoch=False, possible_stop=True, stop=False)
