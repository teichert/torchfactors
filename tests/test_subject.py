from dataclasses import dataclass

import pytest
from torchfactors import Subject


def test_subject_nodataclass():
    with pytest.raises(ValueError):
        class MySubject(Subject):
            pass

        s = MySubject()


def test_subject_nodataclass():
    @dataclass
    class MySubject(Subject):
        pass

    s = MySubject()
