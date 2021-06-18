from dataclasses import dataclass

import pytest
from torchfactors import Subject


def test_subject_nodataclass():
    with pytest.raises(ValueError):
        class MySubject(Subject):
            pass

        MySubject()


def test_subject_good():
    @dataclass
    class MySubject(Subject):
        pass

    MySubject()
