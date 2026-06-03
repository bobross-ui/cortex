from pathlib import Path
import pytest
from cortex.ingestion import resolve, UnknownExportError
from cortex.ingestion.twitter import TwitterParser


FIX = Path(__file__).parent / "fixtures" / "twitter"


def test_resolve_picks_twitter():
    assert isinstance(resolve(FIX), TwitterParser)


def test_unknown_export_raises(tmp_path):
    with pytest.raises(UnknownExportError):
        resolve(tmp_path)
