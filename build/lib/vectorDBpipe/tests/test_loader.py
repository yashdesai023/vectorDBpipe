import os
from vectorDBpipe.data.loader import DataLoader

def test_load_txt(tmp_path):
    test_file = tmp_path / "sample.txt"
    test_file.write_text("Hello AI World!")
    loader = DataLoader(data_path=str(test_file))
    result = loader.load_data()
    assert isinstance(result, list)
    assert "Hello AI World!" in result[0]["content"]

