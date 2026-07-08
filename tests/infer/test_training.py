from satnogs_decoder.infer import corpus, training
from satnogs_decoder.infer.layout import FieldSpan

def test_build_training_rows_shapes(tmp_path):
    conn = corpus.open_corpus(str(tmp_path / "c.db"))
    frames = [bytes([i & 0xFF, (i >> 8) & 0xFF, 0x00]) for i in range(60)]
    corpus.insert_frames(conn, 111, frames)
    corpus.insert_layout(conn, 111, [
        FieldSpan(0, 2, 2, False, False), FieldSpan(2, 3, 1, True, True),
    ])
    Xb, yb, Xf, ys, ye = training.build_training_rows(conn, [111])
    assert Xb.shape[0] == 3 and yb.shape[0] == 3          # one row per byte position
    assert yb.tolist() == [1, 0, 1]                       # starts at 0 and 2
    assert Xf.shape[0] == 2 and ys.tolist() == [0, 1] and ye.tolist() == [0, 1]
