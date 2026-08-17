from PIL import Image

from comas.tiling import TiledScorer, tile_boxes, tile_is_positive


def test_tiles_cover_the_whole_image():
    W, H = 1440, 900
    boxes = tile_boxes(W, H)
    # every pixel must belong to at least one tile, or signal could be missed
    assert min(b[0] for b in boxes) == 0
    assert min(b[1] for b in boxes) == 0
    assert max(b[2] for b in boxes) == W
    assert max(b[3] for b in boxes) == H


def test_tiles_are_square_and_scale_with_height():
    small = tile_boxes(1440, 900)
    large = tile_boxes(3600, 2338)
    s = small[0]
    l = large[0]
    assert (s[2] - s[0]) == (s[3] - s[1])
    # a Retina capture gets proportionally bigger tiles, so a tile always
    # covers the same amount of interface
    assert (l[2] - l[0]) > (s[2] - s[0])


def test_tiles_overlap():
    boxes = tile_boxes(1440, 900, overlap=0.35)
    xs = sorted({b[0] for b in boxes})
    side = boxes[0][2] - boxes[0][0]
    assert xs[1] - xs[0] < side          # stride shorter than the tile


def test_tile_positive_when_it_contains_the_region():
    region = [{"kind": "ghost", "box": [100, 100, 400, 120]}]
    assert tile_is_positive((0, 0, 500, 500), region)
    assert not tile_is_positive((600, 600, 1100, 1100), region)


def test_thin_region_counts_by_its_own_area_not_the_tiles():
    # a ghost line is long and thin and will never fill a square tile;
    # measuring against tile area would label every such tile negative
    region = [{"kind": "ghost", "box": [0, 0, 400, 15]}]
    assert tile_is_positive((0, 0, 800, 800), region)


def test_partial_overlap_below_threshold_is_negative():
    region = [{"kind": "ghost", "box": [0, 0, 400, 20]}]
    # only a quarter of the line falls inside
    assert not tile_is_positive((300, 0, 800, 500), region, min_frac=0.5)


def test_tile_inside_a_large_panel_is_positive():
    # a chat panel is taller than any tile, so no tile can hold half of it;
    # a tile sitting entirely within the panel still clearly shows evidence
    panel = [{"kind": "panel", "box": [1000, 60, 1400, 900]}]
    assert tile_is_positive((1050, 200, 1350, 500), panel)


def test_tile_beside_a_large_panel_is_negative():
    panel = [{"kind": "panel", "box": [1000, 60, 1400, 900]}]
    assert not tile_is_positive((0, 200, 300, 500), panel)


class _FakeScorer:
    """Stands in for the CNN: scores a tile high if it contains any red."""

    def __init__(self):
        self.seen = 0

    def score_images(self, images):
        self.seen += len(images)
        return [0.99 if any(c[0] > 200 and c[1] < 60 for c in im.getdata())
                else 0.02
                for im in images]


def test_tiled_scorer_takes_the_max_and_reports_the_box(tmp_path):
    img = Image.new("RGB", (1200, 900), (0, 0, 0))
    # a red patch that only one tile region will contain
    for x in range(820, 1000):
        for y in range(620, 800):
            img.putpixel((x, y), (255, 0, 0))
    p = tmp_path / "shot.png"
    img.save(p)

    scorer = TiledScorer(_FakeScorer())
    out = scorer.score_batch([p])
    assert out[str(p)] == 0.99            # max over tiles, not average
    box = scorer.box_for(p)
    assert box is not None
    # the reported tile must actually contain the patch
    assert box[0] <= 900 <= box[2] and box[1] <= 700 <= box[3]


def test_tiled_scorer_handles_unreadable_file(tmp_path):
    missing = tmp_path / "nope.png"
    scorer = TiledScorer(_FakeScorer())
    out = scorer.score_batch([missing])
    assert out[str(missing)] == 0.0
    assert scorer.box_for(missing) is None
