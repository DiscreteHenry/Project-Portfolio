import unittest

from crystal_yolo import Box, boxes_for_tile, labelme_shape, tile_positions, yolo_row


class GeometryTests(unittest.TestCase):
    def test_yolo_row_converts_rectangle_to_normalized_center_format(self):
        self.assertEqual(yolo_row(Box(10, 20, 30, 60), 100, 100), "0 0.20000000 0.40000000 0.20000000 0.40000000")

    def test_tiles_cover_final_edge_without_duplicate_position(self):
        self.assertEqual(tile_positions(1000, 512, 128), [0, 384, 488])

    def test_tile_boxes_use_center_assignment_and_clip_at_edges(self):
        boxes = [Box(450, 10, 550, 110), Box(520, 10, 620, 110)]
        self.assertEqual(boxes_for_tile(boxes, 0, 0, 512, 512), [Box(450, 10, 512, 110)])
        self.assertEqual(boxes_for_tile(boxes, 512, 0, 512, 512), [Box(8, 10, 108, 110)])

    def test_labelme_shape_preserves_rectangle_and_confidence(self):
        shape = labelme_shape(Box(1.2, 3.4, 5.6, 7.8), confidence=0.87654321)
        self.assertEqual(shape["label"], "Protein Crystal")
        self.assertEqual(shape["points"], [[1.2, 3.4], [5.6, 7.8]])
        self.assertEqual(shape["flags"]["pseudo_confidence"], 0.876543)
