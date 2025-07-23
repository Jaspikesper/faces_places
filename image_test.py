import unittest
from unittest.mock import mock_open, patch
import numpy as np
from image_loading import read_pgm

class TestReadPGM(unittest.TestCase):
    def setUp(self):
        # Create a valid PGM header and image data (square 12x12)
        self.pgm_data = b'P5\n12 12\n255\n' + bytes([100] * 144)

    @patch('builtins.open', new_callable=mock_open, read_data=b'P5\n12 12\n255\n' + bytes([100] * 144))
    def test_filename_only(self, mock_file):
        img = read_pgm('changed')
        self.assertEqual(img.shape, (12, 12))
        img2 = read_pgm('changed.pgm')
        self.assertEqual(img2.shape, (12, 12))

    @patch('builtins.open', new_callable=mock_open, read_data=b'P5\n12 12\n255\n' + bytes([100] * 144))
    def test_full_path(self, mock_file):
        img = read_pgm('face_data/s1/1.pgm')
        self.assertEqual(img.shape, (12, 12))

    @patch('builtins.open', new_callable=mock_open, read_data=b'P5\n12 12\n255\n' + bytes([100] * 144))
    def test_folder_and_photo_num(self, mock_file):
        img = read_pgm('face_data', 1, 1)
        self.assertEqual(img.shape, (12, 12))
        img2 = read_pgm('face_data/s1', 1, 1)
        self.assertEqual(img2.shape, (12, 12))
        img3 = read_pgm('face_data', 2, 3)
        self.assertEqual(img3.shape, (12, 12))

    def test_invalid_args(self):
        with self.assertRaises(ValueError):
            read_pgm('face_data', 1, None)
        with self.assertRaises(ValueError):
            read_pgm('face_data', None, 1)

if __name__ == '__main__':
    unittest.main()