# test_my_module.py

from map_number import map_number
import unittest
import utils

class TestMapNumber(unittest.TestCase):
    def test_map_number(self):
        out1 = map_number(-600, -500, 500, 100, 200)
        out2 = map_number(-500, -500, 500, 100, 200)
        out3 = map_number(0, -500, 500, 100, 200)
        out4 = map_number(250, -500, 500, 100, 200)
        out5 = map_number(500, -500, 500, 100, 200)
        out6 = map_number(600, -500, 500, 100, 200)
        out7 = map_number(0, 0, 1, 1 ,0)
        utils.debug_print(out7)

        self.assertEqual(out1, 100)
        self.assertEqual(out2, 100)
        self.assertEqual(out3, 150)
        self.assertEqual(out4, 175)
        self.assertEqual(out5, 200)
        self.assertEqual(out6, 200)
        self.assertEqual(out7, 1)

if __name__ == '__main__':
    unittest.main()
