
from calibration.metrics import ece,emitKey
import unittest
class GroupFairnessTest(unittest.TestCase):
	def test_emit_key_5_bins(self):
		predreal =[(0.12,0),(0.34,0),(0.35,1),(0.43,0),(0.43,1),(0.52,0),(0.55,1),(0.65,1),(0.72,0),(0.74,1),(0.91,0),(0.92,1),(0.94,0),(0.95,1)]
		bins = list ( map( lambda x : emitKey(x,5) , predreal ) )
		print ( ' 5 bins result ' )
		self.assertEqual(bins[0][0],0)
		self.assertEqual(bins[1][0],1)
		self.assertEqual(bins[2][0],1)
		print ( bins )


	def test_emit_key_10_bins(self):
		predreal =[(0.12,0),(0.34,0),(0.35,1),(0.43,0),(0.43,1),(0.52,0),(0.55,1),(0.65,1),(0.72,0),(0.74,1),(0.91,0),(0.92,1),(0.94,0),(0.95,1)]
		bins = list ( map( lambda x : emitKey(x,10) , predreal ) )
		print ( ' 10 bins result ' )
		self.assertEqual(bins[0][0],1)
		self.assertEqual(bins[1][0],3)
		self.assertEqual(bins[2][0],3)
		print ( bins )

	def test_emit_key_20_bins(self):
		predreal =[(0.12,0),(0.34,0),(0.35,1),(0.43,0),(0.43,1),(0.52,0),(0.55,1),(0.65,1),(0.72,0),(0.74,1),(0.91,0),(0.92,1),(0.94,0),(0.95,1)]
		bins = list ( map( lambda x : emitKey(x,20) , predreal ) )
		print ( ' 20 bins result ' )
		self.assertEqual(bins[0][0],2)
		self.assertEqual(bins[1][0],6)
		self.assertEqual(bins[2][0],7)
		print ( bins )

	def test_ece_1(self):
		print('ece 1 test')
		predreal =[(0.12,0),(0.34,0),(0.35,1),(0.43,0),(0.43,1),(0.52,0),(0.55,1),(0.65,1),(0.72,0),(0.74,1),(0.91,0),(0.92,1),(0.94,0),(0.95,1)]
		ese = ece (predreal,5)
		print('ese res ',ese)


