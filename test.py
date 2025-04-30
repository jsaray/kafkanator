
import unittest
from kafkanator import *

class TestCalculations(unittest.TestCase):

    def test_cluster(self):
        workers = pd.read_csv('./data/salaries.csv',sep=',',header=0)
        withGini = index_per_cluster(workers,'diploma','salary',index='gini')
        print ( ' the gini is ', withGini )
        withTheil_L = index_per_cluster(workers,'diploma','salary',index='theil-l')
        print ( ' the Theil L is ', withTheil_L )
        withTheil_T_baseE = index_per_cluster(workers,'diploma','salary',index='theil-t')
        print ( ' the Theil base e is ', withTheil_T_baseE )
        withTheil_T_base10 = index_per_cluster(workers,'diploma','salary',index='theil-t',**{'array_type':'gains','base_entropy':10} )
        print ( ' the Theil base 10 is ', withTheil_T_base10 )

if __name__ == '__main__':
    unittest.main()