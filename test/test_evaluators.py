from unittest import TestCase

from models import evaluators as ev

class TestEvaluators(TestCase):
    def test_confusion_matrix_index(self):        
        conf_mat = ev.build_confusion_matrix([1,1,0], [1,0,1], [0,1])
        indexes = conf_mat.index.to_list()
        self.assertEqual(indexes, [('prediction',0),('prediction',1)])

    def test_confusion_matrix_columns(self):        
        conf_mat = ev.build_confusion_matrix([1,1,0], [1,0,1], [0,1])
        indexes = conf_mat.columns.to_list()
        self.assertEqual(indexes, [('actuals',0),('actuals',1)])
