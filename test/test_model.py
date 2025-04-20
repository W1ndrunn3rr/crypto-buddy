from src.model.lstm import MultiStepLSTM
from unittest import TestCase
import torch


class TestMultiStepLSTM(TestCase):
    def setUp(self):
        self.model = MultiStepLSTM(input_size=3, hidden_size=64)
        self.input_tensor = torch.randn(32, 5, 3)

    def test_forward_shape(self):
        output = self.model(self.input_tensor)
        self.assertEqual(output.shape, (32, 1, 1))
