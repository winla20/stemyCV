"""Device policy tests run without PyTorch or accelerator hardware."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from workflow_runtime import select_device, precision, default_batch_size, peak_memory_mb, reset_peak_memory


def hardware(cuda=False, mps=False):
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:cuda,device_count=lambda:2,init=Mock(),
                           reset_peak_memory_stats=Mock(),max_memory_allocated=Mock(return_value=1024**2)),
                           backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda:mps)))


class RuntimeTests(unittest.TestCase):
    def test_auto_priority(self):
        for cuda,mps,expected in [(True,True,'cuda:0'),(False,True,'mps'),(False,False,'cpu')]:
            self.assertEqual(select_device(torch_module=hardware(cuda,mps)),expected)

    def test_explicit_unavailable_device_fails(self):
        for request in ['cuda:0','mps','bogus']:
            with self.assertRaises(ValueError):
                select_device(request,torch_module=hardware())
        self.assertEqual(select_device('1',torch_module=hardware(cuda=True)),'cuda:1')

    def test_cpu_and_mps_never_call_cuda_metrics(self):
        for device in ['cpu','mps']:
            hw=hardware()
            reset_peak_memory(device,torch_module=hw)
            self.assertIsNone(peak_memory_mb(device,torch_module=hw))
            hw.cuda.reset_peak_memory_stats.assert_not_called()
            hw.cuda.init.assert_not_called()
            hw.cuda.max_memory_allocated.assert_not_called()
            self.assertEqual(precision(device),32)
            self.assertEqual(default_batch_size(device),1)

    def test_whisper_uses_cpu_on_apple_gpu(self):
        self.assertEqual(select_device(torch_module=hardware(mps=True),allow_mps=False),'cpu')
        with self.assertRaises(ValueError):
            select_device('mps',torch_module=hardware(mps=True),allow_mps=False)

    def test_cuda_memory_reset_initializes_runtime_first(self):
        hw=hardware(cuda=True)
        calls=Mock()
        calls.attach_mock(hw.cuda.init,'initialize')
        calls.attach_mock(hw.cuda.reset_peak_memory_stats,'reset')
        reset_peak_memory('cuda:0',torch_module=hw)
        self.assertEqual([c[0] for c in calls.mock_calls],['initialize','reset'])
        hw.cuda.reset_peak_memory_stats.assert_called_once_with('cuda:0')


if __name__=='__main__':unittest.main()
