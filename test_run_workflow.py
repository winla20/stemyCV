"""Check the bundled-input CLI and subprocess working directory without ML inference."""
import unittest
from unittest.mock import patch

import run_workflow


class WorkflowEntryTests(unittest.TestCase):
    def test_default_inputs_are_bundled_and_subprocesses_use_repo_root(self):
        with patch('sys.argv', ['run_workflow.py', '--skip-transcription']), patch('run_workflow.subprocess.run') as execute:
            run_workflow.main()
        first = execute.call_args_list[0]
        self.assertEqual(first.args[0][2], str(run_workflow.ROOT / 'data/demo'))
        self.assertEqual(first.args[0][first.args[0].index('--color-policy')+1],'reinterpret_sdr_bt709')
        self.assertTrue(all(call.kwargs['cwd'] == run_workflow.ROOT for call in execute.call_args_list))
        self.assertFalse(any('workflow_transcribe.py' in str(call.args[0]) for call in execute.call_args_list))

    def test_explicit_input_directory_is_preserved(self):
        args = run_workflow.parse_args(['--source-dir', 'another_dataset'])
        self.assertEqual(str(args.source_dir), 'another_dataset')

    def test_device_and_output_reach_stages(self):
        with patch('sys.argv',['run_workflow.py','--device','cpu','--speech-device','cpu','--output','output/test_routing']), patch('run_workflow.subprocess.run') as execute:
            run_workflow.main()
        for call in execute.call_args_list:
            command=call.args[0]
            if command[1].endswith(('workflow_extract.py','workflow_transcribe.py')):
                self.assertEqual(command[command.index('--device')+1],'cpu')
            self.assertTrue(any(str(value).endswith('test_routing') for value in command))

    def test_cannot_write_inference_over_bundled_results(self):
        with patch('sys.argv',['run_workflow.py','--output',str(run_workflow.ROOT/'demo_results')]), patch('run_workflow.subprocess.run') as execute:
            with self.assertRaises(SystemExit):run_workflow.main()
        execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
