"""Check the bundled-input CLI and subprocess working directory without ML inference."""
import unittest
from unittest.mock import patch

import run_workflow


class WorkflowEntryTests(unittest.TestCase):
    def test_default_inputs_are_bundled_and_subprocesses_use_repo_root(self):
        with patch('sys.argv', ['run_workflow.py', '--skip-transcription']), patch('run_workflow.subprocess.run') as execute:
            run_workflow.main()
        first = execute.call_args_list[0]
        self.assertEqual(first.args[0][-1], str(run_workflow.ROOT / 'data/demo'))
        self.assertTrue(all(call.kwargs['cwd'] == run_workflow.ROOT for call in execute.call_args_list))
        self.assertFalse(any('workflow_transcribe.py' in str(call.args[0]) for call in execute.call_args_list))

    def test_explicit_input_directory_is_preserved(self):
        args = run_workflow.parse_args(['--source-dir', 'another_dataset'])
        self.assertEqual(str(args.source_dir), 'another_dataset')


if __name__ == '__main__':
    unittest.main()
