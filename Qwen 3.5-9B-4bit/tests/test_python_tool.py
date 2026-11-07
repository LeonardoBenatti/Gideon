import asyncio
from pathlib import Path
import sys
import unittest

# Permette l'esecuzione dei test sia dalla cartella tests sia dalla radice del progetto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import server


class RestrictedPythonTests(unittest.TestCase):
    def run_code(self, code):
        output, ok, details = server.run_python_snippet(code)
        self.assertEqual(details["code"], code[:server.MAX_PYTHON_CODE_CHARS])
        return output, ok

    def test_exact_arithmetic_and_print(self):
        output, ok = self.run_code("print(123456789 * 987654321)\n123456789 * 987654321")
        self.assertTrue(ok)
        self.assertIn("121932631112635269", output)

    def test_print_does_not_report_its_none_return_as_a_second_result(self):
        output, ok = self.run_code("result = 23**2 * 4 / 5.24\nprint(result)")
        self.assertTrue(ok)
        self.assertIn("403.816", output)
        self.assertNotIn("None", output)
        self.assertNotIn("Output di print()", output)

    def test_count_characters_with_loop(self):
        output, ok = self.run_code('text = "banana"\ncount = 0\nfor ch in text:\n    if ch == "a":\n        count += 1\ncount')
        self.assertTrue(ok)
        self.assertIn("3", output)

    def test_safe_standard_library_modules(self):
        code = "from math import sqrt\nfrom statistics import mean\nfrom decimal import Decimal, ROUND_HALF_UP\nimport re\nprint(sqrt(81), mean([2, 4, 6]))\nprint(Decimal('1.005').quantize(Decimal('0.01'), ROUND_HALF_UP))\nprint(re.findall(r'\\w+', 'una parola'))"
        output, ok = self.run_code(code)
        self.assertTrue(ok, output)
        self.assertIn("9.0 4", output)
        self.assertIn("1.01", output)
        self.assertIn("parola", output)

    def test_large_ranges_are_rejected(self):
        output, ok = self.run_code("for n in range(1000000000):\n    pass")
        self.assertFalse(ok)
        self.assertIn("range supera il limite", output)

    def test_explicit_calculation_requests_are_routed_to_python(self):
        for text in ("quanto fa 4 x 23", "calcola il 15% di 80", "conta le lettere della parola banana", "4 * 23"):
            with self.subTest(text=text):
                self.assertTrue(server.should_require_python(text))
        self.assertFalse(server.should_require_python("Ciao, come stai?"))
        self.assertFalse(server.should_require_python("quanto fa 4 x 23 senza Python"))

    def test_python_tool_works_when_web_is_disabled(self):
        result, ok, details = asyncio.run(server.run_tool("run_python", {"code": "2 + 2"}, None))
        self.assertTrue(ok)
        self.assertIn("4", result)
        self.assertEqual(details["code"], "2 + 2")

    def test_unbounded_loops_stop_at_iteration_limit(self):
        output, ok = self.run_code("while True:\n    pass")
        self.assertFalse(ok)
        self.assertIn("limite di iterazioni", output)

    def test_files_network_processes_and_dynamic_execution_are_unavailable(self):
        for code in (
            "open('segreto.txt').read()",
            "import os",
            "import socket",
            "import subprocess",
            "eval('1 + 1')",
            "__import__('os')",
            "(1).__class__",
        ):
            with self.subTest(code=code):
                output, ok = self.run_code(code)
                self.assertFalse(ok)
                self.assertTrue(output.startswith("Errore:"))

    def test_code_and_output_limits(self):
        output, ok = self.run_code("#" + "x" * (server.MAX_PYTHON_CODE_CHARS + 1))
        self.assertFalse(ok)
        self.assertIn("supera il limite", output)

    def test_expansive_string_operations_are_rejected_before_allocation(self):
        for code in (
            "format('x', '100000000s')",
            "'-'.join(['x' * 50000] * 10000)",
            "import re\nre.sub('a', 'x' * 256, 'a' * 10000)",
        ):
            with self.subTest(code=code):
                output, ok = self.run_code(code)
                self.assertFalse(ok)
                self.assertTrue("limit" in output or "troppo lungo" in output, output)

    def test_regex_backtracking_constructs_are_rejected(self):
        output, ok = self.run_code("import re\nre.findall('(a|aa)+$', 'a' * 10000)")
        self.assertFalse(ok)
        self.assertIn("gruppi o alternanze", output)

    def test_deeply_nested_expressions_are_rejected(self):
        deep_ast = '1'
        for _ in range(120):
            deep_ast = f'(1 + {deep_ast})'
        output, ok = self.run_code(deep_ast)
        self.assertFalse(ok)
        self.assertIn("Espressione troppo complessa", output)


if __name__ == "__main__":
    unittest.main()
