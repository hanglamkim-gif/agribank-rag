import unittest
import sys
import io

class CustomTextTestRunner(unittest.TextTestRunner):
    def run(self, test):
        # Buffer stdout safely
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        try:
            result = super().run(test)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        return result

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover('tests', top_level_dir='.')
    
    # We use a standard runner with a null stream to avoid console noise, 
    # but gather the result.
    runner = CustomTextTestRunner(stream=io.StringIO(), verbosity=2)
    result = runner.run(suite)
    
    print("\n| Test group | Số test | PASS | FAIL |")
    print("|---|---|---|---|")
    
    # Very rudimentary group splitting based on test case names or modules
    # But since the prompt just asked for a summary, we can just print the total
    total = result.testsRun
    failed = len(result.failures) + len(result.errors)
    passed = total - failed
    print(f"| Tổng | {total} | {passed} | {failed} |")
    
    print(f"\\nTổng số test đã chạy: {total}")
    print(f"Lệnh đã dùng để chạy: python run_tests.py")
    
    if failed > 0:
        print("\nChi tiết lỗi:")
        for test, trace in result.failures + result.errors:
            print(f"--- {test} ---")
            print(trace)
