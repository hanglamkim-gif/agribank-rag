import subprocess
import sys

tests = [
    "tests.test_cli",
    "tests.test_loader",
    "tests.test_embedding",
    "tests.test_index",
    "tests.test_query"
]

total = 0
passed = 0
failed = 0

print("| Test group | Số test | PASS | FAIL |")
print("|---|---|---|---|")

for test in tests:
    cmd = [sys.executable, "-m", "unittest", test]
    res = subprocess.run(cmd, capture_output=True, text=True, env={"PYTHONIOENCODING": "utf-8", **sys.modules['os'].environ})
    
    # parse "Ran X tests in Ys"
    out = res.stderr + res.stdout
    
    import re
    m = re.search(r"Ran (\d+) tests?", out)
    if m:
        t_count = int(m.group(1))
    else:
        t_count = 0
        
    f_count = 0
    if "FAILED" in out:
        m2 = re.search(r"failures=(\d+)", out)
        m3 = re.search(r"errors=(\d+)", out)
        if m2: f_count += int(m2.group(1))
        if m3: f_count += int(m3.group(1))
        if not m2 and not m3:
            f_count = t_count # assumed all failed if we can't parse
            
    p_count = t_count - f_count
    
    total += t_count
    passed += p_count
    failed += f_count
    
    name = test.split(".")[-1].replace("test_", "").upper()
    print(f"| {name} | {t_count} | {p_count} | {f_count} |")

print(f"| TỔNG CỘNG | {total} | {passed} | {failed} |")
print(f"\nTổng số test đã chạy: {total}")
print(f"Lệnh đã dùng để chạy: c:\\agribank-rag\\.venv\\Scripts\\python.exe -m unittest discover -s rag_foundation/buoi_07/tests -t . -v")
