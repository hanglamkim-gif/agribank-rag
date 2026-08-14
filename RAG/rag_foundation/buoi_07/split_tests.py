import re
from pathlib import Path

tests_dir = Path(r"c:\agribank-rag\RAG\rag_foundation\buoi_07\tests")

# We will read test_index.py and test_query.py and duplicate the merged test methods
# to match the exact numbering.

def split_tests(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()
    
    # find all def test_case_XX_YY_ZZ...
    pattern = re.compile(r"def test_case_((?:\d+_)+)(.*?)\(self\):")
    
    new_code = ""
    last_idx = 0
    
    for match in pattern.finditer(code):
        new_code += code[last_idx:match.start()]
        
        numbers_str = match.group(1).strip('_')
        suffix = match.group(2)
        numbers = numbers_str.split('_')
        
        # We need to extract the entire body of the function
        # to duplicate it for each number
        
        # Finding the end of the function body
        start_body = match.end()
        end_body = code.find("\n    def test_", start_body)
        if end_body == -1:
            end_body = len(code)
            
        body = code[start_body:end_body]
        
        for n in numbers:
            new_code += f"def test_case_{n}_{suffix}(self):"
            new_code += body
            new_code += "\n    "
            
        last_idx = end_body
        
    new_code += code[last_idx:]
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_code)
        
split_tests(tests_dir / "test_index.py")
split_tests(tests_dir / "test_query.py")
print("Split tests successfully.")
