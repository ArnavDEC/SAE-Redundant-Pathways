import os, importlib.util

def grade():
    """Grade string reverse task."""
    solution_file = "solution.py"
    if not os.path.exists(solution_file):
        return {"score": 0, "feedback": "No solution.py file found"}
    
    try:
        spec = importlib.util.spec_from_file_location("solution", solution_file)
        solution = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(solution)
        
        if not hasattr(solution, 'reverse_words'):
            return {"score": 0.1, "feedback": "No reverse_words function found"}
        
        test_cases = [("hello world", "olleh dlrow"), ("The quick brown fox", "ehT kciuq nworb xof"), ("a", "a"), ("", ""), ("single", "elgnis")]
        
        score, failed = 0, []
        for inp, exp in test_cases:
            try:
                result = solution.reverse_words(inp)
                if result == exp: score += 20
                else: failed.append(f"'{inp}' -> Expected '{exp}', got '{result}'")
            except Exception as e:
                failed.append(f"'{inp}' -> Error: {e}")
        
        feedback = "All tests passed!" if not failed else f"Failed {len(failed)} tests: {'; '.join(failed)}"
        return {"score": score / 100, "feedback": feedback}
        
    except Exception as e:
        return {"score": 0.05, "feedback": f"Error loading: {e}"}
