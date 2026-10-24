import subprocess, os, json, sys

def grade():
    """Grade 1+1=2 task."""
    # The grader runs with cwd=/grader, so reference the agent's file by
    # absolute /workdir path rather than a cwd-relative one.
    sol = "/workdir/solution.py"
    if not os.path.exists(sol):
        return {"score": 0, "feedback": "No solution.py file found"}

    try:
        result = subprocess.run(["python3", sol], cwd="/workdir", capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return {"score": 0, "feedback": f"Script failed: {result.stderr}"}
        
        output = result.stdout.strip()
        score = 1 if output == "2" else 0.5
        feedback = "Perfect!" if score == 1 else f"Expected '2', got '{output}'"
        return {"score": score, "feedback": feedback}
        
    except subprocess.TimeoutExpired:
        return {"score": 0, "feedback": "Solution timed out"}
    except Exception as e:
        return {"score": 0, "feedback": f"Error: {e}"}

# Allow running as standalone script
if __name__ == "__main__":
    result = grade()
    print(json.dumps(result, indent=2))
