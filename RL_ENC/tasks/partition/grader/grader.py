import subprocess, os, json, sys

def partition_count(n):
    """Calculate the number of integer partitions of n using dynamic programming."""
    if n < 0:
        return 0
    if n == 0:
        return 1
    
    # dp[i] will store the number of partitions of i
    dp = [0] * (n + 1)
    dp[0] = 1
    
    # For each number from 1 to n
    for num in range(1, n + 1):
        # Update dp[i] for all i >= num
        for i in range(num, n + 1):
            dp[i] += dp[i - num]
    
    return dp[n]

def grade():
    """Grade partition counting task."""
    if not os.path.exists("solution.py"):
        return {"score": 0, "max_score": 100, "feedback": "No solution.py file found"}
    
    # Test cases with known partition counts
    test_cases = [
        (0, 1),    # p(0) = 1
        (1, 1),    # p(1) = 1
        (2, 2),    # p(2) = 2: {2}, {1,1}
        (3, 3),    # p(3) = 3: {3}, {2,1}, {1,1,1}
        (4, 5),    # p(4) = 5: {4}, {3,1}, {2,2}, {2,1,1}, {1,1,1,1}
        (5, 7),    # p(5) = 7
        (10, 42),  # p(10) = 42
        (20, 627), # p(20) = 627
        (50, 204226), # p(50) = 204226
    ]
    
    total_score = 0
    max_score = len(test_cases) * 10
    feedback_parts = []
    
    for n, expected in test_cases:
        try:
            result = subprocess.run(
                ["python3", "solution.py", str(n)], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            
            if result.returncode != 0:
                feedback_parts.append(f"n={n}: Script failed with error: {result.stderr.strip()}")
                continue
            
            output = result.stdout.strip()
            
            try:
                actual = int(output)
                if actual == expected:
                    total_score += 10
                else:
                    feedback_parts.append(f"n={n}: Expected {expected}, got {actual}")
            except ValueError:
                feedback_parts.append(f"n={n}: Invalid output '{output}' (expected integer)")
                
        except subprocess.TimeoutExpired:
            feedback_parts.append(f"n={n}: Solution timed out")
        except Exception as e:
            feedback_parts.append(f"n={n}: Error - {e}")
    
    if total_score == max_score:
        feedback = "Perfect! All test cases passed."
    else:
        feedback = "Failed test cases:\n" + "\n".join(feedback_parts)
    
    return {
        "score": total_score / max_score, 
        "feedback": feedback,
    }

# Allow running as standalone script
if __name__ == "__main__":
    result = grade()
    print(json.dumps(result, indent=2))
