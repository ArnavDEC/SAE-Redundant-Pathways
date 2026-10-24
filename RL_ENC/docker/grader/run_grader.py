#!/bin/python3

import json
import traceback
from datetime import UTC, datetime

try:
    import GRADER as grader  # type: ignore[import-not-found]
except ModuleNotFoundError:
    import grader  # type: ignore[import-not-found]


def utc_timestamp() -> str:
  # naive UTC, matching the format consumers already expect
  return datetime.now(UTC).replace(tzinfo=None).isoformat()


def main() -> None:
  # Load agent timing (may not exist for regrades of old containers)
  try:
    with open('/grader/timing.json') as f:
      timing = json.load(f)
  except FileNotFoundError:
    timing = {}
  timing['grader_start_time'] = utc_timestamp()

  try:
    grade = grader.grade()
  except Exception as e:
    # Do NOT write grade.json on grader infrastructure errors -- a score=0
    # from a crashed grader is not valid RL signal and must not be trained
    # against.  The missing grade.json and non-zero exit code signal that
    # this run should be discarded.
    timing['grader_end_time'] = utc_timestamp()
    print(f'Grading error: {e}')
    print(traceback.format_exc())
    raise
  if "score" not in grade:
    assert "subscores" in grade, \
            "Grade output must contain either 'score' or 'subscores' key!"
    assert isinstance(grade["subscores"], dict), \
            "'subscores' in grade output must be a dict, "\
            f"not {type(grade['subscores'])}"
    if "weights" not in grade:
      grade["weights"] = {subscore: 1/len(grade["subscores"])
                          for subscore in grade["subscores"].keys()}
    grade["score"] = sum([subscore_score * grade["weights"][subscore_name]
                          for subscore_name, subscore_score in
                          grade["subscores"].items()])

  # Record grader end time
  timing['grader_end_time'] = utc_timestamp()

  # Format grade with timing, preserving all fields from result
  grade.update(timing)  # Add all timing fields

  with open('/grader/grade.json', 'w') as f:
    json.dump(grade, f, indent=2)

if __name__ == "__main__":
    main()
