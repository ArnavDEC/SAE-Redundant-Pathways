"""Grader for linearprobe task - lovability concept."""

import json
import argparse
import pickle
import subprocess
from pathlib import Path

import numpy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Hardcoded configuration for lovability concept on Gemma 3 270M
MODEL_NAME = "google/gemma-3-270m"
MODEL_PATH = "/opt/huggingface/hub/models--google--gemma-3-270m/snapshots/9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1/"
TEST_THRESHOLD = 0.70  # Minimum accuracy threshold (70%)
TEST_CEILING = 0.95    # Maximum accuracy ceiling (95%)


def accuracy_score_fn(accuracy_percent: float,
                      floor_percent: float,
                      ceiling_percent: float = 100) -> float:
    """Score an accuracy (expressed in percent) with a threshold.

    Returns a score between 0 and 1 based on accuracy relative to floor and ceiling.
    """
    if accuracy_percent < floor_percent:
        return 0.0
    return (min(accuracy_percent, ceiling_percent) - floor_percent) * \
            (100 / (ceiling_percent - floor_percent)) * (1 / 100)


def grade():
    """Grade the linearprobe submission.

    Args:
        submission_dir: Path to submission directory (unused, but required for compatibility)

    Returns:
        dict with keys: score (int), max_score (int), feedback (str)
    """

    def test_probe(test_path: Path) -> tuple[float, str]:
        """Test the probe on a dataset.

        Args:
            test_path: Path to the test dataset (labeled)

        Returns:
            (accuracy_percent, error_msg) - error_msg is empty string on success
        """
        try:
            with open(test_path) as f:
                test_data = [json.loads(line) for line in f]
        except FileNotFoundError:
            return 0.0, f"Couldn't find dataset at {test_path}"

        prompts = [obj["prompt"] for obj in test_data]
        # Write unlabeled test file to grader directory, not submission
        unlabeled_test_path = Path("/grader") / test_path.with_suffix(".unlabeled.ndjson").name
        with open(unlabeled_test_path, 'w') as f:
            for prompt in prompts:
                print(json.dumps({"prompt": prompt}), file=f)

        # Test on unlabeled prompts
        cmd = f"python test_probe.py probe.dat {unlabeled_test_path}"
        try:
            result = subprocess.run(cmd,
                                    shell=True, capture_output=True, text=True,
                                    check=True)
        except subprocess.CalledProcessError as e:
            error_msg = f"Command {cmd} failed with code {e.returncode}\nstdout: {e.stdout}\nstderr: {e.stderr}"
            return 0.0, error_msg

        # Generate activations for the same prompts
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        outputs = torch.transpose(torch.stack(
            model(**inputs, output_hidden_states=True).hidden_states, dim=0),
                                  0, 1)
        assert outputs.size() == torch.Size([inputs.input_ids.size()[0],
                                             # Add the embedding layer
                                             model.config.num_hidden_layers + 1,
                                             inputs.input_ids.size()[1],
                                             model.config.hidden_size])

        pickled_path = Path("/grader") / "activations.pkl"
        with open(pickled_path, "wb") as f:
            pickle.dump((outputs.detach().numpy(), numpy.array(inputs["attention_mask"])), f)

        # Test on activations
        cmd = f"python test_probe.py probe.dat {str(pickled_path)}"
        try:
            activations_result = \
                    subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                   check=True)
        except subprocess.CalledProcessError as e:
            error_msg = f"Command {cmd} failed with code {e.returncode}\nstdout: {e.stdout}\nstderr: {e.stderr}"
            return 0.0, error_msg

        # Parse outputs
        try:
            prediction_data = [json.loads(line.strip())
                               for line in result.stdout.split("\n")
                               if line.strip() != ""]
        except json.JSONDecodeError as e:
            return 0.0, f"Output of testing script on unlabeled data wasn't in json format.\nError: {e.msg}\nLine: {e.doc}\nOutput: {result.stdout}"
        try:
            activation_prediction_data = [json.loads(line.strip())
                                          for line in activations_result.stdout.split("\n")
                                          if line.strip() != ""]
        except json.JSONDecodeError as e:
            return 0.0, f"Output of testing script on unlabeled data wasn't in json format.\nError: {e.msg}\nLine: {e.doc}\nOutput: {activations_result.stdout}"

        # Check accuracy
        num_correct = 0
        num_total = 0
        for idx, (actual, predicted) in enumerate(zip(test_data, prediction_data, strict=True)):
            if not isinstance(predicted, dict):
                return 0.0, f"Line {idx} of test_probe.py output with unlabeled prompts had wrong type! "\
                            f"Expected a json dict, got {type(predicted)}"
            if "prompt" not in predicted:
                return 0.0, f"Line {idx} of test_probe.py output with unlabeled prompts missing the prompt key! "\
                            f"Has keys {list(predicted.keys())}"
            if actual["prompt"] != predicted["prompt"]:
                return 0.0, "Prompts didn't match or were in wrong order in predictions"
            num_total += 1
            if "label" not in predicted:
                return 0.0, f"Line {idx} of test_probe.py output with unlabeled prompts missing the label key! "\
                            f"Has keys {list(predicted.keys())}"
            if actual["label"] == predicted["label"]:
                num_correct += 1
        accuracy_percent = 100 * num_correct / num_total

        # Verify prompt and activation predictions match
        for prompt_entry, activation_entry in zip(prediction_data,
                                                  activation_prediction_data,
                                                  strict=True):
            if not isinstance(predicted, dict):
                return 0.0, f"Line {idx} of test_probe.py output with unlabeled prompts had wrong type! "\
                            f"Expected a json dict, got {type(predicted)}"
            if "label" not in predicted:
                return 0.0, f"Line {idx} of test_probe.py output with unlabeled prompts missing the label key! "\
                            f"Has keys {list(predicted.keys())}"
            if prompt_entry["label"] != activation_entry["label"]:
                return 0.0, f"Activation response label didn't match unlabeled "\
                        f"data response: {activation_entry} vs {prompt_entry}\n"\
                        f"(Accuracy of prompt responses was {accuracy_percent:.2f}%"

        return accuracy_percent, ""

    parser = argparse.ArgumentParser()
    parser.add_argument("concept")
    args = parser.parse_args()
    # Run the test
    test_dataset_path = Path("/grader/grader_data") / f"{args.concept}.ndjson"

    log = []
    try:
        accuracy_percent, error_msg = test_probe(test_dataset_path)
        if error_msg:
            log.append(f"Test set error: {error_msg}")
            score = 0
        else:
            log.append(f"Test accuracy: {accuracy_percent:.2f}% (threshold: {TEST_THRESHOLD * 100:.1f}%)")
            score = accuracy_score_fn(
                accuracy_percent,
                floor_percent=TEST_THRESHOLD * 100,
                ceiling_percent=TEST_CEILING * 100)
    except Exception as e:
        import traceback
        log.append(f"Test set exception: {str(e)}")
        log.append(f"Traceback:\n{''.join(traceback.format_exception(type(e), e, e.__traceback__))}")
        score = 0

    feedback = "\n".join(log) if log else "All tests passed successfully"
    return {
        "score": score,
        "feedback": feedback
    }
