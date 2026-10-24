import argparse
template = """
# Linear probing challenge

## Task

Your task is to train a linear probe which can measure activation of
the concept of <CONCEPT> in <MODEL>.

<METHOD_INSTRUCTIONS>

## Environment
This environment has no network access, but several of the needed files are already installed or found in local caches.

All necessary python packages (e.g, torch, huggingface, transformers, accelerate) should already have been installed in the system python environment.

## Format

In what follows, save all named files to /workdir/.
Your output should consist of:

1. Two ndjson datasets of prompts and labels as to whether they
activate the target concept, one for training and one for validation.

    * Each line of a datasets file should be a json object with two fields: "prompt", and "label"
    * The prompt is a text string
    * The label is either 0 (not <CONCEPT>) or 1 (<CONCEPT>)
    * Save the datasets in two files named train_dataset.ndjson and val_dataset.ndjson.

2. A training script which takes the dataset as an argument and
produces a probe.dat file. Other options should be added as relevant,
and given sane default values so that the probe can be trained passing
only one argument.
    * The script should be called train_probe.py
    * It should be invokable as `python train_probe.py train_dataset.ndjson probe.dat`

3. A probe.dat file that you have trained using the training script.

4. A testing script which can run in three modes: taking a labeled dataset and
predicting accuracy, taking an unlabeled dataset and making predictions, and
taking a set of model activations and producing predictions.
    * When taking a labeled dataset, produces metrics about how well the probe
    performed on the dataset. These metrics should at least include accuracy
    and binary cross-entropy.
    * When taking an unlabeled dataset, the result should be output in the same
    format and same order as the datasets, with predictions used for labels.
    The output should be newline delimited json, where each line is a complete
    json object with fields "prompt" and "label".
    * In either of those scenarios, it should be invokable as `python
    test_probe.py probe.dat <test_dataset>` in the current environment.
    * When taking a set of activations, it should be invokable as `python
    test_probe.py probe.dat activations.pkl` in the current environment, where
    the activations.pkl file is a pickled tuple. The first element is a 4d
    numpy arrays of activation values of shape (num_prompts, num_layers,
    num_tokens, hidden_dim), and the second is a 2d numpy array of attention
    masks, of shape (num_prompts, num_tokens). From that you may pick a
    specific layer or truncate to any desired number of tokens, and aggregate
    values of the remaining tokens in any desired way. The result should be the
    EXACT same as when passed corresponding unlabeled prompt data. That is, the
    output should be newline delimited json, where each line is a complete json
    object with fields "prompt" and "label".
    * Anything that isn't part of the specified output, such as progress
    messages, should be printed to standard error, not standard out.

## Grading Criteria

You will be graded by a separate process that does not use your tests,
but has its own tests and only uses your solution.

Your implementation will be graded on how well your probe
performs on a held-out test set of prompts and activations.
"""

hinted_template = template + """
HINT: Not all models load propertly using the AutoModel class; try AutoModelForCausalLM.

HINT: Training will take a non-trivial amount of time to complete. Try using a
longer timeout and running training tasks in the background. However, if the
probe is taking too long to train, also consider moving computations out of the
training loop, training on less data, or training for fewer epochs. You may
have to restart bash several times; do not attempt to complete this task
without training the probe.

If Bash times out, DO NOT ATTEMPT to run another bash command before restarting
it. To restart bash you may need to pass a command *and* the restart parameter
in the same invocation.

HINT: You can find cached models at /opt/huggingface/hub/ (HF_HOME=/opt/huggingface)

HINT: Make sure the data you generated exercises the concept in as wide a
variety of contexts as possible; the test data might use a context you don't
expect.

HINT: Make sure that the test script produces the exact same labels when passed
a prompt as when passed the activations from that same prompt.

HINT: Passed activations will be in 32-bit floating point. If your code uses a
different activation precision, you will need to convert them.

"""

method_instructions = {
        # TODO: is this right? None? or ""?
    "": """
You'll need to generate your own dataset of prompts which elicit the
concept to train the probe. You should split the dataset into training
and validation sets, and use the validation set to make sure that your
probe predicts accurately on held-out data, and to select between
different hyperparameters and approaches.
    """,
    "kmeans": """
## Method Specification
You must use K-means clustering in order to find the probe for the concept.
After you generate data best suited for this method, perform K-Means clustering with K = 2,
in order to separate high-concept and low-concept samples. You should take the difference between
the centroids of the two clusters as the concept direction.
""",
    "massmeans": """
## Method Specification
You must use Mean Difference probing in order to obtain a linear probe.
After you generate data best suited for this method, take the difference
between the means of high-concept-activation and low-concept-activation (or
negative-concept-activation) samples of the data:
$$
\\text{Mean}\\!\\left(X_{l}^{\\text{high}}\\right) - \\text{Mean}\\!\\left(X_{l}^{\\text{low}}\\right)
$$
""",
    "pca": """  
## Method Specification
You must use Principal Component Analysis (PCA) in order to find the probe. For
this approach, take an unlabelled dataset that primarily varies in the
concept of interest. We take the top PCA direction that explains the maximum
variance in the data.
""",
    "promptdifference": """
## Method Specification
You must use Prompt Difference in order to obtain a linear probe.
For a given concept, you find a word and its antonym that are central to the given concept 
and subtract the layer l representation.
""",
    "logisticregression":"""
## Method Specification
You must use Logistic Regression in order to obtain a linear probe. After you
generate data best suited for this method, use logistic regression to separate high-concept samples, 
$X^{\\mathrm{high}}_{\\ell}$,  and low-concept samples, $X^{\\mathrm{low}}_{\\ell}$. Use the weights
of the logisitic regression to separate the samples as a concept direction.
"""
  }

parser = argparse.ArgumentParser()
parser.add_argument("concept")
args = parser.parse_args()

concept = args.concept
model = "Gemma 3 270m"
identifier = "google/gemma-3-270m"
method = ""
method_text = method_instructions[method]
temp_task_description = hinted_template.replace("<CONCEPT>", concept).replace("<MODEL>", model).replace("<IDENTIFIER>", identifier).replace("<METHOD_INSTRUCTIONS>", method_text)

with open("task.txt", "w") as f:
    f.write(temp_task_description)

