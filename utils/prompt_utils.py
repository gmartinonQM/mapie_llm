import numpy as np
from sklearn.preprocessing import LabelEncoder



class TransformCosmosQA:
    """
    Transform CosmosQA data into a format that can be used for prompt-based learning.
    This class takes as input the CosmosQA data which has the following structure:
    [
        {
            "id": int,
            "context": str,
            "question": str,
            "choices": {
                "A": str,
                "B": str,
                "C": str,
                "D": str
            },
            "answer": str
        },
        ...
    ]

    The output of this class is a tuple of two numpy arrays (X, Y) where X is the input prompt and Y is the label.

    The input prompt is a string that contains the context, question, and choices. The context is the same for
    all the few-shot examples and the question-answer pair. The choices are the same for the question-answer pair.
    The context, question, and choices are separated by newlines. The answer is not included in the input prompt.
    """
    base_prompt = '''The following is a multiple-choice question about reading comprehension. You should answer the question based on the given context and you can use commonsense reasoning when necessary. Please reason step-by-step and select the correct answer. You only need to output the option.\n\n'''
    
    few_shot_exp_ids = [1, 3, 5, 7, 9]
    def __init__(self, data):
        self.data = data
        self.label_encoder = LabelEncoder()

    def _get_fewshot_exps(self):
        """
        Retrieve the few-shot examples from the data.
        """
        fewshot_exps = []
        for idx in self.few_shot_exp_ids:
            fewshot_exps.append(self.data[idx])
            assert self.data[idx]["id"] == idx
        return fewshot_exps

    @staticmethod
    def _format_example(example, prompt, with_answer=False):
        """
        Format a question-answer pair. The prompt contains the context, question, and choices.
        WHen this function is called with with_answer=True, the answer is included in the prompt.
        """
        prompt += "Context: " + example["context"] + "\n" + "Question: " + example["question"] + "\nChoices:\n"
        for k, v in example["choices"].items():
            prompt += k + ". " + str(v) + "\n"
        prompt += "Answer:"
        if with_answer:
            prompt += " " + example["answer"] + "\n"   
        return prompt


    def _format_prompt(self, example, fewshot_exps):
        """
        Format the prompt for a question-answer pair. The prompt contains the context,
        question, and choices.
        """
        prompt = ""
        for fs_exp in fewshot_exps:
            prompt = self._format_example(fs_exp, prompt, with_answer=True)

        prompt = self._format_example(example, prompt)
        return prompt

    def _fit_label_encoder(self):
        """
        Fit a label encoder to the possible answers.
        """
        possible_answers = []

        for qa in self.data:
            choices = qa["choices"].keys()
            possible_answers += list(choices)
        possible_answers = sorted(list(set(possible_answers)))
        self.label_encoder.fit(possible_answers)
    
    def transform_data(self):
        """
        Transform the data into a format that can be used for prompt-based learning.
        The transformation is done according to the following steps:
        1. Fit a label encoder to the possible answers.
        2. Retrieve the few-shot examples to put at the beginning of the prompt.
        3. Format the prompt for each question-answer pair.
        4. Transform the answers into labels using the label encoder.
        5. Return the input prompts and the labels as numpy arrays.
        """
        self._fit_label_encoder()
        X, Y = [], []
        fs_examples = self._get_fewshot_exps()
        for question_answer in self.data:
            prompt = self._format_prompt(question_answer, fs_examples)
            X.append(prompt)
            Y.append(self.label_encoder.transform([question_answer["answer"]]))
        
        return np.array(X), np.array(Y).flatten()