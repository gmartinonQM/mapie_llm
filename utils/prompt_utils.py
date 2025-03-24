import numpy as np
from sklearn.preprocessing import LabelEncoder



class TransformData:
    base_prompt = '''The following is a multiple-choice question about reading comprehension. You should answer the question based on the given context and you can use commonsense reasoning when necessary. Please reason step-by-step and select the correct answer. You only need to output the option.\n\n'''
    
    few_shot_exp_ids = [1, 3, 5, 7, 9]
    def __init__(self, data):
        self.data = data
        self.label_encoder = LabelEncoder()

    def _get_fewshot_exps(self):
        # extract demonstrations 
        fewshot_exps = []
        for idx in self.few_shot_exp_ids:
            fewshot_exps.append(self.data[idx])
            assert self.data[idx]["id"] == idx
        return fewshot_exps

    @staticmethod
    def _format_example(example, prompt, with_answer=False):
        prompt += "Context: " + example["context"] + "\n" + "Question: " + example["question"] + "\nChoices:\n"
        for k, v in example["choices"].items():
            prompt += k + ". " + str(v) + "\n"
        prompt += "Answer:"
        if with_answer:
            prompt += " " + example["answer"] + "\n"   
        return prompt


    def _format_base_prompt(self, example, fewshot_exps):
        prompt = ""
        for fs_exp in fewshot_exps:
            prompt = self._format_example(fs_exp, prompt, with_answer=True)

        prompt = self._format_example(example, prompt)
        return prompt

    def _fit_label_encoder(self):
        possible_answers = []

        for qa in self.data:
            choices = qa["choices"].keys()
            possible_answers += list(choices)
        possible_answers = sorted(list(set(possible_answers)))
        self.label_encoder.fit(possible_answers)
    
    def transform_data(self):
        self._fit_label_encoder()
        X, Y = [], []
        fs_examples = self._get_fewshot_exps()
        for question_answer in self.data:
            prompt = self._format_base_prompt(question_answer, fs_examples)
            X.append(prompt)
            Y.append(self.label_encoder.transform([question_answer["answer"]]))
        
        return np.array(X), np.array(Y).flatten()