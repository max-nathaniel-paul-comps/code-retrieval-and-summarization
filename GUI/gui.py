import sys
from tkinter import *
from tkinter import ttk
import random
sys.path.append("../src")
from prod_bvae import ProductionBVAE
from ret_bvae import RetBVAE
import text_data_utils as tdu
sys.path.append("../baselines/IR")
import ir
sys.path.append("../baselines/RET-IR")
import retir


class GUI:
    def __init__(self, master):
        self.master = master
        self.AVAILABLE_LANGUAGES = ['C#', 'Python', 'Java']

        # self.label = Label(master, text="Code Retrieval and Summarization")
        # self.label.pack()

        self.language_frame = Frame(root)
        self.language_frame.pack()
        self.language_label = Label(self.language_frame, text="Select a language ")
        self.language_label.pack(side=LEFT)
        self.language_box = ttk.Combobox(self.language_frame, values=self.AVAILABLE_LANGUAGES)
        self.language_box.pack(side=LEFT)

        self.input = Text(master)
        self.input.config(font=('Arial', 13))
        self.input.pack()

        self.summary_button_frame = Frame(root)
        self.summary_button_frame.pack()

        self.retrieval_button_frame = Frame(root)
        self.retrieval_button_frame.pack()

        self.summarize_bt = Button(self.summary_button_frame, text="Generate Summary", command=self.bvae_summarize)
        self.summarize_bt.pack(side=LEFT)
        self.baseline_summarize_bt = Button(self.summary_button_frame, text="Generate Summary with Baseline Model",
                                            command=self.baseline_summarize)
        self.baseline_summarize_bt.pack(side=LEFT)

        self.retrieve_bt = Button(self.retrieval_button_frame, text="Retrieve Code", command=self.bvae_retrieve)
        self.retrieve_bt.pack(side=LEFT)
        self.baseline_retrieve_bt = Button(self.retrieval_button_frame, text="Retrieve Code with Baseline Model",
                                           command=self.baseline_retrieve)
        self.baseline_retrieve_bt.pack(side=LEFT)

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()

        self.output = Text(master)
        self.output.pack()
        self.output.config(font=('Arial', 13))

        self.models = {}

    def summarize(self, bvae_or_baseline):
        if not self.get_language():
            return

        code = self.input.get(1.0, END)

        if self.language == "Python":
            code = tdu.preprocess_user_generated_python(code)

        if bvae_or_baseline == "bvae":
            model = self.get_model("summ", self.language)
            if model is None:
                return
            summary = model.summarize([code], beam_width=1)[0]

        elif bvae_or_baseline == "baseline":
            if self.language == "C#":
                summaries, _ = tdu.load_iyer_file("../data/iyer_csharp/test.txt")
            elif self.language == "Python":
                _, _, test = tdu.load_edinburgh_dataset("../data/edinburgh_python")
                summaries = [ex[0] for ex in test]
            elif self.language == "Java":
                dataset = tdu.load_json_dataset("../data/leclair_java/test.json")
                dataset = random.sample(dataset, 8000)
                summaries = [ex[0] for ex in dataset]
            else:
                raise Exception()
            summary = ir.ir(code, summaries)

        else:
            raise Exception()

        if self.language == "Python":
            summary = tdu.postprocess_edinburgh_format(summary)

        self.output.delete(1.0, END)
        self.output.insert(END, summary)

    def bvae_summarize(self):
        self.summarize("bvae")

    def baseline_summarize(self):
        self.summarize("baseline")

    def retrieve(self, bvae_or_baseline):
        if not self.get_language():
            return

        summary = self.input.get(1.0, END)

        if self.language == "Python":
            summary = tdu.preprocess_user_generated_python(summary)

        if bvae_or_baseline == "bvae":
            model = self.get_model("ret", self.language)
            if model is None:
                return
            ranked_options = model.rank_options(summary)
            code = model.raw_codes[ranked_options[0]]

        elif bvae_or_baseline == "baseline":
            if self.language == "C#":
                summaries, codes = tdu.load_iyer_file("../data/iyer_csharp/test.txt")
            elif self.language == "Python":
                _, _, test = tdu.load_edinburgh_dataset("../data/edinburgh_python")
                summaries = [ex[0] for ex in test]
                codes = [ex[1] for ex in test]
            elif self.language == "Java":
                dataset = tdu.load_json_dataset("../data/leclair_java/test.json")
                dataset = random.sample(dataset, 8000)
                summaries = [ex[0] for ex in dataset]
                codes = [ex[1] for ex in dataset]
            else:
                raise Exception()
            result_index = retir.retir(summary, summaries, 1)[0]
            code = codes[result_index]

        else:
            raise Exception()

        if self.language == "Python":
            code = tdu.postprocess_edinburgh_format(code)

        self.output.delete(1.0, END)
        self.output.insert(END, code)

    def bvae_retrieve(self):
        self.retrieve("bvae")

    def baseline_retrieve(self):
        self.retrieve("baseline")

    def get_language(self):
        self.language = self.language_box.get()
        if self.language not in self.AVAILABLE_LANGUAGES:
            self.output.delete(1.0, END)
            self.output.insert(END, "Error: Please select a language")
            return False
        else:
            return True

    def get_model(self, type, language):
        if (type, language) in self.models:
            model = self.models[(type, language)]
        else:
            self.output.delete(1.0, END)
            self.output.insert(END, "Model loading...")
            self.master.update_idletasks()
            try:
                # time.sleep(5)
                model = ProductionBVAE("../models/%s_%s_prod" % (type, language))
                if type == "ret":
                    model = self.wrap(model, language)
                self.models[(type, language)] = model
            except FileNotFoundError:
                self.output.delete(1.0, END)
                self.output.insert(END, "Error: Model could not be loaded")
                return None

        return model

    def wrap(self, model, language):
        if language == "C#":
            _, codes = tdu.load_iyer_file("../data/iyer_csharp/test.txt")
        elif language == "Python":
            _, _, test = tdu.load_edinburgh_dataset("../data/edinburgh_python")
            codes = [ex[1] for ex in test]
        elif language == "Java":
            dataset = tdu.load_json_dataset("../data/leclair_java/test.json")
            random.seed(a=420)
            subset = random.sample(dataset, 8000)
            codes = [ex[1] for ex in subset]

        wrapped_model = RetBVAE(model, codes)
        return wrapped_model


root = Tk()
gui = GUI(root)
root.mainloop()
