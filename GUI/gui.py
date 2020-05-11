import sys
from tkinter import *
from tkinter import ttk
sys.path.append("../src")
from prod_bvae import ProductionBVAE
from ret_bvae import RetBVAE
import text_data_utils as tdu


class GUI:
    def __init__(self, master):
        self.master = master
        self.AVAILABLE_LANGUAGES = ['c#', 'Python', 'Java']

        # self.label = Label(master, text="Code Retrieval and Summarization")
        # self.label.pack()

        self.language_box = ttk.Combobox(master, values=self.AVAILABLE_LANGUAGES)
        self.language_box.pack()

        self.input = Text(master)
        self.input.config(font=('Arial', 13))
        self.input.pack()

        self.summary_button_frame = Frame(root)
        self.summary_button_frame.pack()

        self.retrieval_button_frame = Frame(root)
        self.retrieval_button_frame.pack()

        self.summarize_bt = Button(self.summary_button_frame, text="Generate Summary", command=self.summarize)
        self.summarize_bt.pack(side=LEFT)
        self.baseline_summarize_bt = Button(self.summary_button_frame, text="Generate Summary with Baseline Model",
                                            command=self.baseline_summarize)
        self.baseline_summarize_bt.pack(side=LEFT)

        self.retrieve_bt = Button(self.retrieval_button_frame, text="Retrieve Code", command=self.retrieve)
        self.retrieve_bt.pack(side=LEFT)
        self.baseline_retrieve_bt = Button(self.retrieval_button_frame, text="Retrieve Code with Baseline Model",
                                           command=self.baseline_retrieve)
        self.baseline_retrieve_bt.pack(side=LEFT)

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()

        self.output = Text(master)
        self.output.pack()
        self.output.config(font=('Arial', 13))

        print("The Summarization BVAE is initializing, please wait...")
        self.summarization_bvae = ProductionBVAE("../models/summ_sw_2_prod")

    def summarize(self):
        if not self.get_language():
            return
        self.output.delete(1.0, END)
        # self.output.insert(END, "Error: Unable to produce a summary")
        code = self.input.get(1.0, END)
        summary = self.summarization_bvae.summarize([code], beam_width=1)[0]
        self.output.insert(END, summary)

    def baseline_summarize(self):
        if not self.get_language():
            return
        self.output.delete(1.0, END)
        self.output.insert(END, "Error: Unable to produce a summary with Baseline")

    def retrieve(self):
        if not self.get_language():
            return

        if self.language == "C#":
            _, codes = tdu.load_iyer_file("../data/iyer_csharp/dev.txt")
        elif self.language == "Python":
            _, _, test = tdu.load_edinburgh_dataset("../data/edinburgh_python")
            codes = [ex[1] for ex in test]
        elif self.language == "Java":
            dataset = tdu.load_json_dataset("../data/leclair_java/test.json")
            codes = [ex[1] for ex in dataset]

        ret_bvae = RetBVAE(self.summarization_bvae, codes)
        ranked_options = ret_bvae.rank_options(self.input.get(1.0, END))
        result = ret_bvae.raw_codes[ranked_options[0]]

        self.output.delete(1.0, END)
        self.output.insert(END, result)

    def baseline_retrieve(self):
        if not self.get_language():
            return
        self.output.delete(1.0, END)
        self.output.insert(END, "Error: Unable to retrieve code with Baseline")

    def get_language(self):
        self.language = self.language_box.get()
        if self.language not in self.AVAILABLE_LANGUAGES:
            self.output.delete(1.0, END)
            self.output.insert(END, "Error: Please select a language")
            return False
        else:
            return True


root = Tk()
gui = GUI(root)
root.mainloop()
