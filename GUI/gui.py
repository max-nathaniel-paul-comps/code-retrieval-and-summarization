import sys
from tkinter import *
sys.path.append("../src")
from prod_bvae import ProductionBVAE


class GUI:
    def __init__(self, master):
        self.master = master

        # self.label = Label(master, text="Code Retrieval and Summarization")
        # self.label.pack()

        self.input = Text(master)
        self.input.pack()

        self.summary_button_frame = Frame(root)
        self.summary_button_frame.pack()

        self.retrieval_button_frame = Frame(root)
        self.retrieval_button_frame.pack()

        self.summarize_bt = Button(self.summary_button_frame, text="Generate Summary", command=self.summarize)
        self.summarize_bt.pack(side=LEFT)
        self.baseline_summarize_bt = Button(self.summary_button_frame, text="Generate Summary with Baseline Model", command=self.baseline_summarize)
        self.baseline_summarize_bt.pack(side=LEFT)

        self.retrieve_bt = Button(self.retrieval_button_frame, text="Retrieve Code", command=self.retrieve)
        self.retrieve_bt.pack(side=LEFT)
        self.baseline_retrieve_bt = Button(self.retrieval_button_frame, text="Retrieve Code with Baseline Model", command=self.baseline_retrieve)
        self.baseline_retrieve_bt.pack(side=LEFT)

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()

        self.output = Text(master)
        self.output.pack()
        print("The Summarization BVAE is initializing, please wait...")
        self.summarization_bvae = ProductionBVAE("../models/summ_sw_2_prod")

    def greet(self):
        print("Greetings!")

    def summarize(self):
        self.output.delete(1.0, END)
        # self.output.insert(END, "Error: Unable to produce a summary")
        code = self.input.get(1.0, END)
        summary = self.summarization_bvae.summarize([code], beam_width=1)[0]
        self.output.insert(END, summary)

    def baseline_summarize(self):
        self.output.delete(1.0, END)
        self.output.insert(END, "Error: Unable to produce a summary with Baseline")

    def retrieve(self):
        self.output.delete(1.0, END)
        self.output.insert(END, "def success():\n\tprint(success)")

    def baseline_retrieve(self):
        self.output.delete(1.0, END)
        self.output.insert(END, "Error: Unable to retrieve code with Baseline")


root = Tk()
gui = GUI(root)
root.mainloop()
