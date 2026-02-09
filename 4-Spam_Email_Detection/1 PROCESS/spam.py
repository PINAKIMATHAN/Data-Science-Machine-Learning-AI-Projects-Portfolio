import chardet
from tkinter import *
import nltk
nltk.download("punkt")
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split as ttsplit
from sklearn import svm
import pandas as pd
import pickle
import numpy as np

file = "C:/Users/saswa/OneDrive/Desktop/Pinaki_Spam_Email_Detection/Spam Email Detection-spam.csv"
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

df = pd.read_csv(file, encoding='Windows-1252')
message_X = df.iloc[:, 1]  
labels_Y = df.iloc[:, 0]  

lstem = LancasterStemmer()


def mess(messages):
  message_x = []
  for me_x in messages:
    me_x = ''.join(filter(lambda mes: (mes.isalpha() or mes == " "), me_x))
    words = word_tokenize(me_x)
    message_x += [' '.join([lstem.stem(word) for word in words])]
  return message_x


message_x = mess(message_X)
tfvec = TfidfVectorizer(stop_words='english')
x_new = tfvec.fit_transform(message_x).toarray()

y_new = np.array(labels_Y.replace(to_replace=['ham', 'spam'], value=[0, 1]))

x_train, x_test, y_train, y_test = ttsplit(
    x_new, y_new, test_size=0.2, shuffle=True)
classifier = svm.SVC()
classifier.fit(x_train, y_train)

pickle.dump({'classifier': classifier, 'message_x': message_x},
            open("training_data.pkl", "wb"))



BG_COLOR = "#89CFF0"
FONT_BOLD = "Melvetica %d bold"


class SpamHam:
    def __init__(self):
        self.window = Tk()
        self.main_window()
        self.lstem = LancasterStemmer()
        self.tfvec = TfidfVectorizer(stop_words='english')
        self.datafile()

    def datafile(self):
        datafile = pickle.load(open("training_data.pkl", "rb"))
        self.message_x = datafile["message_x"]
        self.classifier = datafile["classifier"]


    def main_window(self):
        self.window.title("Spam Detector")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=520, height=400, bg=BG_COLOR)

        head_label = Label(self.window, bg="#fcd386", fg="#fa0000", text="Welcome to Pinaki Email Spam Detector", font=FONT_BOLD % (14), pady=10)
        head_label.place(relwidth=1)
        line = Label(self.window, width=200, bg="#000")
        line.place(relwidth=0.5, relx=0.25, rely=0.08, relheight=0.008)

        mid_label = Label(self.window, bg="#6c9bbd", fg="#020238", text="Spam Or Ham ? Message Detector", font=FONT_BOLD % (18), pady=10)
        mid_label.place(relwidth=1, rely=0.12)

        self.answer = Label(self.window, bg="#cdfa8e", fg="#000", text="Please type message below.", font=FONT_BOLD % (16), pady=10, wraplength=525)
        self.answer.place(relwidth=1, rely=0.30)

        self.msg_entry = Text(self.window, bg="#faaaaa",
                            fg="#fa4646", font=FONT_BOLD % (14))
        self.msg_entry.place(relwidth=1, relheight=0.4, rely=0.48)
        self.msg_entry.focus()

        check_button = Button(self.window, text="Check",
                            font=FONT_BOLD % (12), width=8, bg="#8efa84", fg="#5c0202",
                            command=lambda: self.on_enter(None))
        check_button.place(relx=0.40, rely=0.90, relheight=0.08, relwidth=0.20)


    def bow(self, message):
        mess_t = self.tfvec.fit(self.message_x)
        message_test = mess_t.transform(message).toarray()
        return message_test

    def mess(self,messages):
        message_x = []           
        for me_x in messages:
            me_x=''.join(filter(lambda mes:(mes.isalpha() or mes==" ") ,me_x))
            words = word_tokenize(me_x)
            message_x+=[' '.join([self.lstem.stem(word) for word in words])]
        return message_x

    def on_enter(self,event):
        msg=str(self.msg_entry.get("1.0","end"))
        message=self.mess([msg])
        self.answer.config(fg="#ff0000",text="Your message is : "+
                            ("spam" if self.classifier.predict(self.bow(message)).reshape(1,-1)
                            else "ham"))

    def run(self):
        self.window.mainloop()


app = SpamHam()
app.run()
