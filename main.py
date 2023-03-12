import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from lib import cleansing, remove_punctuation

import tkinter as tk
from tkinter import *
import tkinter
from PIL import ImageTk, Image

def prediction_fun(sentence):
    file_dataset = 'datasets/shopee_reviews.csv'
    df = pd.read_csv(file_dataset)

    df = df[['userName','reviewId','content','score']]
    df = df.dropna(subset=['content'])

    review = []
    for index, row  in df.iterrows():
        review.append(cleansing(row['content']))

    df['content'] = review

    train_df, test_df = train_test_split(df, test_size = 0.2, random_state = 42)

    # load model from single file
    model_test = load_model('output_model/lstm_model.h5')

    sentences = []

    sentences.append(sentence)

    temp= []
    for i in sentences:
        temp.append(remove_punctuation(i))

    sentences = temp

    top_words = 20000
    tokenizer = Tokenizer(num_words=top_words)
    tokenizer.fit_on_texts(train_df['content'])

    tokenized_test = tokenizer.texts_to_sequences(sentences)
    X_test = pad_sequences(tokenized_test, maxlen=200)
    prediction = model_test.predict(X_test)
    print(prediction)
    pred_labels = []
    for i in prediction:
        if i > 0.5:
            pred_labels.append(1)
        else:
            pred_labels.append(0)
            
    for i in range(len(sentences)):
        print(sentences[i])
        if pred_labels[i] == 1:
            s = 'Positive'
        else:
            s = 'Negative'
    
    return s


def s_exit():
    exit(0)


def putwindow():

    window = Tk()
    window.geometry("800x550")
    window.configure(background='#2A8D87')
    window.title("Kelompok 5 Metodologi Penelitian L1")
    window.geometry("800x500")
    window.resizable(False, False)
    tkinter.Label(window, text="\nSentiment Analysis for Shopee App Rating Review \non Google Play Store Using LSTM Method", font=(
        "Poppins", 15, "bold"), fg="white", bg="#2A8D87").pack(fill="x")

    # Label Input
    label1 = tk.Label(window, text="Input :", font=(
        "Poppins", 12, "bold"), fg="white", bg="#2A8D87")
    label1.place(relx=0.077, rely=0.27)

    # Label Output/Result
    label2 = tk.Label(window, text="Output :", font=(
        "Poppins", 12, "bold"), fg="white", bg="#2A8D87")
    label2.place(relx=0.077, rely=0.74)

    # Frame Atas
    frame1 = tk.Frame(window,  bg='#2A8D87', bd=1)
    frame1.place(relx=0.33, rely=0.33, relwidth=0.5,
                relheight=0.3, anchor='n')
    
    # Text Input
    textbox = tk.Entry(frame1,  font=("Poppins", 12))
    textbox.place(relwidth=1, relheight=1)

    # Frame Bawah
    frame2 = tk.Frame(window, bg='#2A8D87', bd=1)
    frame2.place(relx=0.23, rely=0.8, relwidth=0.3,
                relheight=0.1, anchor='n')
    
    # Text Output
    bg_color = 'white'
    results = tk.Label(frame2, anchor='nw', justify='left', bd=1)
    results.config(font=40, bg=bg_color)
    results.place(relwidth=1, relheight=1)

    # Function
    def analyse():
        text = textbox.get()
        result = prediction_fun(text)
        results.config(text=result)

    def clear_input():
        textbox.delete(0, END)
        results.config(text="")
        

    # Button
    btn1 = tkinter.Button(window, text="ANALYZE", fg="black",
                        bg='#F1F1EF', command=analyse)
    btn1.place(relx=0.082, rely=0.65, width=90)

    btn2 = tkinter.Button(window, text="CLEAR",
                        fg="black", bg='#F1F1EF', command=clear_input)
    btn2.place(relx=0.210, rely=0.65, width=90)

    btn4 = tkinter.Button(window, text="EXIT", fg="black",
                        bg='#F1F1EF', command=s_exit)
    btn4.place(relx=0.41, rely=0.82, width=90)

    # gambar
    img = Image.open("gambar1.png")
    img = img.resize((250, 250))
    img = ImageTk.PhotoImage(img)

    # Membuat canvas
    canvas = tk.Canvas(window, width=250, height=250,
                    bg="#2A8D87", highlightthickness=0)
    canvas.place(relx=0.65, rely=0.47)

    # Menambahkan gambar ke canvas
    canvas.create_image(0, 0, anchor="nw", image=img)

    window.mainloop()


if __name__ == "__main__":
    # show_create()
    putwindow()